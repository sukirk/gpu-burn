#!/usr/bin/env python3
"""gpu_burn_torch.py -- 刷 NVIDIA GPU 的显存与算力利用率（PyTorch 版）。

原理和 gpu_burn_ascend.py 一样：不自己写 kernel，而是用 PyTorch 的 matmul（cuBLAS）反复喂计算单元，
对监控面板来说这就是"真实的高利用率负载"（cuBLAS 是厂商调过的 GEMM，FP32 走 FFMA 打满 FP32 管线，
fp16/bf16 走 Tensor Core）：

    python3 gpu_burn_torch.py --devices all --duration 600 --stats 5

一个进程，每张卡一个线程；Ctrl-C / SIGTERM 时每张卡都释放显存再退出。
统计行报实测 TFLOPS（2*M*N*K*次数/耗时），可以拿它对照面板数字判断是不是口径差异。
"""

import argparse
import signal
import sys
import threading
import time

VERSION = "1.3.0"
MAX_DEVICES = 32
GIB = 1 << 30
DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
ITEMSIZE = {"fp16": 2, "bf16": 2, "fp32": 4}


def log(msg, quiet=False):
    if not quiet:
        try:
            print("[gpu_burn_torch %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)
        except BrokenPipeError:
            pass


def die(msg, code=2):
    print("[gpu_burn_torch] ERR: %s" % msg, file=sys.stderr, flush=True)
    sys.exit(code)


def load_torch():
    try:
        import torch
    except ImportError:
        die("找不到 torch（pip install torch 即可；只要纯 C 版请用 ./gpu_burn）")
    if not torch.cuda.is_available():
        die("torch 在，但看不到 CUDA 设备：容器要加 --gpus all，宿主要装 NVIDIA 驱动", 3)
    return torch


def resolve_devices(spec, count):
    if spec.strip().lower() == "all":
        return list(range(min(count, MAX_DEVICES)))
    devs = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            dev = int(part)
        except ValueError:
            raise ValueError("--devices 解析不了: %r（写 0,1,2 或 all）" % spec)
        if dev < 0 or dev >= count:
            raise ValueError("--devices 里的 %d 超出范围（本机共 %d 张卡）" % (dev, count))
        if dev not in devs:
            devs.append(dev)
    return devs


def norm_pct(v, quiet):
    if 1 < v <= 100:
        log("--mem-pct %g > 1，按百分比理解（%g%%）；下次可以直接写 %.2f" % (v, v, v / 100), quiet)
        return v / 100
    if not 0 < v <= 1:
        die("--mem-pct 必须在 (0, 1] 之间（0.6 = 60%）")
    return v


def self_check(torch, dev, quiet):
    torch.cuda.set_device(dev)
    log("设备 #%d: %s" % (dev, torch.cuda.get_device_name(dev)), quiet)
    a = torch.ones(128, 128, dtype=torch.float32, device="cuda:%d" % dev)
    got = float(torch.mm(a, a)[0][0])
    if abs(got - 128.0) > 1e-3:
        die("自检失败：matmul 结果不对（期望 128，实际 %g）" % got)
    log("自检通过：PyTorch → cuBLAS → 显存这条链路正常", quiet)


def mem_info(torch, dev):
    """返回 (空闲, 总量)；老版本 torch 没有 mem_get_info 时退回设备属性。"""
    try:
        return torch.cuda.mem_get_info(dev)
    except (AttributeError, TypeError):
        total = torch.cuda.get_device_properties(dev).total_memory
        return total, total


def alloc_vram(torch, dev, want, quiet, tag):
    chunks, got = [], 0
    while got < want:
        n = min(1 * GIB, want - got)
        while n >= 32 << 20:
            try:
                chunks.append(torch.zeros(n, dtype=torch.uint8, device="cuda:%d" % dev))
                got += n
                break
            except Exception as exc:
                if "out of memory" not in str(exc).lower() and "memory" not in str(exc).lower():
                    raise
                log("%s分不到 %.1f GiB，减半重试" % (tag, n / GIB), quiet)
                n //= 2
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            break
    return chunks, got


def chip_worker(torch, dev, args, stop, stats, lock):
    tag = "#%d " % dev if args.ndev > 1 else ""
    try:
        torch.cuda.set_device(dev)
        free, total = mem_info(torch, dev)
        s, dt = args.size, getattr(torch, DTYPES[args.dtype])
        reserve = 3 * s * s * ITEMSIZE[args.dtype] + (256 << 20)
        want = max(0, int((free - reserve) * args.mem_pct))
        chunks, got = alloc_vram(torch, dev, want, args.quiet, tag)
        with lock:
            stats[dev].update(total=total, hbm=got)
        log("%s显存已占 %.2f GiB/%.2f GiB" % (tag, got / GIB, total / GIB), args.quiet)
        a = torch.randn(s, s, dtype=dt, device="cuda:%d" % dev)
        b = torch.randn(s, s, dtype=dt, device="cuda:%d" % dev)
        out = torch.empty(s, s, dtype=dt, device="cuda:%d" % dev)
        with lock:
            stats[dev]["flops_per_call"] = 2.0 * s * s * s
        while not stop.is_set():
            for _ in range(args.queue):
                torch.mm(a, b, out=out)
                with lock:
                    stats[dev]["calls"] += 1
            torch.cuda.synchronize()
        del chunks[:]
    except Exception as exc:
        with lock:
            stats[dev]["error"] = str(exc).splitlines()[0][:120]
        log("%s出错: %s" % (tag, stats[dev]["error"]), args.quiet)
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        log("%s已释放显存与 CUDA 上下文" % tag, args.quiet)


def print_stats(stats, devs, elapsed, quiet):
    calls = sum(stats[d]["calls"] for d in devs)
    flops = sum(stats[d]["calls"] * stats[d]["flops_per_call"] for d in devs)
    vram = " ".join("#%d %.2f GiB" % (d, stats[d]["hbm"] / GIB) for d in devs)
    log("运行 %.1fs | matmul 发射 %d 次 | 累计 %.1f TFLOP | 实测 %.1f TFLOPS | 显存 %s"
        % (elapsed, calls, flops / 1e12, flops / max(elapsed, 1e-9) / 1e12, vram), quiet)


def main():
    p = argparse.ArgumentParser(description="刷 NVIDIA GPU 的显存与算力利用率（PyTorch + cuBLAS 版）")
    p.add_argument("--devices", default="0", help="要刷的卡：0 / 0,1,2 / all（默认 0）")
    p.add_argument("--mem-pct", type=float, default=0.60, help="占每张卡空闲显存的比例（默认 0.60；写 60 也认）")
    p.add_argument("--size", type=int, default=8192, help="matmul 的 M=N=K（默认 8192）")
    p.add_argument("--dtype", default="fp16", choices=sorted(DTYPES),
                   help="matmul 数据类型：fp16/bf16 走 Tensor Core，fp32 走 FP32 管线（默认 fp16）")
    p.add_argument("--queue", type=int, default=4, help="每轮异步提交几次再同步（默认 4）")
    p.add_argument("--duration", type=float, default=0, help="跑多少秒后退出（默认 0 = 一直跑）")
    p.add_argument("--stats", type=float, default=5, help="多少秒报一次（默认 5）")
    p.add_argument("--check", action="store_true", help="只做自检，不刷")
    p.add_argument("--quiet", action="store_true", help="只打统计行和错误")
    p.add_argument("--version", action="version", version="gpu_burn_torch %s (python)" % VERSION)
    args = p.parse_args()

    torch = load_torch()
    count = torch.cuda.device_count()
    if count <= 0:
        die("看不到任何 GPU（torch.cuda.device_count()=%d）" % count, 3)
    log("可见设备数: %d" % count, args.quiet)
    try:
        devs = resolve_devices(args.devices, count)
    except ValueError as exc:
        die(str(exc), 3)
    args.ndev, args.mem_pct = len(devs), norm_pct(args.mem_pct, args.quiet)

    if args.check:
        for dev in devs:
            self_check(torch, dev, args.quiet)
        return 0

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    log("开始刷: 卡 %s，matmul %d^3 %s，目标显存比例 %.0f%%"
        % (devs, args.size, args.dtype, args.mem_pct * 100), args.quiet)
    stats = {d: {"calls": 0, "hbm": 0, "total": 0, "flops_per_call": 0.0, "error": None} for d in devs}
    lock = threading.Lock()
    threads = [threading.Thread(target=chip_worker, args=(torch, d, args, stop, stats, lock), daemon=True)
               for d in devs]
    t_start = t0 = time.time()
    for t in threads:
        t.start()
    deadline = t_start + args.duration if args.duration > 0 else None
    last_stats = t0
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
            now = time.time()
            if deadline and now >= deadline and not stop.is_set():
                log("到达 --duration %gs，主动收工" % args.duration, args.quiet)
                stop.set()
            if now - last_stats >= args.stats and not stop.is_set():
                print_stats(stats, devs, now - t_start, args.quiet)
                last_stats = now
    except KeyboardInterrupt:
        stop.set()
    stop.set()
    for t in threads:
        t.join(timeout=60)
    print_stats(stats, devs, time.time() - t_start, args.quiet)

    bad = [d for d in devs if stats[d]["error"]]
    if bad:
        log("有 %d 张卡出错（见上面带 #N 的行）" % len(bad), args.quiet)
        return 3
    log("全部 %d 张卡已收工，显存都已释放" % len(devs), args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
