#!/usr/bin/env python3
"""gpu_burn_ascend.py -- 刷昇腾 NPU 的 HBM 与 AI Core 利用率（torch_npu 版）。

只用 torch + torch_npu，不用编译、不要 CANN 头文件：

    python3 gpu_burn_ascend.py --devices all --duration 600 --stats 5

行为和同目录的 C 版 gpu_burn_ascend 对齐：一个进程，每颗芯片一个线程，
Ctrl-C / SIGTERM 时各芯片都会释放 HBM 再退出（进程一死驱动就回收）。
算力用 torch.mm 反复喂 AI Core；实测 TFLOPS 按 2*M*N*K*次数/耗时算，不猜理论峰值。
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
            print("[gpu_burn_ascend %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)
        except BrokenPipeError:  # 被 | head 之类截断时安静退出
            pass


def die(msg, code=2):
    print("[gpu_burn_ascend] ERR: %s" % msg, file=sys.stderr, flush=True)
    sys.exit(code)


def load_torch():
    try:
        import torch
    except ImportError:
        die("找不到 torch（pip install torch；昇腾机器上还要 torch_npu）")
    try:
        import torch_npu  # noqa: F401  只为注册 torch.npu 后端
    except ImportError:
        die("找不到 torch_npu（它随 CANN 提供，确认 PYTHONPATH 里有 torch_npu）")
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
            raise ValueError("--devices 里的 %d 超出范围（本机共 %d 颗芯片）" % (dev, count))
        if dev not in devs:
            devs.append(dev)
    return devs


def norm_pct(v, quiet):
    if 1 < v <= 100:
        log("--mem-pct %g > 1，按百分比理解（%g%%）；下次可以直接写 %.2f" % (v, v, v / 100), quiet)
        return v / 100
    if not 0 < v <= 1:
        die("--mem-pct 必须在 (0, 1] 之间（0.6 = 60%，和 gpu_burn 一样）")
    return v


def self_check(torch, dev, quiet):
    torch.npu.set_device(dev)
    name = "?"
    try:
        name = torch.npu.get_device_name(dev)
    except Exception:
        pass
    a = torch.ones(64, 64, dtype=torch.float32, device="npu:%d" % dev)
    c = torch.mm(a, a)
    torch.npu.synchronize()
    got = float(c[0][0])
    if abs(got - 64.0) > 1e-3:
        die("自检失败：matmul 结果不对（期望 64，实际 %g）" % got)
    log("设备 #%d: %s" % (dev, name), quiet)
    log("自检通过：torch_npu → AI Core → 显存读写这条链路正常", quiet)


def hbm_info(torch, dev):
    """返回 (空闲, 总量)；老版本 torch_npu 没有 mem_get_info 时退回设备属性。"""
    try:
        return torch.npu.mem_get_info(dev)
    except (AttributeError, TypeError):
        total = torch.npu.get_device_properties(dev).total_memory
        return total, total


def alloc_hbm(torch, dev, want, quiet, tag):
    chunks, got = [], 0
    while got < want:
        n = min(1 * GIB, want - got)
        while n >= 32 << 20:
            try:
                chunks.append(torch.zeros(n, dtype=torch.uint8, device="npu:%d" % dev))
                got += n
                break
            except Exception as exc:
                log("%s分不到 %.1f GiB（%s），减半重试"
                    % (tag, n / GIB, str(exc).splitlines()[0][:70]), quiet)
                n //= 2
        else:
            break
    return chunks, got


def chip_worker(torch, dev, args, stop, stats, lock):
    tag = "#%d " % dev if args.ndev > 1 else ""
    chunks = []
    try:
        torch.npu.set_device(dev)
        free, total = hbm_info(torch, dev)
        s, dt = args.size, getattr(torch, DTYPES[args.dtype])
        reserve = 3 * s * s * ITEMSIZE[args.dtype] + (128 << 20)
        want = max(0, int((free - reserve) * args.mem_pct))
        chunks, got = alloc_hbm(torch, dev, want, args.quiet, tag)
        with lock:
            stats[dev].update(total=total, hbm=got)
        log("%sHBM 已占 %.2f GiB/%.2f GiB" % (tag, got / GIB, total / GIB), args.quiet)
        a = torch.randn(s, s, dtype=dt, device="npu:%d" % dev)
        b = torch.randn(s, s, dtype=dt, device="npu:%d" % dev)
        out = torch.empty(s, s, dtype=dt, device="npu:%d" % dev)
        with lock:
            stats[dev]["flops_per_call"] = 2.0 * s * s * s
        while not stop.is_set():
            for _ in range(args.queue):
                torch.mm(a, b, out=out)
                with lock:
                    stats[dev]["calls"] += 1
            torch.npu.synchronize()
    except Exception as exc:
        with lock:
            stats[dev]["error"] = str(exc).splitlines()[0][:120]
        log("%s出错: %s" % (tag, stats[dev]["error"]), args.quiet)
    finally:
        del chunks[:]
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
        log("%s已释放 HBM 与设备上下文" % tag, args.quiet)


def print_stats(stats, devs, elapsed, quiet):
    calls = sum(stats[d]["calls"] for d in devs)
    flops = sum(stats[d]["calls"] * stats[d]["flops_per_call"] for d in devs)
    hbm = " ".join("#%d %.2f GiB" % (d, stats[d]["hbm"] / GIB) for d in devs)
    log("运行 %.1fs | MatMul 发射 %d 次 | 累计 %.1f TFLOP | 实测 %.1f TFLOPS | HBM %s"
        % (elapsed, calls, flops / 1e12, flops / max(elapsed, 1e-9) / 1e12, hbm), quiet)


def main():
    p = argparse.ArgumentParser(description="刷昇腾 NPU 的 HBM 与 AI Core 利用率（torch_npu 版）")
    p.add_argument("--devices", default="0", help="要刷的芯片：0 / 0,1,2 / all（默认 0）")
    p.add_argument("--mem-pct", type=float, default=0.60, help="占每颗芯片空闲 HBM 的比例（默认 0.60；写 60 也认）")
    p.add_argument("--size", type=int, default=8192, help="MatMul 的 M=N=K（默认 8192）")
    p.add_argument("--dtype", default="fp16", choices=sorted(DTYPES), help="MatMul 数据类型（默认 fp16）")
    p.add_argument("--queue", type=int, default=4, help="每轮异步提交几次再同步（默认 4）")
    p.add_argument("--duration", type=float, default=0, help="跑多少秒后退出（默认 0 = 一直跑）")
    p.add_argument("--stats", type=float, default=5, help="多少秒报一次（默认 5）")
    p.add_argument("--check", action="store_true", help="只做自检，不刷")
    p.add_argument("--quiet", action="store_true", help="只打统计行和错误")
    p.add_argument("--version", action="version", version="gpu_burn_ascend %s (python)" % VERSION)
    args = p.parse_args()

    torch = load_torch()
    try:
        count = torch.npu.device_count()
    except Exception as exc:
        die("问不到 NPU 数量（%s）：确认驱动和 CANN 正常，npu-smi info 能看到芯片" % exc, 3)
    if count <= 0:
        die("看不到任何 NPU（torch.npu.device_count()=%d）" % count, 3)
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
    log("开始刷: 芯片 %s，MatMul %d^3 %s，目标 HBM 比例 %.0f%%"
        % (devs, args.size, args.dtype, args.mem_pct * 100), args.quiet)
    stats = {d: {"calls": 0, "hbm": 0, "total": 0, "flops_per_call": 0.0, "error": None} for d in devs}
    lock = threading.Lock()
    threads = [threading.Thread(target=chip_worker, args=(torch, d, args, stop, stats, lock), daemon=True)
               for d in devs]
    t_start = t0 = time.time()
    for t in threads:
        t.start()
    deadline = t0 + args.duration if args.duration > 0 else None
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
        t.join(timeout=30)
    print_stats(stats, devs, time.time() - t_start, args.quiet)

    bad = [d for d in devs if stats[d]["error"]]
    if bad:
        log("有 %d 颗芯片出错（见上面带 #N 的行）" % len(bad), args.quiet)
        return 3
    log("全部 %d 颗芯片已收工，HBM 都已释放" % len(devs), args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
