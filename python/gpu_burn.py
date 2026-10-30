#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_burn -- 占满 CUDA 显存，并把 GPU 利用率刷到接近 100%。

只依赖 Python 3 标准库（ctypes），不需要任何 pip 包，也不需要 CUDA Toolkit：
  * 直接 dlopen("libcuda.so.1") 调 CUDA Driver API，这个库由宿主机的 NVIDIA 驱动提供；
  * 计算 kernel 用内联 PTX 文本，交给驱动自带的 JIT 现场编译成目标架构的机器码。

用法见 --help，典型：
    python3 gpu_burn.py                 # 占 60% 空闲显存，一直刷，每 5 秒报一次
    python3 gpu_burn.py --mem-gb 70     # 占 70GiB
    python3 gpu_burn.py --duration 600  # 刷 10 分钟后自动退出
    python3 gpu_burn.py --check         # 只做环境自检，不常驻
"""

from __future__ import print_function

import argparse
import ctypes
import os
import signal
import sys
import threading
import time
from ctypes import (POINTER, Structure, byref, c_char_p, c_float, c_int,
                    c_size_t, c_uint, c_uint32, c_uint64, c_void_p)

VERSION = "1.0.1"

CUDA_SUCCESS = 0
CUDA_ERROR_OUT_OF_MEMORY = 2
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_LAUNCH_FAILED = 719

CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

CU_EVENT_DEFAULT = 0x0

# --------------------------------------------------------------------------- #
# kernel 源码：内联 PTX
#
# 每个线程做一个固定次数的循环，循环体是 4 条互相独立的 FP32 FMA（加一个递减
# 计数器和分支）。选 4 条独立 FMA 是为了让每个 SM 的 4 个 warp 调度器都能被占满：
# 每轮 7 条指令里 4 条是 FMA，正好卡在 FP32 吞吐上，既不空转也不被整数指令拖走。
# 累加器用 a = 0.5*a + seed 的形式，保证数值有界（不会 Inf/NaN），同时因为系数和
# seed 都来自寄存器，ptxas 无法把它折叠掉。循环结束每线程写一次全局内存，
# 防止整个循环被当成死代码删掉。
# --------------------------------------------------------------------------- #

_PTX_HEADER = """\
//
// gpu_burn_kernel -- bounded FP32 FMA spin loop, used only to keep every SM busy.
//
// Kept as PTX text on purpose: the CUDA driver JITs it (via libcuda.so.1), so the
// program needs no CUDA toolkit, no nvcc and no per-architecture cubin.
//
//   params: (u64 out_ptr, u32 iters)
//   each thread: iters * 4 FMA on 4 independent accumulators, then one store so
//   ptxas cannot prove the loop dead.
//
"""

_KERNEL_BODY = """
    .reg .pred  %p1;
    .reg .b32   %r1, %r2;
    .reg .b64   %rd1, %rd2;
    .reg .f32   %f1, %f2, %f3, %f4, %f5, %f6;

    ld.param.u64    %rd1, [out_ptr];
    ld.param.u32    %r1,  [iters];

    mov.u32         %r2, %tid.x;
    cvt.rn.f32.u32  %f5, %r2;           // per-thread seed
    mov.f32         %f1, 0f3F800000;    // 1.0
    mov.f32         %f2, 0f3F800000;
    mov.f32         %f3, 0f3F800000;
    mov.f32         %f4, 0f3F800000;
    mov.f32         %f6, 0f3F000000;    // 0.5

LOOP:
    fma.rn.f32      %f1, %f1, %f6, %f5;
    fma.rn.f32      %f2, %f2, %f6, %f5;
    fma.rn.f32      %f3, %f3, %f6, %f5;
    fma.rn.f32      %f4, %f4, %f6, %f5;
    add.u32         %r1, %r1, -1;
    setp.gt.u32     %p1, %r1, 0;
    @%p1 bra        LOOP;

    add.f32         %f1, %f1, %f2;
    add.f32         %f3, %f3, %f4;
    add.f32         %f1, %f1, %f3;

    mad.wide.u32    %rd2, %r2, 4, %rd1;
    st.global.f32   [%rd2], %f1;
    ret;
"""

_KERNEL_SIG = (
    "\n.visible .entry gpu_burn_kernel(\n"
    "    .param .u64 out_ptr,\n"
    "    .param .u32 iters\n"
    ")\n{\n"
)

# 按顺序尝试：正常情况第一个就能被 JIT，后面几个只是给老/新驱动留退路
# （sm_80 + PTX ISA 7.0 是 CUDA 11.0 起就有的组合，A100 必然满足）。
KERNEL_VARIANTS = (
    ("sm_80 / PTX ISA 7.0", _PTX_HEADER + ".version 7.0\n.target sm_80\n.address_size 64\n" + _KERNEL_SIG + _KERNEL_BODY + "}\n"),
    ("sm_80 / PTX ISA 8.0", _PTX_HEADER + ".version 8.0\n.target sm_80\n.address_size 64\n" + _KERNEL_SIG + _KERNEL_BODY + "}\n"),
    ("sm_70 / PTX ISA 6.4", _PTX_HEADER + ".version 6.4\n.target sm_70\n.address_size 64\n" + _KERNEL_SIG + _KERNEL_BODY + "}\n"),
)

DRIVER_CANDIDATES = (
    "libcuda.so.1",
    "libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
    "/usr/local/cuda/lib64/libcuda.so.1",
    "/usr/lib/wsl/lib/libcuda.so.1",
)

NVML_CANDIDATES = (
    "libnvidia-ml.so.1",
    "libnvidia-ml.so",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    "/usr/lib64/libnvidia-ml.so.1",
)

DRIVER_HINT = """\
libcuda.so.1 没能加载。这个库由宿主机的 NVIDIA 驱动提供，不在镜像里，需要容器
以 GPU 方式启动才会被注入：

    docker run --gpus all -it <image>              # 需要宿主机装 nvidia-container-toolkit
    docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ...   # 老版本 docker

自检顺序：
  1) ls -l /dev/nvidia*                          # 有没有设备节点
  2) ldconfig -p | grep libcuda                  # 有没有驱动库
  3) nvidia-smi                                  # 宿主机驱动是否正常
"""


class GpuBurnError(RuntimeError):
    """CUDA 调用失败或环境不满足。"""


def _human(nbytes):
    v = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024.0 or unit == "TiB":
            return "%.2f %s" % (v, unit) if unit != "B" else "%d B" % nbytes
        v /= 1024.0
    return "%.2f TiB" % v


# --------------------------------------------------------------------------- #
# CUDA Driver API 封装
# --------------------------------------------------------------------------- #

class Cuda(object):
    def __init__(self, log):
        self.log = log
        self.lib = None
        self.libpath = None
        errors = []
        for cand in DRIVER_CANDIDATES:
            try:
                lib = ctypes.CDLL(cand)
            except OSError as exc:
                errors.append("%s: %s" % (cand, exc))
                continue
            missing = self._bind(lib)
            if missing:
                errors.append("%s: missing symbols: %s" % (cand, ", ".join(missing)))
                continue
            self.lib, self.libpath = lib, cand
            break
        if self.lib is None:
            raise GpuBurnError(DRIVER_HINT + "\n加载尝试：\n  " + "\n  ".join(errors))

    # -- 符号绑定 ---------------------------------------------------------- #
    _PROTOS = {
        "cuInit": ([c_uint], c_int),
        "cuDriverGetVersion": ([POINTER(c_int)], c_int),
        "cuDeviceGetCount": ([POINTER(c_int)], c_int),
        "cuDeviceGet": ([POINTER(c_int), c_int], c_int),
        "cuDeviceGetName": ([c_char_p, c_int, c_int], c_int),
        "cuDeviceTotalMem_v2": ([POINTER(c_size_t), c_int], c_int),
        "cuDeviceGetAttribute": ([POINTER(c_int), c_int, c_int], c_int),
        "cuCtxCreate_v2": ([POINTER(c_void_p), c_uint, c_int], c_int),
        "cuCtxDestroy_v2": ([c_void_p], c_int),
        "cuCtxSynchronize": ([], c_int),
        "cuCtxSetCurrent": ([c_void_p], c_int),
        "cuMemGetInfo_v2": ([POINTER(c_size_t), POINTER(c_size_t)], c_int),
        "cuMemAlloc_v2": ([POINTER(c_uint64), c_size_t], c_int),
        "cuMemFree_v2": ([c_uint64], c_int),
        "cuMemsetD32Async": ([c_uint64, c_uint, c_size_t, c_void_p], c_int),
        "cuMemcpyDtoDAsync_v2": ([c_uint64, c_uint64, c_size_t, c_void_p], c_int),
        "cuModuleLoadData": ([POINTER(c_void_p), c_char_p], c_int),
        "cuModuleUnload": ([c_void_p], c_int),
        "cuModuleGetFunction": ([POINTER(c_void_p), c_void_p, c_char_p], c_int),
        "cuStreamCreate": ([POINTER(c_void_p), c_uint], c_int),
        "cuStreamDestroy_v2": ([c_void_p], c_int),
        "cuStreamSynchronize": ([c_void_p], c_int),
        "cuEventCreate": ([POINTER(c_void_p), c_uint], c_int),
        "cuEventRecord": ([c_void_p, c_void_p], c_int),
        "cuEventSynchronize": ([c_void_p], c_int),
        "cuEventQuery": ([c_void_p], c_int),
        "cuEventElapsedTime": ([POINTER(c_float), c_void_p, c_void_p], c_int),
        "cuEventDestroy_v2": ([c_void_p], c_int),
        "cuGetErrorString": ([c_int, POINTER(c_char_p)], c_int),
        "cuLaunchKernel": ([c_void_p, c_uint, c_uint, c_uint,
                            c_uint, c_uint, c_uint, c_uint, c_void_p,
                            POINTER(c_void_p), POINTER(c_void_p)], c_int),
    }

    _REQUIRED = (
        "cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetAttribute",
        "cuCtxCreate_v2", "cuCtxDestroy_v2", "cuCtxSynchronize",
        "cuMemGetInfo_v2", "cuMemAlloc_v2", "cuMemFree_v2",
        "cuModuleLoadData", "cuModuleGetFunction", "cuModuleUnload",
        "cuLaunchKernel", "cuStreamCreate", "cuStreamSynchronize",
        "cuEventCreate", "cuEventRecord", "cuEventSynchronize",
        "cuEventElapsedTime", "cuGetErrorString",
    )

    def _bind(self, lib):
        missing = []
        for name in self._REQUIRED:
            try:
                getattr(lib, name)
            except AttributeError:
                missing.append(name)
        if missing:
            return missing
        for name, (args, res) in self._PROTOS.items():
            try:
                fn = getattr(lib, name)
            except AttributeError:
                continue  # 可选符号
            fn.argtypes = args
            fn.restype = res
        return []

    # -- 基础调用 ---------------------------------------------------------- #
    def error_str(self, rc):
        """错误码带上数字：驱动对某些码只回 "unknown"，没有数字就无从查起。"""
        try:
            buf = c_char_p()
            if self.lib.cuGetErrorString(rc, byref(buf)) == CUDA_SUCCESS and buf.value:
                return "%s (rc=%d)" % (buf.value.decode("utf-8", "replace"), rc)
        except Exception:
            pass
        return "unknown (rc=%d)" % rc

    def check(self, rc, what):
        if rc != CUDA_SUCCESS:
            raise GpuBurnError("%s -> %s (code %d)" % (what, self.error_str(rc), rc))
        return rc

    def call(self, name, *args):
        rc = getattr(self.lib, name)(*args)
        self.check(rc, name)
        return rc

    # -- 设备与上下文 ------------------------------------------------------ #
    def driver_version(self):
        v = c_int()
        if self.lib.cuDriverGetVersion(byref(v)) != CUDA_SUCCESS:
            return None
        return v.value // 1000, (v.value % 1000) // 10

    def init(self):
        self.check(self.lib.cuInit(0), "cuInit")
        n = c_int()
        self.check(self.lib.cuDeviceGetCount(byref(n)), "cuDeviceGetCount")
        if n.value <= 0:
            raise GpuBurnError(
                "cuInit 成功但设备数为 0（code %d: %s）。通常是容器没拿到 GPU 设备节点，"
                "检查是否以 --gpus all 启动。" % (CUDA_ERROR_NO_DEVICE, self.error_str(CUDA_ERROR_NO_DEVICE)))
        return n.value

    def device(self, ordinal):
        d = c_int()
        self.check(self.lib.cuDeviceGet(byref(d), ordinal), "cuDeviceGet(%d)" % ordinal)
        return d.value

    def device_name(self, dev):
        buf = ctypes.create_string_buffer(256)
        if self.lib.cuDeviceGetName(ctypes.cast(buf, c_char_p), len(buf), dev) == CUDA_SUCCESS:
            return buf.value.decode("utf-8", "replace")
        return "<unknown>"

    def device_total_mem(self, dev):
        n = c_size_t()
        self.check(self.lib.cuDeviceTotalMem_v2(byref(n), dev), "cuDeviceTotalMem")
        return n.value

    def attr(self, dev, which):
        v = c_int()
        rc = self.lib.cuDeviceGetAttribute(byref(v), which, dev)
        if rc != CUDA_SUCCESS:
            return None
        return v.value

    def ctx_create(self, dev):
        ctx = c_void_p()
        self.check(self.lib.cuCtxCreate_v2(byref(ctx), 0, dev), "cuCtxCreate")
        return ctx

    def ctx_set_current(self, ctx):
        return self.call("cuCtxSetCurrent", ctx)

    def ctx_sync(self):
        return self.call("cuCtxSynchronize")

    def ctx_destroy(self, ctx):
        rc = self.lib.cuCtxDestroy_v2(ctx)
        if rc != CUDA_SUCCESS:
            self.log("WARN: cuCtxDestroy -> %s" % self.error_str(rc))

    # -- 显存 -------------------------------------------------------------- #
    def mem_info(self):
        free_, total = c_size_t(), c_size_t()
        self.check(self.lib.cuMemGetInfo_v2(byref(free_), byref(total)), "cuMemGetInfo")
        return free_.value, total.value

    def try_mem_alloc(self, size):
        ptr = c_uint64()
        rc = self.lib.cuMemAlloc_v2(byref(ptr), c_size_t(size))
        return rc, ptr.value

    def mem_free(self, ptr):
        rc = self.lib.cuMemFree_v2(c_uint64(ptr))
        if rc != CUDA_SUCCESS:
            self.log("WARN: cuMemFree -> %s" % self.error_str(rc))

    def memset_d32_async(self, ptr, nwords, stream):
        self.check(self.lib.cuMemsetD32Async(c_uint64(ptr), 0, c_size_t(nwords), stream),
                   "cuMemsetD32Async")

    def memcpy_dtod_async(self, dst, src, nbytes, stream):
        self.check(self.lib.cuMemcpyDtoDAsync_v2(c_uint64(dst), c_uint64(src), c_size_t(nbytes), stream),
                   "cuMemcpyDtoDAsync")

    # -- stream ------------------------------------------------------------ #
    def stream_create(self):
        s = c_void_p()
        self.check(self.lib.cuStreamCreate(byref(s), 0), "cuStreamCreate")
        return s

    def stream_sync(self, stream):
        self.check(self.lib.cuStreamSynchronize(stream), "cuStreamSynchronize")

    def stream_destroy(self, stream):
        rc = self.lib.cuStreamDestroy_v2(stream)
        if rc != CUDA_SUCCESS:
            self.log("WARN: cuStreamDestroy -> %s" % self.error_str(rc))

    # -- module / kernel --------------------------------------------------- #
    def load_kernel(self, out_ptr):
        """依次尝试内联 PTX 变体，返回 (module, function, variant_name)。"""
        failures = []
        for name, ptx in KERNEL_VARIANTS:
            module = c_void_p()
            image = ptx.encode("utf-8")
            rc = self.lib.cuModuleLoadData(byref(module), image)
            if rc != CUDA_SUCCESS:
                failures.append("%s: %s (code %d)" % (name, self.error_str(rc), rc))
                continue
            fn = c_void_p()
            rc = self.lib.cuModuleGetFunction(byref(fn), module, b"gpu_burn_kernel")
            if rc != CUDA_SUCCESS:
                failures.append("%s: cuModuleGetFunction -> %s" % (name, self.error_str(rc)))
                self.lib.cuModuleUnload(module)
                continue
            return module, fn, name
        raise GpuBurnError("PTX JIT 全部失败：\n  " + "\n  ".join(failures))

    def module_unload(self, module):
        rc = self.lib.cuModuleUnload(module)
        if rc != CUDA_SUCCESS:
            self.log("WARN: cuModuleUnload -> %s" % self.error_str(rc))

    def launch(self, fn, grid, block, stream, out_ptr, iters):
        a_out = c_uint64(out_ptr)
        a_iters = c_uint32(iters & 0xFFFFFFFF)
        args = [a_out, a_iters]
        params = (c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), c_void_p) for a in args])
        rc = self.lib.cuLaunchKernel(fn, c_uint(grid), c_uint(1), c_uint(1),
                                     c_uint(block), c_uint(1), c_uint(1),
                                     c_uint(0), stream, params, None)
        if rc != CUDA_SUCCESS:
            raise GpuBurnError("cuLaunchKernel -> %s (code %d)" % (self.error_str(rc), rc))


# --------------------------------------------------------------------------- #
# NVML（可选）：读利用率 / 温度 / 功耗，仅用于打印
# --------------------------------------------------------------------------- #

class _NvmlUtilization(Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class _NvmlMemory(Structure):
    _fields_ = [("total", c_uint64), ("free", c_uint64), ("used", c_uint64)]


class Nvml(object):
    def __init__(self, index, log):
        self.lib = None
        self.dev = c_void_p()
        self.ok = False
        for cand in NVML_CANDIDATES:
            try:
                lib = ctypes.CDLL(cand)
            except OSError:
                continue
            try:
                if lib.nvmlInit_v2() != 0:
                    log("WARN: %s 存在但 nvmlInit 失败，跳过统计" % cand)
                    continue
            except AttributeError:
                continue
            self.lib = lib
            break
        if self.lib is None:
            return
        try:
            fn = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or \
                 getattr(self.lib, "nvmlDeviceGetHandleByIndex")
            fn.argtypes = [c_uint, POINTER(c_void_p)]
            fn.restype = c_int
            if fn(index, byref(self.dev)) != 0:
                return
            self.lib.nvmlDeviceGetUtilizationRates.argtypes = [c_void_p, POINTER(_NvmlUtilization)]
            self.lib.nvmlDeviceGetUtilizationRates.restype = c_int
            self.lib.nvmlDeviceGetMemoryInfo.argtypes = [c_void_p, POINTER(_NvmlMemory)]
            self.lib.nvmlDeviceGetMemoryInfo.restype = c_int
            self.lib.nvmlDeviceGetTemperature.argtypes = [c_void_p, c_uint, POINTER(c_uint)]
            self.lib.nvmlDeviceGetTemperature.restype = c_int
            self.lib.nvmlDeviceGetPowerUsage.argtypes = [c_void_p, POINTER(c_uint)]
            self.lib.nvmlDeviceGetPowerUsage.restype = c_int
            self.lib.nvmlShutdown.argtypes = []
            self.lib.nvmlShutdown.restype = c_int
            self.ok = True
        except Exception:
            self.ok = False

    def stats(self):
        if not self.ok:
            return None
        out = {}
        try:
            u = _NvmlUtilization()
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, byref(u)) == 0:
                out["gpu_util"] = u.gpu
                out["mem_util"] = u.memory
        except Exception:
            pass
        try:
            m = _NvmlMemory()
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, byref(m)) == 0:
                out["vram_total"] = m.total
                out["vram_used"] = m.used
        except Exception:
            pass
        try:
            t = c_uint()
            if self.lib.nvmlDeviceGetTemperature(self.dev, 0, byref(t)) == 0:
                out["temp_c"] = t.value
        except Exception:
            pass
        try:
            p = c_uint()
            if self.lib.nvmlDeviceGetPowerUsage(self.dev, byref(p)) == 0:
                out["power_w"] = p.value / 1000.0
        except Exception:
            pass
        return out or None

    def shutdown(self):
        if self.ok:
            try:
                self.lib.nvmlShutdown()
            except Exception:
                pass
            self.ok = False


# --------------------------------------------------------------------------- #
# 显存占用
# --------------------------------------------------------------------------- #

def alloc_vram(cu, want, log, chunk=1 << 30):
    """按块分配，单块失败就减半重试，尽量多占（返回 (chunks, 总量)）。"""
    chunks = []
    got = 0
    while got < want:
        size = min(chunk, want - got)
        while True:
            rc, ptr = cu.try_mem_alloc(size)
            if rc == CUDA_SUCCESS:
                break
            if rc != CUDA_ERROR_OUT_OF_MEMORY or size <= (32 << 20):
                log("INFO: 显存分配到 %s 后停住（%s）" % (_human(got), cu.error_str(rc)))
                return chunks, got
            size //= 2
        chunks.append((ptr, size))
        got += size
        if len(chunks) % 8 == 0:
            log("  已占 %s" % _human(got))
    return chunks, got


def touch_vram(cu, chunks, stream, log):
    """把每块显存 memset 一遍：确认真的能用，同时让显存计数立刻反映出来。"""
    total = 0
    for i, (ptr, size) in enumerate(chunks):
        cu.memset_d32_async(ptr, size // 4, stream)
        total += size
        if (i + 1) % 8 == 0 or i + 1 == len(chunks):
            cu.stream_sync(stream)
            log("  已写入 %s / %s" % (_human(total), _human(sum(s for _, s in chunks))))
    cu.stream_sync(stream)


# --------------------------------------------------------------------------- #
# kernel 时长标定
# --------------------------------------------------------------------------- #

def calibrate(cu, fn, grid, block, stream, out_ptr, target_ms, log):
    """试跑几次，把每线程迭代数标定到目标时长（换卡不用改参数）。

    计时只用 host 时钟 + stream 同步，不碰 event API：某些共享/虚拟化环境里
    event 计时会直接失败，而 stream 同步本来就绕不开。
    """
    probe = 20000
    for _ in range(4):
        t0 = time.time()
        cu.launch(fn, grid, block, stream, out_ptr, probe)
        cu.stream_sync(stream)
        ms = (time.time() - t0) * 1000.0
        if ms >= 8.0:
            iters = int(round(probe * target_ms / ms))
            iters = max(1000, min(iters, 2000000000))
            log("标定: %d 次迭代耗时 %.1f ms -> 目标 %.0f ms 用 %d 次迭代"
                % (probe, ms, target_ms, iters))
            return iters
        if ms <= 0.0:
            probe *= 4  # 计时器精度不够
            continue
        probe = int(probe * max(2.0, target_ms / max(ms, 0.001)))
        probe = max(1000, min(probe, 2000000000))
    iters = max(1000, min(probe, 2000000000))
    log("标定: 计时不稳，退化用 %d 次迭代" % iters)
    return iters


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

class Burner(object):
    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.stop = threading.Event()
        self.stop_signal = None
        self.cu = None
        self.nvml = None
        self.ctx = None
        self.stream = None
        self.module = None
        self.fn = None
        self.chunks = []
        self.sink = None
        self.churn_thread = None
        self.launches = 0
        self.started = 0.0
        self.devs = 0
        self.dev = 0
        self.sms = 0
        self.max_threads = 1024
        self.block = 256
        self.grid = 1
        self.iters = 0
        self.streams = []
        self.inflight = []
        self.depth = 4
        self.churn_thread = None
        self._closed = False

    # -- 生命周期 ---------------------------------------------------------- #
    def _on_signal(self, signum, _frame):
        self.stop_signal = signum
        self.stop.set()

    def setup(self):
        args = self.args
        self.cu = cu = Cuda(self.log)
        self.log("驱动库: %s" % cu.libpath)
        dv = cu.driver_version()
        if dv:
            self.log("驱动 API 版本: %d.%d" % dv)

        self.devs = cu.init()
        if args.device >= self.devs:
            raise GpuBurnError("设备号 %d 越界，本机只有 %d 张卡" % (args.device, self.devs))
        dev = cu.device(args.device)
        self.dev = dev
        name = cu.device_name(dev)
        cc_major = cu.attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        cc_minor = cu.attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        self.sms = cu.attr(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) or 0
        self.max_threads = cu.attr(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) or 1024
        self.log("设备 #%d: %s  (sm_%s%s, %d SM)"
                 % (args.device, name, cc_major, cc_minor, self.sms))

        self.ctx = cu.ctx_create(dev)
        free, total = cu.mem_info()
        self.log("显存: 总 %s / 空闲 %s" % (_human(total), _human(free)))

        # --- 占显存 ---
        if args.mem_gb is not None:
            want = int(args.mem_gb * (1 << 30))
        else:
            want = int(free * args.mem_pct)
        want = min(want, max(0, free - (256 << 20)))  # 留一点给驱动自身
        if want < (64 << 20):
            raise GpuBurnError("可占显存太少（%.1f GiB 空闲），检查是否有别的进程占着" % (free / 2.0 ** 30))
        self.log("计划占用: %s（%.0f%% 空闲显存）" % (_human(want), 100.0 * want / max(free, 1)))
        self.stream = cu.stream_create()
        # 先把 kernel 输出缓冲（几 KB）拿到手，再去吃大块显存：万一显存被占满，
        # 至少 kernel 还能跑，程序不会因为“差 4KB”整体失败。
        out_ptr = self._alloc_sink(max(args.threads, 1024) * 4)
        t0 = time.time()
        self.chunks, got = alloc_vram(cu, want, self.log)
        if got == 0:
            raise GpuBurnError("显存分配失败，一块都没拿到")
        touch_vram(cu, self.chunks, self.stream, self.log)
        self.log("显存就绪: %s，耗时 %.1fs" % (_human(got), time.time() - t0))

        # --- kernel ---
        self.module, self.fn, variant = cu.load_kernel(out_ptr)
        self.log("kernel: 内联 PTX JIT 成功（%s）" % variant)

        block = args.threads
        if block > self.max_threads:
            block = self.max_threads
        if block < 32:
            block = 32
        block -= block % 32
        self.block = block
        self.grid = max(1, self.sms * args.blocks_per_sm)

        if args.iters > 0:
            self.iters = args.iters
        else:
            self.iters = calibrate(cu, self.fn, self.grid, self.block, self.stream,
                                   out_ptr, args.target_ms, self.log)
        self.log("发射配置: grid=%d  block=%d  streams=%d  iters=%d"
                 % (self.grid, self.block, args.streams, self.iters))

        # 每个 stream 用 in-flight 计数限流（保证队列里始终有活，但不会无限堆积）
        self.streams = [self.stream]
        for _ in range(args.streams - 1):
            self.streams.append(cu.stream_create())
        self.depth = max(2, args.queue_depth)
        self.inflight = [0] * len(self.streams)

        if args.churn:
            self._start_churn()

    def _alloc_sink(self, nbytes):
        rc, ptr = self.cu.try_mem_alloc(nbytes)
        if rc != CUDA_SUCCESS:
            raise GpuBurnError("无法分配 kernel 输出缓冲: %s" % self.cu.error_str(rc))
        self.sink = (ptr, nbytes)
        return ptr

    # -- 显存带宽 churn ---------------------------------------------------- #
    def _start_churn(self):
        """另起一线程，在两个显存块之间不停 D2D 拷贝，把显存带宽也刷上去。"""
        pairs = [c for c in self.chunks if c[1] >= (64 << 20)]
        if len(pairs) < 2:
            self.log("WARN: 显存块不足 2 块（每块 >=64MiB），--churn 跳过了")
            return
        a, b = pairs[0], pairs[1]
        n = min(a[1], b[1])
        n = min(n, self.args.churn_mb << 20)
        n -= n % 4
        cu = self.cu
        stream = cu.stream_create()

        def run():
            cu.ctx_set_current(self.ctx)
            idx = 0
            try:
                while not self.stop.is_set():
                    if idx % 2 == 0:
                        cu.memcpy_dtod_async(a[0], b[0], n, stream)
                    else:
                        cu.memcpy_dtod_async(b[0], a[0], n, stream)
                    idx += 1
                    if idx % 2 == 0:
                        cu.stream_sync(stream)
            except GpuBurnError as exc:
                self.log("WARN: churn 线程退出: %s" % exc)
            finally:
                try:
                    cu.stream_sync(stream)
                except GpuBurnError:
                    pass

        self.churn_thread = threading.Thread(target=run, name="churn", daemon=True)
        self.churn_thread.start()
        self.log("显存带宽 churn 已启动（每轮 %s）" % _human(n))

    # -- 主循环 ------------------------------------------------------------ #
    def run(self):
        args = self.args
        cu = self.cu
        self.started = time.time()
        next_stats = self.started + (args.stats if args.stats > 0 else 1e18)
        out_ptr = self.sink[0]
        err = None
        try:
            while not self.stop.is_set():
                for i, stream in enumerate(self.streams):
                    cu.launch(self.fn, self.grid, self.block, stream, out_ptr, self.iters)
                    self.launches += 1
                    self.inflight[i] += 1
                    # 每个 stream 排满 depth 个 kernel 就等这一批跑完：GPU 永远有活干，
                    # 同时不会把无限的 kernel 堆在驱动队列里。
                    if self.inflight[i] >= self.depth:
                        cu.stream_sync(stream)
                        self.inflight[i] = 0
                if args.duration > 0 and (time.time() - self.started) >= args.duration:
                    self.log("到达 --duration %.0fs，主动收工" % args.duration)
                    break
                now = time.time()
                if now >= next_stats:
                    self._print_stats(now - self.started)
                    next_stats = now + args.stats
        except GpuBurnError as exc:
            err = exc
        finally:
            self.shutdown()
        if err is not None:
            raise err
        return 0

    def _print_stats(self, elapsed):
        cu = self.cu
        parts = ["运行 %6.1fs" % elapsed, "kernel 发射 %d 次" % self.launches]
        try:
            free, total = cu.mem_info()
            parts.append("显存 %s/%s" % (_human(total - free), _human(total)))
        except GpuBurnError:
            pass
        if self.nvml:
            s = self.nvml.stats()
            if s:
                if "gpu_util" in s:
                    parts.append("GPU 利用率 %d%%" % s["gpu_util"])
                if "mem_util" in s:
                    parts.append("显存带宽 %d%%" % s["mem_util"])
                if "temp_c" in s:
                    parts.append("%dC" % s["temp_c"])
                if "power_w" in s:
                    parts.append("%.0fW" % s["power_w"])
        self.log(" | ".join(parts))

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        cu = self.cu
        if cu is None:
            return
        self.stop.set()
        if self.churn_thread is not None:
            self.churn_thread.join(timeout=5.0)
            self.churn_thread = None
        # 先把队列里的 kernel 排干，再释放资源
        try:
            if self.ctx is not None:
                cu.ctx_sync()
        except GpuBurnError as exc:
            self.log("WARN: 退出前 ctx sync 失败（可能上下文已丢失）: %s" % exc)
        for ptr, _size in self.chunks:
            cu.mem_free(ptr)
        self.chunks = []
        if self.sink is not None:
            cu.mem_free(self.sink[0])
            self.sink = None
        # self.streams[0] 就是分配/标定用的 self.stream，统一在这里销毁一次
        for s in self.streams:
            cu.stream_destroy(s)
        self.streams = []
        self.stream = None
        if self.module is not None:
            cu.module_unload(self.module)
            self.module = None
        if self.ctx is not None:
            cu.ctx_destroy(self.ctx)
            self.ctx = None
        if self.nvml:
            self.nvml.shutdown()
            self.nvml = None
        self.log("已释放显存与 CUDA 上下文")


def build_parser():
    p = argparse.ArgumentParser(
        prog="gpu_burn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="占满 CUDA 显存并把 GPU 利用率刷到接近 100%（不依赖 PyTorch / CUDA Toolkit）",
        epilog="""示例:
  python3 gpu_burn.py                          # 占 60% 空闲显存，一直刷，5 秒报一次
  python3 gpu_burn.py --mem-gb 70              # 固定占 70GiB
  python3 gpu_burn.py --duration 1800          # 刷 30 分钟
  python3 gpu_burn.py --churn --stats 2        # 顺带刷显存带宽
  python3 gpu_burn.py --check                  # 只自检，不常驻
""")
    p.add_argument("--device", type=int, default=0, help="GPU 序号（默认 0）")
    p.add_argument("--mem-gb", type=float, default=None, metavar="GB",
                   help="固定占用多少 GiB 显存（默认按 --mem-pct 算）")
    p.add_argument("--mem-pct", type=float, default=0.60, metavar="P",
                   help="占空闲显存的比例，0~1（默认 0.60）")
    p.add_argument("--duration", type=float, default=0, metavar="SEC",
                   help="运行多少秒后自动退出，0=一直跑（默认 0）")
    p.add_argument("--stats", type=float, default=5.0, metavar="SEC",
                   help="多少秒打一次状态，0=不打（默认 5）")
    p.add_argument("--target-ms", type=float, default=120.0, metavar="MS",
                   help="单个 kernel 的目标时长，越大发射开销占比越小（默认 120）")
    p.add_argument("--iters", type=int, default=0, metavar="N",
                   help="每线程迭代数，0=自动标定（默认 0）")
    p.add_argument("--blocks-per-sm", type=int, default=8, metavar="N",
                   help="每个 SM 排多少个 block（默认 8，256 线程时正好吃满 2048 线程/SM）")
    p.add_argument("--threads", type=int, default=256, metavar="N", help="每个 block 的线程数（默认 256）")
    p.add_argument("--streams", type=int, default=1, metavar="N", help="并发 stream 数（默认 1）")
    p.add_argument("--queue-depth", type=int, default=4, metavar="N",
                   help="每个 stream 最多预排的 kernel 数（默认 4）")
    p.add_argument("--churn", action="store_true", help="另起线程做 D2D 拷贝，把显存带宽也刷起来")
    p.add_argument("--churn-mb", type=int, default=512, metavar="MB", help="churn 单次拷贝大小(默认 512)")
    p.add_argument("--check", action="store_true", help="只做环境自检和短时试跑，然后退出")
    p.add_argument("--diag", action="store_true", help="逐步检查驱动/CUDA 链路（排障用）")
    p.add_argument("--dump-ptx", action="store_true",
                   help="打印实际交给驱动的 PTX 文本后退出（调试/校验用）")
    p.add_argument("--quiet", action="store_true", help="只打错误")
    p.add_argument("--version", action="version", version="gpu_burn " + VERSION)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dump_ptx:
        sys.stdout.write(KERNEL_VARIANTS[0][1])
        return 0
    lock = threading.Lock()

    def log(msg):
        if args.quiet and not msg.startswith(("ERR", "WARN")):
            return
        with lock:
            sys.stdout.write("[gpu_burn %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
            sys.stdout.flush()

    burner = Burner(args, log)
    old_handlers = {}

    def install_signals():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, burner._on_signal)
            except (ValueError, OSError):
                pass

    def restore_signals():
        for sig, old in old_handlers.items():
            try:
                signal.signal(sig, old)
            except (ValueError, OSError):
                pass

    install_signals()
    code = 0
    try:
        log("gpu_burn %s，pid %d" % (VERSION, os.getpid()))
        if args.diag:
            _diag(args, log)
            return 0
        if args.check:
            _self_check(args, log)
            return 0
        burner.setup()
        burner.nvml = Nvml(args.device, log)
        if args.check:
            pass
        burner.run()
    except GpuBurnError as exc:
        log("ERR: %s" % exc)
        code = 3 if burner.ctx is not None or burner.cu is not None else 2
    except KeyboardInterrupt:
        log("收到 Ctrl-C，退出")
        code = 0
    finally:
        try:
            burner.shutdown()
        except Exception as exc:  # 清理阶段不再抛
            log("WARN: 清理时异常: %s" % exc)
        restore_signals()
    return code


def _diag(args, log):
    """--diag：把整条链路拆成一步一步，哪一步炸就停在哪一步（真机排障用）。"""
    cu = Cuda(log)
    log("驱动库: %s" % cu.libpath)
    dv = cu.driver_version()
    if dv:
        log("驱动 API 版本: %d.%d" % dv)
    state = {"step": 0}

    def step(desc, fn, *a):
        state["step"] += 1
        try:
            r = fn(*a)
        except GpuBurnError as exc:
            log("ERR: diag %d) %s 失败: %s" % (state["step"], desc, exc))
            log("提示: 设备侧错误时，最后一条 OK 之后的那次调用出问题了；"
                "可以再试 --threads 128 --blocks-per-sm 1 --iters 2000")
            raise SystemExit(3)
        log("diag %d) %s: OK" % (state["step"], desc))
        return r

    def alloc(nbytes):
        rc, ptr = cu.try_mem_alloc(nbytes)
        if rc != CUDA_SUCCESS:
            raise GpuBurnError("cuMemAlloc: %s" % cu.error_str(rc))
        return ptr

    ctx = stream = module = None
    ptr_sink = ptr_chunk = 0
    try:
        n = step("cuInit", cu.init)
        log("     可见设备数: %d" % n)
        if args.device >= n:
            log("ERR: diag: --device %d 越界，本机只有 %d 张卡" % (args.device, n))
            raise SystemExit(3)
        dev = step("cuDeviceGet", cu.device, args.device)
        sms = cu.attr(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) or 1
        max_threads = cu.attr(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) or 1024
        log("diag 4) 设备属性: %s sm_%s%s %d SM maxThreads/block=%d"
            % (cu.device_name(dev),
               cu.attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
               cu.attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR), sms, max_threads))
        ctx = step("cuCtxCreate", cu.ctx_create, dev)
        stream = step("cuStreamCreate", cu.stream_create)
        ptr_sink = step("cuMemAlloc 4KiB 输出缓冲", alloc, 4096)
        ptr_chunk = step("cuMemAlloc 64MiB", alloc, 64 << 20)
        step("cuMemsetD32Async 64MiB", cu.memset_d32_async, ptr_chunk, (64 << 20) // 4, stream)
        step("cuStreamSynchronize（等 memset 落地）", cu.stream_sync, stream)
        module, fn, variant = step("PTX JIT（cuModuleLoadData + cuModuleGetFunction）",
                                   cu.load_kernel, ptr_sink)
        log("     JIT 用的是 %s" % variant)
        step("小规模 launch（grid=SM block=32 iters=1000）",
             cu.launch, fn, sms, 32, stream, ptr_sink, 1000)
        step("cuStreamSynchronize（小规模 kernel）", cu.stream_sync, stream)
        block = min(args.threads, max_threads)
        block -= block % 32
        grid = max(1, sms * max(1, args.blocks_per_sm))
        t0 = time.time()
        step("满规模 launch（grid=SM*blocks-per-sm iters=20000）",
             cu.launch, fn, grid, block, stream, ptr_sink, 20000)
        step("cuStreamSynchronize（满规模 kernel）", cu.stream_sync, stream)
        log("     满规模 kernel 耗时 %.1f ms（grid=%d block=%d）"
            % ((time.time() - t0) * 1000.0, grid, block))
        log("diag 完成：从驱动到 kernel 这条链路可以跑")
    finally:
        if module is not None:
            cu.module_unload(module)
        if ptr_chunk:
            cu.mem_free(ptr_chunk)
        if ptr_sink:
            cu.mem_free(ptr_sink)
        if stream is not None:
            cu.stream_destroy(stream)
        if ctx is not None:
            try:
                cu.ctx_sync()
            except GpuBurnError:
                pass
            cu.ctx_destroy(ctx)


def _self_check(args, log):
    """--check：不常驻，只验证「驱动可见 / 能分配显存 / PTX 能 JIT / kernel 能跑」。"""
    cu = Cuda(log)
    log("驱动库: %s" % cu.libpath)
    dv = cu.driver_version()
    if dv:
        log("驱动 API 版本: %d.%d" % dv)
    n = cu.init()
    log("可见设备数: %d" % n)
    if args.device >= n:
        raise GpuBurnError("设备号 %d 越界，本机只有 %d 张卡" % (args.device, n))
    dev = cu.device(args.device)
    sms = cu.attr(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) or 1
    log("设备 #%d: %s  (sm_%s%s, %d SM)"
        % (args.device, cu.device_name(dev),
           cu.attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
           cu.attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR), sms))
    ctx = cu.ctx_create(dev)
    stream = module = None
    ptr = 0
    try:
        free, total = cu.mem_info()
        log("显存: 总 %s / 空闲 %s" % (_human(total), _human(free)))
        stream = cu.stream_create()
        rc, ptr = cu.try_mem_alloc(256 << 20)
        if rc != CUDA_SUCCESS:
            raise GpuBurnError("分配 256MiB 测试显存失败: %s" % cu.error_str(rc))
        cu.memset_d32_async(ptr, (256 << 20) // 4, stream)
        cu.stream_sync(stream)
        log("分配 + 写入 256MiB 测试显存: OK")
        out_ptr = ptr + (128 << 20)
        module, fn, variant = cu.load_kernel(out_ptr)
        log("PTX JIT: OK（%s）" % variant)
        grid = max(1, sms * 8)
        iters = calibrate(cu, fn, grid, 256, stream, out_ptr, 50.0, log)
        t0 = time.time()
        cu.launch(fn, grid, 256, stream, out_ptr, iters)
        cu.stream_sync(stream)
        log("kernel 试跑: OK，grid=%d block=256 iters=%d，耗时 %.1f ms"
            % (grid, iters, (time.time() - t0) * 1000.0))
        nv = Nvml(args.device, log)
        if nv.ok:
            s = nv.stats() or {}
            log("NVML: OK  %s" % " ".join("%s=%s" % kv for kv in sorted(s.items())))
            nv.shutdown()
        else:
            log("NVML: 不可用（不影响刷，只是没有利用率数字可看）")
    finally:
        if module is not None:
            cu.module_unload(module)
        if stream is not None:
            cu.stream_destroy(stream)
        if ptr:
            cu.mem_free(ptr)
        try:
            cu.ctx_sync()
        except GpuBurnError:
            pass
        cu.ctx_destroy(ctx)
    log("自检通过：这个容器可以在当前这张卡上刷起来")


if __name__ == "__main__":
    sys.exit(main())
