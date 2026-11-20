"""极简假 torch：给 python/tests/test_ascend_py.sh 用，本机没有 NPU 也能跑通控制流。

只实现 gpu_burn_ascend.py 用到的那几个 API，matmul 用 sleep 模拟耗时。
"""
import time


class dtype:
    def __init__(self, name, itemsize):
        self.name, self.itemsize = name, itemsize

    def __repr__(self):
        return "torch." + self.name


float16, bfloat16, float32, uint8 = (dtype("float16", 2), dtype("bfloat16", 2),
                                     dtype("float32", 4), dtype("uint8", 1))


class Tensor:
    def __init__(self, shape, dt, dev, fill=0.0):
        self.shape, self.dtype, self.device, self.fill = shape, dt, dev, fill
        self.nbytes = dt.itemsize
        for d in shape:
            self.nbytes *= d
        # 超过 64MiB 的“显存”不真分配，避免测试把本机内存吃掉
        self.mem = bytearray(self.nbytes) if self.nbytes <= (64 << 20) else None

    def __getitem__(self, i):
        return self

    def __float__(self):
        return self.fill


def zeros(n, dtype=None, device=None):
    return Tensor((n,), dtype, device)


def empty(*shape, dtype=None, device=None):
    return Tensor(shape, dtype, device)


def ones(*shape, dtype=None, device=None):
    return Tensor(shape, dtype, device, fill=1.0)


def randn(*shape, dtype=None, device=None):
    return Tensor(shape, dtype, device)


def mm(a, b, out=None):
    time.sleep(min(0.002, 8.0 * a.shape[0] ** 3 / 1e12))
    if out is None:
        out = Tensor((a.shape[0], b.shape[1]), a.dtype, a.device)
    out.fill = float(a.shape[0])  # 相当于 ones @ ones 的结果
    return out


class npu:
    count, free, total = 2, 1536 << 20, 2048 << 20

    @staticmethod
    def device_count():
        return npu.count

    @staticmethod
    def set_device(dev):
        pass

    @staticmethod
    def get_device_name(dev):
        return "Mock 910B2 #%d" % dev

    @staticmethod
    def mem_get_info(dev):
        return npu.free, npu.total

    @staticmethod
    def synchronize():
        time.sleep(0.001)

    @staticmethod
    def empty_cache():
        pass


class _Props:
    total_memory = 2048 << 20


class cuda:  # 给 python/gpu_burn_torch.py 用的假 CUDA 后端
    count, free, total = 2, 1536 << 20, 2048 << 20

    @staticmethod
    def is_available():
        return cuda.count > 0

    @staticmethod
    def device_count():
        return cuda.count

    @staticmethod
    def set_device(dev):
        pass

    @staticmethod
    def get_device_name(dev):
        return "Mock A100-SXM4-80GB #%d" % dev

    @staticmethod
    def mem_get_info(dev):
        return cuda.free, cuda.total

    @staticmethod
    def get_device_properties(dev):
        return _Props

    @staticmethod
    def synchronize():
        time.sleep(0.001)

    @staticmethod
    def empty_cache():
        pass
