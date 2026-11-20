# gpu-burn v1.3.0

占满 NVIDIA 显卡显存、把 GPU 利用率刷到接近 100%。单个二进制，不需要 CUDA Toolkit、PyTorch
或 Python；另外还有纯 Python 版和昇腾 NPU 版。

这次是 v1.1.1 之后两个版本的合并发布：1.2.0 重写了计算 kernel、加了昇腾 NPU 版；
1.3.0 补了 PyTorch 版的 CUDA / 昇腾实现，并把仓库结构整理了一遍。

## 新增

- **CUDA PyTorch 版**（`python/gpu_burn_torch.py`）：不写 kernel，用 cuBLAS GEMM 反复压计算单元。
  面板上"浮点利用率"只显示一半时，用它对照就能分清是面板口径还是卡真被限了：
  fp16/bf16 走 Tensor Core，fp32 走 FP32 管线，统计行直接报实测 TFLOPS。
- **昇腾 PyTorch 版**（`python/gpu_burn_ascend.py`）：torch_npu + `torch.mm`，参数和 C 版一致，
  一个进程每颗芯片一个线程，`Ctrl-C` 时每颗芯片都会释放 HBM。
- **昇腾原生版**（`ascend/`，1.2.0 起）：dlopen `libascendcl.so`，不需要 CANN 头文件；
  兼容 `npu-smi` 24.1.0 的输出；一个进程每颗芯片一个线程（池化平台按进程分配 NPU 会话，所以不 fork）。
- **`--peak-tflops`**（CUDA 版）：卡型不在内置表里时自己填 FP32 峰值，统计行按它算占比。
- **统计行报实测算力**：`实测算力 X TFLOPS（FP32 峰值 Y，占 Z%）`，CUDA 和昇腾版都有。
- 仓库整理：`tests/cuda/` 和 `tests/python/` 分开；CUDA 三种实现的参数对齐；
  CI 增加了 Python 版行为测试（假 torch，不需要 GPU / NPU）。

## 修复

- **CUDA kernel 重写（1.2.0）**：旧 kernel 每轮 4 条 FMA + 3 条循环开销（FMA 只占 57%），
  4 个累加器的依赖链又刚好压着 FFMA 的 4 周期延迟，FP32 管线理论上限就在 50%~60%。
  新 kernel 8 个累加器 × 8 次展开 = 64 条 FMA + 3 条开销（95.5% 是 FMA），
  ptxas 汇编 20 个寄存器、0 spill，2048 线程/SM 的满 occupancy 不受影响。
- **"nvidia-smi 100% 但浮点利用率一半"说不清原因**：GPU-Util 只表示"有 kernel 在跑"，
  和浮点管线无关。现在统计行给出实测 TFLOPS，可以直接判断是面板口径还是达峰问题。
- **多卡在池化 / vGPU 平台起不来（1.1.1）**：fork 出来的子进程要重新申请 GPU 会话，
  而池化平台按进程分配会话，于是每个子进程都报 `Fail to init session`。改成单进程多线程
  （一张卡一个线程、共用一个 CUDA 会话），`kill` 一次全部收尾，也不会有孤儿进程。
- **`--devices` 越界或写错格式**：现在直接报错退出 3，不会闷头跑到一半。
- **打包脚本缺 `make clean`**：容器里 make 认为宿主编的二进制是最新的，测试拿宿主编的
  （要求 glibc 2.38）在 ubuntu 20.04 里跑，报 `GLIBC_2.34 not found`、25/36 失败。
- **测试脚本依赖 pgrep/ps**：debian 最小镜像没有这些工具，`kill -9` 用例因此误判"没有释放"，
  CI 红过一轮。改成直接扫 `/proc/*/cmdline`，不依赖外部命令。
- **假驱动跨机器拷贝后不重编**：`.so` 比源码新，make 就不重编，测试 dlopen 莫名其妙失败。
  现在 `make mocks / check` 每次强制重编假驱动。

## 下哪个

按编译环境命名，内容一样，挑一个能跑在你的系统上就行：

| 文件 | 编译环境 |
| --- | --- |
| `gpu-burn-*-linux-x86_64-ubuntu-20.04.tar.gz` | ubuntu 20.04 / gcc 9 |
| `gpu-burn-*-linux-x86_64-debian-13.tar.gz` | debian 13 / gcc 14 |
| `gpu-burn-*-linux-x86_64-ubuntu-22.04.tar.gz` | ubuntu 22.04 / gcc 11 |
| `gpu-burn-*-linux-x86_64-debian-12.tar.gz` | debian 12 / gcc 12 |
| `gpu-burn-*-linux-x86_64-ubuntu-24.04.tar.gz` | ubuntu 24.04 / gcc 13 |

每个包里附了 `BUILD-INFO`，写着编译环境、gcc 版本和这个二进制要求的最高 glibc 符号版本。
拿不准就下 `ubuntu-20.04` 那个，要求最低，在更新的发行版上照样能跑。

昇腾 NPU 版是单独的包，x86_64 和 aarch64 各一个：

| 文件 | 架构 / 编译环境 |
| --- | --- |
| `gpu-burn-ascend-*-linux-x86_64-ubuntu-22.04.tar.gz` | x86_64 / ubuntu 22.04 |
| `gpu-burn-ascend-*-linux-aarch64-ubuntu-22.04.tar.gz` | aarch64 / ubuntu 22.04 |

昇腾包在装好 CANN 的机器上跑（运行时 dlopen `libascendcl.so`），第一次先
`./gpu_burn_ascend --diag` 看 14 步返回码。

两个 Python 版（`python/gpu_burn_torch.py`、`python/gpu_burn_ascend.py`）不出独立包，
直接用仓库里的文件就行：它们只依赖 `torch` / `torch_npu`，和架构无关。

## 用

```bash
sha256sum -c SHA256SUMS                       # 可选：校验（在放压缩包的目录里跑）
tar xzf gpu-burn-*-linux-x86_64-ubuntu-20.04.tar.gz
cd gpu-burn-*/

./gpu_burn --check          # 先自检：看得到卡、能分配显存、kernel 能 JIT
./gpu_burn                  # 开刷：占 60% 空闲显存，一直跑，每 5 秒报一次
./gpu_burn --duration 1800  # 跑 30 分钟自动退出
    ./gpu_burn --devices all    # 所有卡一起刷（一个进程，每张卡一个线程）
```

在容器里跑要带 `--gpus all`，并且宿主机装了 nvidia-container-toolkit。
完整参数、退出码、面板利用率对不上的排查，见仓库 README。
