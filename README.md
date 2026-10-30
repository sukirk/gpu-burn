# gpu-burn

[![CI](https://github.com/sukirk/gpu-burn/actions/workflows/ci.yml/badge.svg)](https://github.com/sukirk/gpu-burn/actions/workflows/ci.yml)

把 NVIDIA 显卡的显存占满、GPU 利用率刷到接近 100%，用来做压测、烤机、占卡。
一个 43KB 的单文件二进制，只依赖 glibc。

```
$ ./gpu_burn --check
[gpu_burn 09:39:50] 驱动库: libcuda.so.1
[gpu_burn 09:39:50] 设备 #0: NVIDIA A100-SXM4-80GB  (sm_80, 108 SM)
[gpu_burn 09:39:50] 显存: 总 80.00 GiB / 空闲 79.25 GiB
[gpu_burn 09:39:50] 分配 + 写入 256MiB 测试显存: OK
[gpu_burn 09:39:50] kernel: 内联 PTX JIT 成功（sm_80 / PTX ISA 7.0）
[gpu_burn 09:39:50] 自检通过：当前环境可以正常刷
```

## 特性

- **零依赖**：不需要 CUDA Toolkit、nvcc、PyTorch、Python 或任何 pip 包，只要宿主机 NVIDIA 驱动注入的 `libcuda.so.1`
- **单个二进制**：`gpu_burn` 约 43KB，拷进容器就能跑，glibc ≥ 2.17
- **自适应**：启动时自动标定 kernel 时长，换卡、换驱动不用改参数
- **PTX JIT**：计算 kernel 以内联 PTX 交给驱动编译，一份程序覆盖 sm_70 及以上
- **显存占满**：按空闲比例（`--mem-pct`）或绝对值（`--mem-gb`）分配，OOM 自动降级
- **利用率拉满**：每线程 4 条独立 FP32 FMA 并行发射，填满每个 SM 的 4 个 warp 调度器
- **可选带宽压测**：`--churn` 额外做 device-to-device 拷贝
- **多卡**：`--devices all` 或 `--devices 0,2`，一个进程里每张卡一个线程，一起刷一起收尾
- **干净退出**：`Ctrl-C` / `SIGTERM` / `SIGHUP` 都会排干队列、释放显存再退出
- **状态可见**：周期性打印利用率、显存、温度、功耗（容器内有 NVML 时）
- **附带 Python 版**：`python/gpu_burn.py`，只用标准库，作为没有编译环境时的备选

## 环境要求

| 项 | 要求 |
| --- | --- |
| GPU | NVIDIA sm_70 及以上（A100 / A30 / V100 / H100 等） |
| 驱动 | 支持 CUDA 11+ 的 NVIDIA 驱动 |
| 系统 | Linux x86_64，glibc ≥ 2.17 |
| 容器 | 以 `--gpus all` 启动（需要宿主机装 nvidia-container-toolkit） |
| 编译 | gcc + make（只用来编这个程序，不涉及 CUDA） |
| Python 版 | Python ≥ 3.6，无需 pip 包 |

## 下载预编译二进制

不想自己编译就从 [Releases](https://github.com/sukirk/gpu-burn/releases) 下：每个发行版一个包，内容相同，只是编译环境不同。

| 文件 | 编译环境 |
| --- | --- |
| `gpu-burn-VERSION-linux-x86_64-ubuntu-20.04.tar.gz` | ubuntu 20.04 / gcc 9 |
| `gpu-burn-VERSION-linux-x86_64-debian-13.tar.gz` | debian 13 / gcc 14 |
| `gpu-burn-VERSION-linux-x86_64-ubuntu-22.04.tar.gz` | ubuntu 22.04 / gcc 11 |
| `gpu-burn-VERSION-linux-x86_64-debian-12.tar.gz` | debian 12 / gcc 12 |
| `gpu-burn-VERSION-linux-x86_64-ubuntu-24.04.tar.gz` | ubuntu 24.04 / gcc 13 |

包里是 `gpu_burn` + `README.md` + `LICENSE` + `BUILD-INFO`（编译环境、gcc 版本、这个二进制要求的最高 glibc 符号版本）。
拿不准就下 `ubuntu-20.04` 那个，要求最低，在更新的发行版上照样能跑。

```bash
tar xzf gpu-burn-*-linux-x86_64-ubuntu-20.04.tar.gz
cd gpu-burn-*/
./gpu_burn --check
```

Release 里带 `SHA256SUMS`，可用 `sha256sum -c SHA256SUMS` 校验。

## 快速开始

不用下载也行：`make` 编出 `./gpu_burn`（见[编译](#编译)），或者直接用 `python/gpu_burn.py`（只依赖标准库）。

```bash
# 二进制放进容器（挂载最省事）
docker run --gpus all -it -v "$PWD":/opt/gpu-burn <你的镜像> bash

/opt/gpu-burn/gpu_burn --check      # 先自检：看得到卡、能分配显存、kernel 能 JIT
/opt/gpu-burn/gpu_burn              # 开刷：占 60% 空闲显存，一直跑，每 5 秒报一次
```

跑固定时长、跑完自动退出：

```bash
docker run --gpus all --rm -v "$PWD":/opt/gpu-burn <你的镜像> \
    /opt/gpu-burn/gpu_burn --duration 1800
```

用 Python 版（不需要编译）：

```bash
docker run --gpus all -it -v "$PWD":/opt/gpu-burn <你的镜像> \
    python3 /opt/gpu-burn/python/gpu_burn.py --duration 1800
```

## 用法

### 常用示例

```bash
./gpu_burn                          # 占 60% 空闲显存，一直刷，每 5 秒报一次
./gpu_burn --mem-gb 78              # 固定占 78GiB（80GB 卡上约 97%）
./gpu_burn --mem-pct 0.9            # 占空闲显存的 90%
./gpu_burn --duration 3600          # 刷 1 小时
./gpu_burn --churn --stats 2        # 顺带刷显存带宽，2 秒报一次
./gpu_burn --streams 4 --queue-depth 8   # 4 个 stream 并发，队列排深一点
./gpu_burn --check                  # 只自检，不常驻
./gpu_burn --diag                   # 逐步检查驱动 → 显存 → JIT → kernel，排障用
./gpu_burn --device 1               # 用第二张卡
./gpu_burn --devices all            # 机器上所有卡一起刷，每张一个子进程
./gpu_burn --devices 0,2            # 只刷 0 号和 2 号卡
```

### 参数

```
用法: gpu_burn [选项]

选项:
  --device N          GPU 序号（默认 0），等价于 --devices N
  --devices LIST      要刷的卡号，逗号分隔或 all（如 0,1,2 / all）
  --mem-gb GB         固定占用多少 GiB 显存（默认按 --mem-pct 算）
  --mem-pct P         占空闲显存的比例，0~1（默认 0.60）
  --duration SEC      运行多少秒后自动退出，0=一直跑（默认 0）
  --stats SEC         多少秒打一次状态，0=不打（默认 5）
  --target-ms MS      单个 kernel 的目标时长（默认 120）
  --iters N           每线程迭代数，0=自动标定（默认 0）
  --blocks-per-sm N   每个 SM 排多少个 block（默认 8）
  --threads N         每个 block 的线程数（默认 256）
  --streams N         并发 stream 数（默认 1）
  --queue-depth N     每个 stream 最多预排的 kernel 数（默认 4）
  --churn             另起线程做 D2D 拷贝，把显存带宽也刷起来
  --churn-mb MB       churn 单次拷贝大小（默认 512）
  --check             只做环境自检和短时试跑，然后退出
  --diag              逐步检查驱动/CUDA 链路，打印每一步的结果（排障用）
  --quiet             只打错误
  --version           打印版本
  -h, --help          打印本页
```

### 多卡

`--devices` 接受 `all` 或逗号分隔的卡号，默认跟 `--device` 一样只刷 0 号卡。

```bash
./gpu_burn --devices all                    # 所有卡
./gpu_burn --devices 0,1 --mem-pct 0.8      # 指定两张卡，各占各自空闲显存的 80%
./gpu_burn --devices all --check            # 每张卡都自检一遍
```

一个进程，每张卡一个线程：各自建 CUDA 上下文、各自分配显存、各自的统计行，日志前缀
带上卡号（`[gpu_burn 12:00:00 #1]`）。`--mem-pct` / `--mem-gb` 是**按每张卡**分别算的，
不是总量。

不 fork 子进程是有意的：有些池化/虚拟化平台按**进程**分配 GPU 会话，子进程继承不了父进程
的会话，重新申请会直接失败。单进程只建立一个会话，多张卡各自建上下文，这是 CUDA 本来就
支持的用法。

收尾因此天然干净，不会留下占着显存的进程：

- 收到 `Ctrl-C` / `SIGTERM` / `SIGHUP`：主线程把停止标志置上，每个卡的线程排干队列、
  释放显存、销毁上下文后退出，最后打印"全部 N 张卡已收工"；
- 进程被 `kill -9`：本来就只有一个进程，进程一没，它持有的显存全部由驱动回收；
- 某张卡失败了（上下文建不起来、kernel 报错）：只标记这张卡，其他卡照常刷，退出码取最坏的
  那个（`3` > `2` > `0`）。

如果只想让程序看见部分卡（比如别的任务在用 0 号卡），外面套 `CUDA_VISIBLE_DEVICES=1,2`
更省事，此时程序内部看到的编号会重新从 0 开始。

### 输出

运行期间的统计行长这样（数字随卡、随参数变化）：

```
[gpu_burn 09:41:12] 发射配置: grid=864  block=256  streams=1  iters=811710
[gpu_burn 09:41:17] 运行   10.4s | kernel 发射 87 次 | 显存 48.00 GiB/80.00 GiB | GPU 利用率 99% | 显存带宽 41% | 71C | 398W
```

一次 `--check` 会依次验证：驱动库能否加载 → 设备可见 → 能分配并写入 256MiB 显存 →
PTX 能被 JIT → kernel 能执行。哪一步失败会直接给出要查的东西。

`--diag` 把同一条链路拆成 15 步（`cuInit`、取设备、建上下文、建 stream、分配、memset、
JIT、小规模 launch、满规模 launch……），每一步单独打印 OK/失败，适合在真卡上定位
"到底哪一步不行"。所有 CUDA 报错都带错误码（例如 `unknown (rc=999)`），
`rc=999` / `rc=719` / `rc=700` 基本都是设备侧错误（kernel 非法访问、上下文丢失）。

### 退出码与信号

| 退出码 | 含义 |
| --- | --- |
| `0` | 正常结束 |
| `2` | 环境不满足（加载不到驱动、看不到设备、显存不够等） |
| `3` | CUDA 调用失败 |

`Ctrl-C`、`SIGTERM`、`SIGHUP` 都会走同一套收尾：排队等 GPU 干完 → 释放所有显存 →
销毁 stream / module / context。多卡模式下每张卡的线程各自收尾；被 `kill -9` 时进程直接
消失，显存由驱动回收。容器被停掉时不会留下占着显存的进程。

## 编译

```bash
make            # 生成 ./gpu_burn
make check      # 编译 + 跑端到端测试（不需要 GPU）
make clean
```

编译只需要 gcc 和 make，不链接任何 CUDA 库。产物链接的是 `libc`、`libdl`、`libpthread`。
在哪个发行版上编，就要那个发行版及以上才能跑：要兼容 ubuntu 20.04（glibc 2.31），
就在对应容器里编：

```bash
docker run --rm -v "$PWD":/work -w /work ubuntu:20.04 \
    bash -c "apt-get update -qq && apt-get install -y -qq gcc libc6-dev make && make"
```

CI 会分别在 ubuntu 20.04 / 22.04 / 24.04、debian 12 / 13 上编译并跑测试，打好的包推到 Release。

## 测试

`make check` 用 `tests/` 下的假驱动（`mock_libcuda.c`、`mock_nvml.c`）替换真驱动，
跑 32 项端到端用例：自检、`--diag`、常驻刷、`--duration`、OOM 降级、多 stream、`--churn`、
`SIGTERM` 优雅退出、显存被别的进程占着、设备侧错误带码退出、分配/释放配平、上下文销毁，
以及多卡相关的 `--devices 0,1` / `--devices all` / 越界卡号 / 只起一个进程 / 单卡失败隔离 /
`SIGTERM` 后全部释放 / `kill -9` 后无残留进程。原生版和 Python 版各跑一遍。

`tests/verify_ptx.sh` 校验两版生成的 PTX 是否逐字节一致；机器上有 `ptxas` 时
（`PTXAS=/path/to/ptxas`）还会汇编一遍确认指令合法。

## 目录结构

```
.
├── Makefile
├── src/gpu_burn.c          # 原生实现（单文件）
├── python/gpu_burn.py      # Python 实现（单文件，仅标准库）
├── tests/
│   ├── mock_libcuda.c      # 假 libcuda.so.1，模拟 Driver API
│   ├── mock_nvml.c         # 假 libnvidia-ml.so.1
│   ├── run_tests.sh        # 端到端用例（make check）
│   └── verify_ptx.sh       # PTX 一致性 / ptxas 校验
└── .github/
    ├── workflows/ci.yml            # 测试 + 五个发行版构建 + 发 Release
    ├── scripts/package.sh          # 打包成 dist/gpu-burn-*.tar.gz
    ├── scripts/build-in-container.sh
    └── RELEASE_NOTES.md            # Release 正文
```

## 典型表现

A100 80GB 单卡，`--mem-pct 0.60` 时：

| 指标 | 表现 |
| --- | --- |
| 显存 | ≈ 48 GiB（`--mem-gb 78` 可到 ≈ 78 GiB） |
| `nvidia-smi` 利用率 | ≈ 100% |
| 显存带宽（`--churn`） | 20~45% |
| 功耗 | 接近 TDP（A100 SXM ≈ 400W） |
| 温度 | 60~80℃，取决于散热 |

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `libcuda.so.1 没能加载` | 容器没以 GPU 方式启动。用 `--gpus all`；宿主机需装 nvidia-container-toolkit。旧版本 docker 可用 `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` |
| `看不到任何 GPU 设备` | 同上。也可以在容器里看 `ls /dev/nvidia*`、`ldconfig -p \| grep libcuda`，在宿主机跑 `nvidia-smi` |
| 可占显存太少 | 卡上已有别的进程，用 `--mem-gb` 给小一点 |
| 统计行没有温度/功耗 | 容器里没有 `libnvidia-ml.so.1`，不影响刷 |
| `所有 PTX 版本都不能被驱动接受` | 驱动太老（早于 CUDA 11），升级宿主机驱动 |
| `标定失败`、`rc=999`/`rc=719`/`rc=700`、`上下文已丢失` | 设备侧错误。先跑 `--diag` 看卡在哪一步；能过 `--diag` 但刷的时候挂，试 `--streams 1 --queue-depth 2`，或先把卡上别的进程清掉。1.0.0 里标定用的是 CUDA event 计时，在少数共享/虚拟化环境会直接报错，**1.0.1 起改成 host 时钟 + stream 同步，不再依赖 event** |
| 想知道实际刷到什么程度 | 另开终端 `nvidia-smi -l 1`，或本程序 `--stats 1` |
| 想换成更小的显存占用 | `--mem-pct 0.3` 或 `--mem-gb 16` |

## 注意

- 默认会占掉 60% 的空闲显存、把功耗推到接近 TDP。**共享卡上先问清楚**，用 `--mem-pct`/`--mem-gb` 调小。
- 长时间满载烤机注意散热和电费。
- 程序只管刷，不校验计算结果，也不做算力跑分。

## License

[MIT](LICENSE)
