# gpu_burn_ascend — 昇腾 NPU 版「刷利用率」

把 Ascend 910B2 的 HBM 占满、AICore 利用率刷到接近 100%，用来做拷机、稳定性验证、
池化平台占位测试。行为和仓库里的 CUDA 版 `src/gpu_burn.c` 对齐：同样的日志前缀、
参数风格、setup / run_loop / cleanup 分层和退出码 0 / 2 / 3。

不依赖 CANN 头文件，编译期只需要 gcc；CANN 的 `libascendcl.so` 在运行时用 `dlopen`
加载，所有 ACL 类型和函数原型都在源码里自己声明。所以在没装 CANN 的开发机上也能
`make` 出二进制。

**单进程多线程**：一个进程里 `aclInit()` 一次，每颗芯片一个线程（各自
`aclrtSetDevice` + 自己的 stream + 自己的 HBM 缓冲）。K8s 池化平台按进程分配
NPU 会话，绝对不能 fork 子进程去刷第二颗卡，否则父进程退出时整批会话都会被收回。

```
ascend/
├── gpu_burn_ascend.c        # 单文件实现（只依赖 libc / libdl / libpthread）
├── Makefile                 # make / make mocks / make check / make clean
├── README.md
└── tests/
    ├── mock_libascendcl.c   # 假驱动：导出同名符号，严格校验参数
    └── run_tests.sh         # 端到端测试，BIN= 可指向被测二进制
```

## 编译

```sh
cd ascend
make                 # -> ./gpu_burn_ascend，零警告（-Wall -Wextra）
make check           # 编假驱动 + 跑 tests/run_tests.sh
```

只需要 `cc`、`libdl`、`libpthread`。真机上不需要把 CANN 的 include 加进来，也不需要
链接 `libascendcl.so`。

## 快速开始

```sh
./gpu_burn_ascend --check                 # 先确认这张卡能刷（自检，退出码 0/3）
./gpu_burn_ascend --diag                  # 刷不起来时逐步看哪一步 ACL 调用失败
./gpu_burn_ascend                         # 0 号卡，占 60% 空闲 HBM，一直刷到 Ctrl-C
./gpu_burn_ascend --devices all --npu-smi # 8 张卡一起刷（仍然是一个进程）
./gpu_burn_ascend --devices 0,2 --duration 1800 --stats 5 --npu-smi
./gpu_burn_ascend --mem-gb 40 --size 8192 --queue 2
```

容器里跑（设备节点和驱动目录都要挂）：

```sh
docker run --rm -it \
  --device=/dev/davinci0 --device=/dev/davinci_manager \
  --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64 \
  <镜像> /path/to/gpu_burn_ascend --check
```

多卡时按需把 `/dev/davinci1` … `/dev/davinci7` 一起挂进来。

## 参数

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--device N` | — | 单芯片快捷方式，等价于 `--devices N` |
| `--devices LIST` | `0` | 要刷的芯片：`0,1,2` 或 `all`。去重；越界、解析失败都退出 3 |
| `--mem-gb GB` | — | 每芯片固定占用多少 GiB HBM（优先级高于 `--mem-pct`） |
| `--mem-pct P` | `0.60` | 占每芯片**空闲** HBM 的比例，0.6 = 60%（与 CUDA 版一致；写 60 也认） |
| `--duration SEC` | `0` | 跑多少秒后正常收工；0 = 一直跑，等 SIGINT/SIGTERM/SIGHUP |
| `--stats SEC` | `5` | 每多少秒打一行统计；0 = 不打 |
| `--size N` | `8192` | MatMul 的 M=N=K。FP16 时每个矩阵 `2*N*N` 字节，8192 → 128 MiB/个 |
| `--queue N` | `2` | 每芯片异步队列深度：提交 N 次 `aclopExecuteV2` 后同步一次 |
| `--check` | | 单芯片自检：aclInit → 设备数 → SoC → 256 MiB 分配 + memset → 一次 2048 MatMul → 同步 → 释放 |
| `--diag` | | 逐步骤（14 步）打印每步返回码，失败处停下并给出 `aclGetRecentErrMsg` |
| `--npu-smi` | | 统计行里附上 `npu-smi info` 解析出的 AICore% / HBM 用量 / 温度 / 功耗 |
| `--quiet` | | 只打错误 |
| `--version` | | 打印 `gpu_burn_ascend 1.3.0`（和仓库同版本号） |
| `-h`, `--help` | | 帮助 |

HBM 分配方式：按 1 GiB 分块 `aclrtMalloc(ACL_MEM_MALLOC_HUGE_FIRST)`，每块立刻
`aclrtMemset` 落盘（不 memset 的话物理页可能没真正绑定，`npu-smi` 里的 HBM-Usage
不会涨）；某一块 OOM 就把块大小减半重试，一直到 32 MiB，仍失败就保留已拿到的量、
打一条 WARN 继续跑。

## 输出长什么样

自检：

```
[gpu_burn_ascend 17:30:02] 可见芯片数: 8
[gpu_burn_ascend 17:30:02] 设备 #0: Ascend910B2
[gpu_burn_ascend 17:30:02] 自检通过：这张卡可以刷起来（aclInit → 设备 → HBM 读写 → MatMul 全链路正常）
```

单芯片运行（`--stats 1 --npu-smi`）：

```
[gpu_burn_ascend 17:30:10] 运行    1.0s | MatMul 发射 266 次 | 累计 292.6 TFLOP | 实测 292.4 TFLOPS | HBM 已占 38.40 GiB/64.00 GiB | npu-smi: AICore 96% | HBM 3401/65536 MB | 88.3W | 34C
```

多芯片收尾（每行带 `#N` 卡号，日志整体加锁，不会串行）：

```
[gpu_burn_ascend 17:30:20 #0] 到达 --duration 60s，主动收工
[gpu_burn_ascend 17:30:20 #0] 已释放 HBM 与设备上下文
[gpu_burn_ascend 17:30:21 #2] 到达 --duration 60s，主动收工
[gpu_burn_ascend 17:30:21 #2] 已释放 HBM 与设备上下文
[gpu_burn_ascend 17:30:21] 全部 2 个芯片已收工，HBM 都已释放
```

`--diag` 的每一步都是 `diag N) <做什么>: OK`，最后一行是
`diag 完成：从 aclInit 到 MatMul 这条链路可以跑`。

## 910B2 上的预期表现

- 每芯片 HBM 65536 MB。`--mem-pct 0.6`（=60%）→ 每芯片约 38.4 GiB 占位，加上 MatMul 的
  3 个 128 MiB 缓冲，`npu-smi` 的 HBM-Usage 应该稳定在 39 GiB 上下。
- `--size 8192` 时单次 MatMul 是 `2*8192³ ≈ 1.1 TFLOP`；统计行里的「累计 TFLOP」是
  `2*M*N*K*发射次数`，除以运行时长就是后面的实测 TFLOPS。910B2 的 FP16 稠密峰值
  以官方规格 / `npu-smi info -t board -i 0` 为准，走 `aclopExecuteV2` 单算子方式
  实测一般能到峰值的 70%~90%，统计行里的 TFLOPS 是**实测算力**（只算 MatMul 的
  `2*M*N*K`，含搬运和提交开销），比峰值低是正常的。
- `npu-smi` 的 AICore% 应该稳定在 90% 以上，偶尔掉到 80 多属于 MatMul 之间的提交
  间隙；如果长期低于 70%，见下面 FAQ「AICore 上不去」。
- 温度和功耗会明显上升（接近板卡 TDP），拷机前建议先用 `--duration 60 --npu-smi`
  看一眼散热余量。
- 8 卡全开时 `--mem-pct 0.6` 一共占约 307 GiB HBM、8 个线程；注意整机功耗和池化
  平台的会话配额。

## 退出码

| 码 | 含义 |
| --- | --- |
| 0 | 正常结束（含 `--check` 通过、被信号正常收工） |
| 2 | 环境不满足：找不到 `libascendcl.so`、看不到芯片、容器缺 `/dev/davinci*` 等；输出里会给出该挂哪些设备节点和库目录 |
| 3 | 参数错误（`--devices` 越界/解析失败）或 ACL 调用失败：打印 ACL 数字错误码 + `aclGetRecentErrMsg` 文案 + 是哪一步 |

多芯片时某颗卡失败不会拖死其它卡：该卡标 failed，其它卡继续跑，最终退出码取最坏。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `ASCENDCL_LIB` | 直接指定 `libascendcl.so` 的绝对路径（优先级最高，排障时最方便） |
| `ASCEND_TOOLKIT_HOME` / `ASCEND_HOME_PATH` / `ASCEND_OPP_PATH` | 帮助拼出候选库路径（`<root>/lib64/libascendcl.so`、`<root>/latest/lib64/...` 等） |
| `LD_LIBRARY_PATH` | 常规动态库搜索路径，没设 `ASCENDCL_LIB` 时会先 `dlopen("libascendcl.so")` |
| `NPU_SMI_BIN` | 覆盖 `--npu-smi` 用的命令，默认 `npu-smi info`；测试用它喂假数据 |

测试用假驱动专用的变量（只在 `make check` 里用，真机不需要）：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `MOCK_CHIPS` | 8 | `aclrtGetDeviceCount` 返回几颗 |
| `MOCK_TOTAL_MB` | 65536 | 每芯片 HBM 总量 |
| `MOCK_USED_MB` | 0 | 预置「别的进程已占用」的 HBM，用来测 OOM 降级 |
| `MOCK_SOC` | Ascend910B2 | `aclrtGetSocName` 返回值 |
| `MOCK_FLOPS` | 300e12 | 每次 MatMul 按 `2*M*N*K/FLOPS` 睡，模拟算力 |
| `MOCK_LOG` | — | 假驱动日志文件（追加写），测试拿它核对 alloc/free 配平 |
| `MOCK_FAIL_AT` | 0 | 第 N 次 `aclopExecuteV2` 注入错误；之后保持「粘性失败」 |
| `MOCK_FAIL_CODE` | 507000 | 注入的错误码 |

## 常见问题（容器 / 池化平台）

**1. `ERR: 找不到 libascendcl.so`，退出码 2**

宿主机有 CANN、容器里没挂进来。挂
`-v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit`
和 `-v /usr/local/Ascend/driver:/usr/local/Ascend/driver`，并把
`/usr/local/Ascend/ascend-toolkit/latest/lib64`、`/usr/local/Ascend/driver/lib64`
加进 `LD_LIBRARY_PATH`。临时排障可以直接
`ASCENDCL_LIB=/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so ./gpu_burn_ascend --check`。

**2. 库能找到，但提示「没有可见芯片」，退出码 2**

`/dev` 下没有 `davinci*`。容器要 `--device=/dev/davinci0`（多卡逐个加）、
`--device=/dev/davinci_manager`、`--device=/dev/devmm_svm`、`--device=/dev/hisi_hdc`。
K8s 池化平台则要确认 pod 真的申请到了 NPU 资源（如 `huawei.com/Ascend910`），
不是只挂了个镜像。

**3. `aclInit` 就失败（`--diag` 第 1 步非 0）**

多半是驱动版本 / CANN 版本不匹配，或者容器少挂了 `/dev/hisi_hdc`、
`/dev/devmm_svm`。先把宿主机上 `npu-smi info` 跑通，再对照 `--diag` 的返回码。

**4. 运行中出现 `507000` 之类的设备侧错误码**

这类码是设备侧 / 上下文错误：池化平台会话被别人抢走、pod 被驱逐、驱动复位、
或者别的进程 reset 了同一颗卡。程序会打印 `aclGetRecentErrMsg` 的文案、指明是哪一步
（通常第 12 步 `aclopExecuteV2`），该卡标记失败、其它卡继续。重开会话再跑；
反复出现就看宿主 `dmesg` 和 `npu-smi info -t health -i N`。

**5. HBM 占不满**

别人（或上一次没退干净的进程）占着。`npu-smi info` 看 HBM-Usage，用
`--mem-gb` 手动压到能拿到的量，或者等对方释放。程序本身在 OOM 时会自动把块从
1 GiB 减半到 32 MiB，并把已拿到的量保留下来。

**6. AICore% 上不去（长期 < 70%）**

- `--size` 太小：MatMul 太快，时间都花在提交和轮询上。建议 8192。
- `--queue` 太浅：默认 2，可以试 4（显存够的话），但队列越深，单次同步的等待越长。
- `--stats` 太密：统计本身很轻，但 `--npu-smi` 每次要 `popen` 一条 `npu-smi info`，
  它自身有几百毫秒开销。拷机时 `--stats 5` 或 10，别用 0.5。
- 池化平台单卡被切给多个会话时，AICore% 是共享的，单进程刷不到 90% 属正常。
- `aclopExecuteV2` 是单算子接口，每次提交都有宿主侧开销。判断是不是提交跟不上：
  跑的时候在宿主上看这个进程的 CPU 占用（`top -H -p <pid>`），如果某个线程常年
  100%、AICore 却上不去，就加大 `--size`（单次算得更久）或者 `--queue`。

**7. 池化平台上的注意事项**

- 一个进程 = 一套 ACL 上下文；`--devices all` 在池化环境里通常只能看到分配给你的那几张，
  `aclrtGetDeviceCount` 返回的就是你的配额，别写死 8。
- 不要拿 `--devices` 去点不属于本会话的芯片，会直接报无效设备。
- 程序只用 pthread，不 fork；唯一的外部进程是 `--npu-smi` 里 `popen("npu-smi info")`，
  它是一次性的短命子进程，不碰 ACL 上下文，也不影响会话归属。

**8. 容器里没有 `npu-smi`**

`--npu-smi` 会 WARN 一句然后继续跑，不影响刷卡；统计行里只是少了那一段。

**9. `--check` 通过，`--duration` 一跑就报 `aclrtMalloc` 失败**

多半是 `--size 8192` 需要的 3×128 MiB 加 HBM 占位超过了空闲量。调小
`--mem-pct` / `--mem-gb`，或调小 `--size`（例如 `--size 4096`，每个矩阵 32 MiB）。

## 实现要点

- **自己声明 ACL**：`typedef int aclError;`，`aclrtStream` / `aclTensorDesc` /
  `aclDataBuffer` / `aclopAttr` 都是 `void *`；常量 `ACL_FLOAT16=1`、
  `ACL_FORMAT_ND=2`、`ACL_MEM_MALLOC_HUGE_FIRST=0`、`ACL_HBM_MEM=1`。
  函数原型按 CANN 文档声明，运行时 `dlsym` 逐个绑定，缺哪个在报错里点名。
  `aclrtGetMemInfo`、`aclrtGetSocName`、`aclopCreateAttr/aclopSetAttrBool/aclopDestroyAttr`
  是可选符号：老版本库没有就降级，不影响刷卡。
- **MatMul 用 `aclopExecuteV2`**：FP16、2 输入 1 输出、`[M,K] × [K,N] → [M,N]`，
  属性 `transpose_a=false` / `transpose_b=false` 通过 `aclopAttr` 显式传
  （部分 CANN 版本不传 attr 会报「attr 未设置」；库里没有 `aclopCreateAttr` 时退回
  `attr=NULL`）。
- **每芯片一个线程**：`aclrtSetDevice` → `aclrtCreateStream` → `aclrtGetMemInfo` →
  分配 HBM → 分配 A/B/C → `run_loop`；`run_loop` 里提交 `queue` 次
  `aclopExecuteV2` 后 `aclrtSynchronizeStream` 一次，统计发射次数和实际耗时。
- **收尾**：`aclrtSynchronizeStream` 排干 → 释放所有 HBM 块和 MatMul 缓冲 →
  `aclrtDestroyStream` → `aclrtResetDevice`；`cleanup` 幂等，信号打断也能正常释放。
  最后统一打「全部 N 个芯片已收工，HBM 都已释放」。
- **日志**：`[gpu_burn_ascend HH:MM:SS]`（多芯片带 `#N`），整行先在内存里拼好再
  加锁 `fwrite`，多线程不会互相插字。

## 测试

```sh
make check                       # 全部用例，最后打印「用例断言: 56 通过, 0 失败」
BIN=/path/to/gpu_burn_ascend make check   # 指定被测二进制
```

`tests/run_tests.sh` 用 `LD_LIBRARY_PATH` 把假驱动 `tests/mock/libascendcl.so`
顶到前面，覆盖 10 组场景、56 条断言：

1. 没有 `libascendcl.so`：退出码 2，提示里有库名、`/dev/davinci` 和 CANN 路径；
2. `--check`：退出 0，输出自检通过 / SoC 名 / 设备数，假驱动里记录到 MatMul；
3. `--duration 3 --stats 1`：退出 0，有 TFLOPS 统计行和「已收工 / HBM 都已释放」，
   按卡核对 alloc/free 配平；
4. `--devices 0,2` 与 `--devices all`：每颗卡都有带 `#N` 的日志，并且扫
   `/proc/*/cmdline`（只比 `argv[0]`，不用 pgrep）断言**全程只有一个进程**；
5. `SIGTERM`：发 TERM 后正常退出，每芯片 malloc/free 计数配平、`reset-device`
   各一次；
6. `--diag`：退出 0，每一步 OK，最后有「diag 完成」；
7. `MOCK_FAIL_AT=1 MOCK_FAIL_CODE=507000`：退出码 3，输出含错误码 507000 且指出
   失败在 `aclopExecuteV2`；假驱动日志里能看到粘性失败；
8. `--npu-smi`：用一个 shell 脚本假扮 `npu-smi`（`NPU_SMI_BIN`），喂 24.1.0 的
   双行样例，断言统计行里 AICore%、HBM 用量、温度、功耗都被正确解析；
9. `--devices 9`：越界，退出码 3；
10. `--version` / `--help` / `--mem-gb 2` 生效。

假驱动不是「只返回成功」：它按芯片记账并返回互不重叠的假地址区间，
`aclrtMalloc` 超总量返回失败，`aclrtMemset` 校验指针归属和长度，
`aclopExecuteV2` 逐项校验 opType / 输入输出个数 / dtype / format / 形状 /
每个 `aclDataBuffer` 的大小和指针是否落在本芯片已分配的块里 / 属性是否设置，
再用 `nanosleep` 按 `2*M*N*K/MOCK_FLOPS` 模拟耗时。测试产物（`tests/.out`、
`tests/.mock.log`）跑完自动清理。

## 已知限制 / 只能在真机验证的部分

- 本仓库的开发机上没有 NPU、没有 CANN，所以 ACL 原型是照文档写的。**第一次上真机
  请先跑 `--diag`**：它会逐步骤打印返回码，能一次性确认 `aclInit`、
  `aclrtGetDeviceCount`、`aclrtSetDevice`、`aclrtMalloc/Memset`、`aclCreateTensorDesc`、
  `aclopExecuteV2` 的参数顺序和签名是否与当前 CANN 版本匹配。
- `aclopExecuteV2` 走的是单算子模式（不是 aclnn 图模式），不同 CANN 版本对 MatMul
  的 attr 要求略有差异；我们传了 `transpose_a/transpose_b=false`，若某版本仍报
  属性问题，`--diag` 会直接暴露。
- 真实 HBM 占用量、AICore% 能否到 90%+、TFLOPS 数字、温度/功耗曲线都只能在
  910B2 上实测，README 里给的只是量级参考。
- 多芯片并发时的池化平台会话配额、`npu-smi` 24.1.0 的实际输出格式（包含 Process
  表）需要真机复核；解析器按「板卡行 + 芯片行」的双行格式做的，对样例之外的行会
  跳过。
