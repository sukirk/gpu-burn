# gpu-burn

占满 NVIDIA 显卡显存、把 GPU 利用率刷到接近 100%。单个二进制，不需要 CUDA Toolkit、PyTorch 或 Python。

## 下哪个

按编译环境命名，内容一样，挑一个能跑在你的系统上就行：

| 文件 | 编译环境 |
| --- | --- |
| `gpu-burn-*-linux-x86_64-ubuntu-20.04.tar.gz` | ubuntu 20.04 / gcc 9 |
| `gpu-burn-*-linux-x86_64-debian-11.tar.gz` | debian 11 / gcc 10 |
| `gpu-burn-*-linux-x86_64-ubuntu-22.04.tar.gz` | ubuntu 22.04 / gcc 11 |
| `gpu-burn-*-linux-x86_64-debian-12.tar.gz` | debian 12 / gcc 12 |
| `gpu-burn-*-linux-x86_64-ubuntu-24.04.tar.gz` | ubuntu 24.04 / gcc 13 |

每个包里附了 `BUILD-INFO`，写着编译环境、gcc 版本和这个二进制要求的最高 glibc 符号版本。
拿不准就下 `ubuntu-20.04` 那个，要求最低，在更新的发行版上照样能跑。

## 用

```bash
sha256sum -c SHA256SUMS                       # 可选：校验（在放压缩包的目录里跑）
tar xzf gpu-burn-*-linux-x86_64-ubuntu-20.04.tar.gz
cd gpu-burn-*/

./gpu_burn --check          # 先自检：看得到卡、能分配显存、kernel 能 JIT
./gpu_burn                  # 开刷：占 60% 空闲显存，一直跑，每 5 秒报一次
./gpu_burn --duration 1800  # 跑 30 分钟自动退出
```

在容器里跑要带 `--gpus all`，并且宿主机装了 nvidia-container-toolkit。
完整参数、退出码、常见问题见仓库 README。
