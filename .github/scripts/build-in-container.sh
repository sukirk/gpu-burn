#!/usr/bin/env bash
# 在指定发行版的容器里编译、跑测试、打包。
#
# 用法：build-in-container.sh <docker image> <label>
#   build-in-container.sh ubuntu:20.04 ubuntu-20.04
#
# 打包出来的二进制放在 dist/ 下，属主是容器里的 root，只读即可，不需要 chown。
set -euo pipefail

image=${1:?用法: build-in-container.sh <docker image> <label>}
label=${2:?用法: build-in-container.sh <docker image> <label>}

docker run --rm \
    -v "$PWD":/work -w /work \
    -e VERSION="${VERSION:-dev}" \
    -e LABEL="$label" \
    "$image" bash -c '
        set -e
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
            gcc libc6-dev make python3 binutils >/dev/null
        make
        make check
        echo "==> glibc 最高要求：$(objdump -T gpu_burn | grep -o "GLIBC_[0-9.]*" | sort -Vu | tail -1)"
        bash .github/scripts/package.sh "$VERSION" "$LABEL"
    '
