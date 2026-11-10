#!/usr/bin/env bash
# 把编好的昇腾版打成一个发布包：二进制 + README + LICENSE + BUILD-INFO。
#
# 用法：package-ascend.sh <version> <arch> <label> [binary]
#   package-ascend.sh v1.2.0 x86_64 ubuntu-22.04
#   package-ascend.sh v1.2.0 aarch64 ubuntu-22.04 ascend/gpu_burn_ascend
#   OBJDUMP=aarch64-linux-gnu-objdump package-ascend.sh v1.2.0 aarch64 ubuntu-22.04
#
# 产物：dist/gpu-burn-ascend-<version>-linux-<arch>-<label>.tar.gz
set -euo pipefail

version=${1:?用法: package-ascend.sh <version> <arch> <label> [binary]}
version=${version#v}          # tag 名可能是 v1.2.0，包名里不要 v
arch=${2:?用法: package-ascend.sh <version> <arch> <label> [binary]}
label=${3:?用法: package-ascend.sh <version> <arch> <label> [binary]}
binary=${4:-ascend/gpu_burn_ascend}

[ -x "$binary" ] || { echo "找不到可执行文件：$binary" >&2; exit 1; }

objdump_bin=${OBJDUMP:-objdump}
glibc_req=$("$objdump_bin" -T "$binary" 2>/dev/null \
    | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1 || true)
if [ -n "$glibc_req" ]; then
    glibc_note=$(echo "$glibc_req" | sed 's/GLIBC_/GLIBC_/')
else
    glibc_note="unknown"
fi

name=gpu-burn-ascend-${version}-linux-${arch}-${label}
mkdir -p dist "pkg/$name"

cp "$binary" "pkg/$name/gpu_burn_ascend"
cp ascend/README.md "pkg/$name/README.md"
cp LICENSE "pkg/$name/LICENSE"

{
    echo "name: $name"
    echo "os: $label"
    echo "arch: $arch"
    echo "binary: $("$binary" --version 2>/dev/null | head -1 || echo 'unknown')"
    echo "cc: $(cc --version 2>/dev/null | head -1 || echo 'unknown')"
    echo "glibc_required: $glibc_note"
    echo "built_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "pkg/$name/BUILD-INFO"

chmod 755 "pkg/$name/gpu_burn_ascend"
chmod 644 "pkg/$name/README.md" "pkg/$name/LICENSE" "pkg/$name/BUILD-INFO"

tar -czf "dist/$name.tar.gz" -C pkg "$name"
ls -l "dist/$name.tar.gz"
