#!/usr/bin/env bash
# 把编好的 gpu_burn 打成发布用的压缩包：
#   dist/gpu-burn-<version>-linux-x86_64-<label>.tar.gz
# 包里：gpu_burn、README.md、LICENSE、BUILD-INFO（编译环境与 glibc 要求）
#
# 用法：package.sh <version> <label>
#   version  版本号，带不带前缀 v 都行；tag 构建时传 tag 名
#   label    编译环境标签，写进文件名，如 ubuntu-20.04
set -euo pipefail

version=${1:?用法: package.sh <version> <label>}
label=${2:?用法: package.sh <version> <label>}
version=${version#v}

bin=${BIN:-./gpu_burn}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

if [ ! -x "$bin" ]; then
    echo "找不到可执行文件：$bin（先 make）" >&2
    exit 1
fi

name="gpu-burn-${version}-linux-x86_64-${label}"
rm -rf dist pkg
mkdir -p dist "pkg/$name"
cp "$bin" "pkg/$name/gpu_burn"
cp README.md LICENSE "pkg/$name/"

os_name=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release 2>/dev/null | tr -d '"' || true)
cc_version=$("${CC:-cc}" --version 2>/dev/null | head -1 || true)
glibc_req=$(objdump -T "$bin" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1 || true)

{
    echo "os: ${os_name:-unknown}"
    echo "cc: ${cc_version:-unknown}"
    echo "glibc_required: ${glibc_req:-unknown}"
    echo "built_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "pkg/$name/BUILD-INFO"

# 统一权限：不管源文件在本地是什么 mode，包里都是可执行 + 可读
chmod 755 "pkg/$name/gpu_burn"
chmod 644 "pkg/$name/README.md" "pkg/$name/LICENSE" "pkg/$name/BUILD-INFO"

tar -czf "dist/$name.tar.gz" -C pkg "$name"

echo "==> dist/$name.tar.gz"
cat "pkg/$name/BUILD-INFO"
ls -l "dist/$name.tar.gz"
