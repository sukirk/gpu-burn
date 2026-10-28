#!/usr/bin/env bash
# 校验原生版与 python 版生成的 PTX 逐字节相同，并在有 ptxas 时汇编一遍确认语法没问题。
#
# mock 驱动只检查 PTX 里有没有该有的指令，真正决定 kernel 能不能跑的是驱动的 JIT；
# 有 ptxas 就等于有一个离线 JIT 前端，可以提前发现语法错。
#
#   bash tests/verify_ptx.sh
#   PTXAS=/path/to/ptxas bash tests/verify_ptx.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
PYBIN="$ROOT/python/gpu_burn.py"
BIN="${BIN:-./gpu_burn}"
TMP="${TMPDIR:-/tmp}/gb_ptx.$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0
ok()  { pass=$((pass + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

echo "== PTX 一致性 =="
"$PY" "$PYBIN" --dump-ptx >"$TMP/py.ptx" || { bad "python 版导出 PTX"; exit 1; }
if [ -x "$BIN" ]; then
    "$BIN" --dump-ptx >"$TMP/c.ptx" || bad "原生版导出 PTX"
    if diff -q "$TMP/py.ptx" "$TMP/c.ptx" >/dev/null; then
        ok "原生版与 python 版生成的 PTX 逐字节相同"
    else
        bad "原生版与 python 版的 PTX 不一致"
        diff "$TMP/py.ptx" "$TMP/c.ptx" | head -20
    fi
else
    echo "  SKIP 没有原生版二进制（先 make）"
fi

# 找 ptxas：$PTXAS > PATH
PTXAS="${PTXAS:-}"
if [ -z "$PTXAS" ]; then
    if command -v ptxas >/dev/null 2>&1; then
        PTXAS="$(command -v ptxas)"
    fi
fi

echo "== 真 ptxas 汇编 =="
if [ -z "$PTXAS" ]; then
    echo "  SKIP 没有 ptxas（可选：PTXAS=/path/to/ptxas bash tests/verify_ptx.sh）"
else
    echo "  用: $PTXAS"
    # 三个变体：sm_80(ISA 7.0)、sm_80(ISA 8.0)、sm_70(ISA 6.4)
    "$PY" - "$TMP" <<'PYEOF'
import importlib.util, os, sys
out = sys.argv[1]
spec = importlib.util.spec_from_file_location("gb", "python/gpu_burn.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for i, (label, text) in enumerate(m.KERNEL_VARIANTS):
    arch = "sm_70" if "sm_70" in text.split(".target")[1][:20] else "sm_80"
    with open(os.path.join(out, "v%d_%s.ptx" % (i, arch)), "w") as f:
        f.write(text)
PYEOF
    for f in "$TMP"/v*.ptx; do
        arch="$(basename "$f" .ptx)"; arch="${arch#*_}"
        out="$TMP/$(basename "$f" .ptx).cubin"
        if log=$("$PTXAS" -arch="$arch" -v "$f" -o "$out" 2>&1); then
            regs=$(printf '%s\n' "$log" | grep -o 'Used [0-9]* registers' | head -1)
            ok "$(basename "$f") 汇编通过（$arch，${regs:-寄存器信息未打印}）"
        else
            bad "$(basename "$f") 汇编失败（$arch）"
            printf '%s\n' "$log" | tail -10
        fi
    done
fi

echo
echo "通过 $pass 项，失败 $fail 项"
[ "$fail" -eq 0 ]
