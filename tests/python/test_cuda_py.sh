#!/usr/bin/env bash
# gpu_burn_torch.py 的端到端测试：用假 torch（本机无 GPU 也能跑）。
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYBIN:-python3}
S="$ROOT/python/gpu_burn_torch.py"
FAKE="$ROOT/tests/python/fake_torch"
OUT="$ROOT/tests/python/.out"
mkdir -p "$OUT"
pass=0 fail=0

ok()  { pass=$((pass + 1)); printf '  ok   %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$1"; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

echo "== gpu_burn_torch.py =="

out=$(PYTHONPATH="$FAKE" $PY "$S" --check --devices 0,1 2>&1); rc=$?
check "自检 2 张卡退出 0" "[ $rc -eq 0 ]"
check "自检打印通过" "grep -q '自检通过' <<<\"\$out\""

out=$(PYTHONPATH="$FAKE" $PY "$S" --duration 2 --stats 1 --devices all --size 512 --mem-pct 0.05 2>&1); rc=$?
check "刷 2 秒退出 0" "[ $rc -eq 0 ]"
check "统计行有实测 TFLOPS" "grep -q '实测 .* TFLOPS' <<<\"\$out\""
check "两张卡各自释放" "[ \$(grep -c '已释放显存' <<<\"\$out\") -eq 2 ]"
check "收尾总结" "grep -q '全部 2 张卡已收工' <<<\"\$out\""

out=$(PYTHONPATH="$FAKE" $PY "$S" --devices 5 --duration 1 2>&1); rc=$?
check "越界卡号退出 3" "[ $rc -eq 3 ]"
check "越界提示" "grep -q '超出范围' <<<\"\$out\""

out=$(PYTHONPATH="$FAKE" $PY "$S" --mem-pct 0 --duration 1 2>&1); rc=$?
check "mem-pct 0 退出 2" "[ $rc -eq 2 ]"

PYTHONPATH="$FAKE" $PY "$S" --duration 60 --stats 1 --size 512 --mem-pct 0.05 > "$OUT/sig.log" 2>&1 &
pid=$!
sleep 2.5
kill -TERM "$pid" 2>/dev/null
wait "$pid"; rc=$?
check "SIGTERM 后退出 0" "[ $rc -eq 0 ]"
check "SIGTERM 后打印释放" "grep -q '已释放显存' '$OUT/sig.log'"
check "SIGTERM 后进程已退出" "! kill -0 $pid 2>/dev/null"

out=$($PY "$S" --check 2>&1); rc=$?
check "没有 torch / 看不到 CUDA 时退出 2 或 3" "[ $rc -eq 2 ] || [ $rc -eq 3 ]"
check "并说清缺什么" "grep -qE '找不到 torch|看不到 CUDA' <<<\"\$out\""

rm -rf "$OUT"
echo
if [ "$fail" -eq 0 ]; then echo "通过 $pass 项，失败 0 项"; else echo "通过 $pass 项，失败 $fail 项"; fi
[ "$fail" -eq 0 ]
