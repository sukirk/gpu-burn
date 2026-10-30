#!/usr/bin/env bash
# 端到端测试：用 tests/mock/ 下的假 libcuda/libnvidia-ml 代替真驱动，
# 跑完整流程并检查退出码、关键日志、mock 侧看到的调用序列。不需要 GPU。
#
# 用法（在目标镜像或任意有 python3 + gcc 的环境里）：
#   bash tests/run_tests.sh                 # 只测 python 版
#   BIN=./gpu_burn bash tests/run_tests.sh  # 连原生版一起测（make check 就是这么跑的）
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MOCK="$ROOT/tests/mock"
PYBIN="$ROOT/python/gpu_burn.py"
PY="${PY:-python3}"
BIN="${BIN:-}"
LOG="$ROOT/tests/.mock.log"
TMP="$ROOT/tests/.out"
mkdir -p "$TMP"

if [ ! -f "$MOCK/libcuda.so.1" ]; then
    make -C "$ROOT" mocks >/dev/null || { echo "编不出 tests/mock 下的假驱动，需要 gcc"; exit 1; }
fi

pass=0
fail=0
declare -a FAILED

ok()   { pass=$((pass + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { fail=$((fail + 1)); FAILED+=("$1"); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

# runner <名字> <退出码> <输出文件> <必须出现的正则> [必须不出现的正则]
expect() {
    local name="$1" code="$2" file="$3" must="$4" never="${5:-}"
    if [ "$code" -ne 0 ]; then
        bad "$name（退出码 $code）"
        sed 's/^/      /' "$file" | tail -15
        return
    fi
    if ! grep -qE "$must" "$file"; then
        bad "$name（输出里没有 /$must/）"
        sed 's/^/      /' "$file" | tail -15
        return
    fi
    if [ -n "$never" ] && grep -qE "$never" "$file"; then
        bad "$name（出现了不该出现的 /$never/）"
        return
    fi
    ok "$name"
}

echo "== 用例：$PY 版 =="

# 1) 没有任何驱动库时的报错路径
out="$TMP/nodriver.txt"
env -u LD_LIBRARY_PATH -u CUDA_VISIBLE_DEVICES "$PY" "$PYBIN" --check >"$out" 2>&1
code=$?
[ "$code" -eq 2 ] && grep -qE "libcuda.so.1 没能加载|nvidia-container-toolkit" "$out" \
    && ok "无驱动时给出可操作的报错并退出 2" || bad "无驱动时的报错路径（退出码 $code）"

# 2) 自检
out="$TMP/check.txt"
LD_LIBRARY_PATH="$MOCK" MOCK_LOG="$LOG" "$PY" "$PYBIN" --check >"$out" 2>&1
expect "--check 全流程自检" $? "$out" "自检通过"

# 2b) --diag 逐步检查链路
out="$TMP/diag.txt"
LD_LIBRARY_PATH="$MOCK" "$PY" "$PYBIN" --diag >"$out" 2>&1
expect "--diag 逐步检查链路" $? "$out" "diag 完成"

# 3) 正常刷 + 每 1 秒统计
out="$TMP/run.txt"
LD_LIBRARY_PATH="$MOCK" MOCK_LOG="$LOG" "$PY" "$PYBIN" --duration 2 --stats 1 --target-ms 40 >"$out" 2>&1
expect "常驻刷 2 秒后自行退出" $? "$out" "已释放显存与 CUDA 上下文"
expect "统计行带利用率" $? "$out" "GPU 利用率 99%"

# 4) 显存不足时的降级（OOM 后减半重试，再拿不到就带已有量继续跑）
out="$TMP/oom.txt"
LD_LIBRARY_PATH="$MOCK" MOCK_OOM_AFTER_GB=6 "$PY" "$PYBIN" --mem-gb 20 --duration 1 --stats 0 >"$out" 2>&1
expect "显存分配 OOM 后降级继续" $? "$out" "停住"

# 5) 多 stream + churn
out="$TMP/churn.txt"
LD_LIBRARY_PATH="$MOCK" MOCK_LOG="$LOG" "$PY" "$PYBIN" --streams 2 --churn --duration 2 --stats 1 >"$out" 2>&1
expect "多 stream + 显存带宽 churn" $? "$out" "churn 已启动"

# 6) SIGTERM 优雅退出
out="$TMP/sig.txt"
LD_LIBRARY_PATH="$MOCK" "$PY" "$PYBIN" --stats 0 --target-ms 40 >"$out" 2>&1 &
pid=$!
sleep 2
kill -TERM "$pid" 2>/dev/null
wait "$pid"
code=$?
[ "$code" -eq 0 ] && grep -qE "已释放显存" "$out" \
    && ok "SIGTERM 优雅退出并释放显存" || { bad "SIGTERM 退出（退出码 $code）"; tail -5 "$out"; }

# 7) 设备侧错误（上下文丢失）：报错必须带错误码，退出 3
out="$TMP/sticky.txt"
LD_LIBRARY_PATH="$MOCK" MOCK_FAIL_LAUNCH_AT=1 MOCK_FAIL_CODE=999 "$PY" "$PYBIN" --stats 0 >"$out" 2>&1
code=$?
[ "$code" -eq 3 ] && grep -qE "rc=999" "$out" \
    && ok "设备侧错误时带错误码退出 3" || { bad "设备侧错误路径（退出码 $code）"; tail -8 "$out"; }

# 8) 显存被别的进程占着（总 39GiB / 空闲 2.6GiB）也要能正常刷
out="$TMP/tight.txt"
LD_LIBRARY_PATH="$MOCK" MOCK_TOTAL_GB=39 MOCK_USED_GB=36.4 \
    "$PY" "$PYBIN" --duration 1 --stats 0 >"$out" 2>&1
expect "显存被别的程序占着时也能正常刷" $? "$out" "已释放显存与 CUDA 上下文"

# 9) mock 侧断言：那次完整运行里所有参数合法性检查都过了，且分配/释放配平
if grep -qE "BAD SETUP|OUT OF RANGE|INVALID_PTX" "$LOG"; then
    bad "mock 侧参数校验（发现非法调用）"
else
    ok "mock 侧参数校验（设备指针/迭代数/区间检查全过）"
fi
allocs=$(grep -c "cuMemAlloc(" "$LOG")
frees=$(grep -c "cuMemFree(" "$LOG")
if [ "$allocs" -gt 0 ] && [ "$allocs" -eq "$frees" ]; then
    ok "显存分配/释放配平（$allocs 次分配，$frees 次释放）"
else
    bad "显存分配/释放不配平（alloc=$allocs free=$frees）"
fi
grep -q "cuCtxDestroy" "$LOG" && ok "上下文已销毁" || bad "上下文没有销毁"
grep -qE "PTX ISA 8.0|PTX ISA 7.0" "$LOG" && ok "JIT 收到了合法的 PTX" || bad "PTX 内容不对"

if [ -n "$BIN" ] && [ -x "$BIN" ]; then
    echo "== 用例：原生版 $BIN =="
    out="$TMP/bin_check.txt"
    LD_LIBRARY_PATH="$MOCK" "$BIN" --check >"$out" 2>&1
    expect "原生版 --check" $? "$out" "自检通过"
    out="$TMP/bin_diag.txt"
    LD_LIBRARY_PATH="$MOCK" "$BIN" --diag >"$out" 2>&1
    expect "原生版 --diag" $? "$out" "diag 完成"
    out="$TMP/bin_run.txt"
    LD_LIBRARY_PATH="$MOCK" "$BIN" --duration 2 --stats 1 --target-ms 40 >"$out" 2>&1
    expect "原生版常驻刷" $? "$out" "已释放显存与 CUDA 上下文"
    out="$TMP/bin_sig.txt"
    LD_LIBRARY_PATH="$MOCK" "$BIN" --stats 0 --target-ms 40 >"$out" 2>&1 &
    pid=$!
    sleep 2
    kill -TERM "$pid" 2>/dev/null
    wait "$pid"
    code=$?
    [ "$code" -eq 0 ] && grep -qE "已释放显存" "$out" \
        && ok "原生版 SIGTERM 优雅退出" || bad "原生版 SIGTERM（退出码 $code）"
fi

echo
echo "通过 $pass 项，失败 $fail 项"
if [ "$fail" -ne 0 ]; then
    printf '失败: %s\n' "${FAILED[*]}"
    exit 1
fi
exit 0
