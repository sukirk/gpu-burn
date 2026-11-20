#!/usr/bin/env bash
#
# gpu_burn_ascend 的端到端测试。
#
# 全部用 tests/mock/libascendcl.so 这个假驱动跑，不需要 NPU、也不需要 CANN：
#   * 假驱动严格校验 MatMul 的形状/dtype/指针归属，不是「一律返回成功」；
#   * 用一个只认 mock 的 LD_LIBRARY_PATH 让被测程序 dlopen 到假库；
#   * 每条用例都检查退出码 + 关键输出，失败时最后打印相关日志尾部。
#
# 用法:
#   bash tests/run_tests.sh                # 测 ./gpu_burn_ascend
#   BIN=/path/to/gpu_burn_ascend bash tests/run_tests.sh
#
# 退出码: 0=全过，1=有失败。

set -u

cd "$(dirname "$0")/.." || exit 1

BIN=${BIN:-./gpu_burn_ascend}
MOCK_LIB=tests/mock
OUT=tests/.out

PASSED=0
FAILED=0
FAILED_CASES=""

hr() { printf '%s\n' '============================================================'; }
ok_case() {
    PASSED=$((PASSED + 1))
    printf '  ok   %s\n' "$1"
}
fail_case() {
    FAILED=$((FAILED + 1))
    FAILED_CASES="$FAILED_CASES
  - $1"
    printf '  FAIL %s\n' "$1"
}

# check_rc <用例名> <期望退出码> <实际退出码>
check_rc() {
    if [ "$3" -eq "$2" ]; then ok_case "$1（退出码 $2）"; else fail_case "$1：期望退出码 $2，实际 $3"; fi
}

# check_eq <用例名> <期望> <实际>
check_eq() {
    if [ "$3" = "$2" ]; then ok_case "$1（= $2）"; else fail_case "$1：期望 $2，实际 $3"; fi
}

# check_grep <用例名> <文件> <固定字符串>
check_grep() {
    if grep -qF -- "$3" "$2" 2>/dev/null; then
        ok_case "$1"
    else
        fail_case "$1：输出里找不到 \"$3\""
    fi
}

# 用假驱动跑被测程序：$1=假驱动日志文件，其余参数透传
mock_run() {
    local log=$1
    shift
    MOCK_LOG="$log" LD_LIBRARY_PATH="$MOCK_LIB" ASCENDCL_LIB= "$BIN" "$@"
}

# 扫 /proc/*/cmdline 数「被测二进制」的进程数：只比 argv[0]，避免把测试脚本
# 自己起的 grep/tr 命令行（参数里带 BIN 路径）也算进去。
# 不用 pgrep：最小容器里可能没有 procps。
count_burn_procs() {
    local n=0 d argv0
    for d in /proc/[0-9]*; do
        [ -r "$d/cmdline" ] || continue
        argv0=""
        IFS= read -r -d '' argv0 2>/dev/null <"$d/cmdline" || true
        if [ "$argv0" = "$BIN" ]; then
            n=$((n + 1))
        fi
    done
    printf '%s' "$n"
}

if [ ! -x "$BIN" ]; then
    echo "找不到被测二进制 $BIN，先 make" >&2
    exit 1
fi
if [ ! -f "$MOCK_LIB/libascendcl.so" ]; then
    echo "找不到假驱动 $MOCK_LIB/libascendcl.so，先 make mocks" >&2
    exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"
cleanup() { rm -rf "$OUT"; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1) 没有 libascendcl.so：退出码 2，提示库名和容器挂载
# ---------------------------------------------------------------------------
hr
echo "1) 找不到 libascendcl.so"
ASCENDCL_LIB=/nonexistent/libascendcl.so env -u LD_LIBRARY_PATH \
    "$BIN" --device 0 >"$OUT/t1.log" 2>&1
rc=$?
check_rc "1 无库时退出码" 2 "$rc"
check_grep "1 提示里有 libascendcl.so" "$OUT/t1.log" "libascendcl.so"
check_grep "1 提示里有容器设备节点挂载" "$OUT/t1.log" "/dev/davinci"
check_grep "1 提示里有 LD_LIBRARY_PATH 路径" "$OUT/t1.log" "/usr/local/Ascend/ascend-toolkit"

# ---------------------------------------------------------------------------
# 2) --check：自检通过
# ---------------------------------------------------------------------------
hr
echo "2) --check 自检"
mock_run "$OUT/t2.mock.log" --check >"$OUT/t2.log" 2>&1
rc=$?
check_rc "2 --check 退出码" 0 "$rc"
check_grep "2 输出有自检通过" "$OUT/t2.log" "自检通过"
check_grep "2 输出有 SoC 名称" "$OUT/t2.log" "Ascend910B2"
check_grep "2 输出有设备数" "$OUT/t2.log" "可见设备数: 8"
check_grep "2 mock 里 MatMul 参数被校验通过" "$OUT/t2.mock.log" "matmul chip=0"

# ---------------------------------------------------------------------------
# 3) --duration 3 --stats 1：有 TFLOPS 统计行，正常收工并释放 HBM
# ---------------------------------------------------------------------------
hr
echo "3) --duration 3 --stats 1"
mock_run "$OUT/t3.mock.log" --duration 3 --stats 1 >"$OUT/t3.log" 2>&1
rc=$?
check_rc "3 定时退出码" 0 "$rc"
check_grep "3 有 TFLOPS 统计行" "$OUT/t3.log" "TFLOPS"
check_grep "3 有收工总结" "$OUT/t3.log" "已收工"
check_grep "3 说明 HBM 都已释放" "$OUT/t3.log" "HBM 都已释放"
if grep -qE '实测 [0-9]+\.[0-9]+ TFLOPS' "$OUT/t3.log"; then
    ok_case "3 实测算力是有限数值"
else
    fail_case "3 统计行里没有形如 \"实测 X.X TFLOPS\" 的数字"
fi
alloc=$(grep -c '^malloc chip=0 ' "$OUT/t3.mock.log" || true)
free_=$(grep -c '^free chip=0 ' "$OUT/t3.mock.log" || true)
check_eq "3 单芯片 alloc/free 配平" "$alloc" "$free_"

# ---------------------------------------------------------------------------
# 4) --devices 0,2 与 --devices all：每芯片都有日志，且全程只有一个进程
# ---------------------------------------------------------------------------
hr
echo "4) 多芯片 --devices 0,2 / all（单进程多线程）"
run_two() { # $1=名字 $2=日志前缀，$3...=--devices 的参数
    local name=$1 prefix=$2
    shift 2
    LD_LIBRARY_PATH="$MOCK_LIB" ASCENDCL_LIB= MOCK_LOG="$OUT/$prefix.mock.log" \
        "$BIN" --devices "$1" --duration 2 --stats 0 >"$OUT/$prefix.log" 2>&1 &
    local pid=$!
    # 等它真的跑起来再数进程（CI 机器慢的时候 0.7s 可能还没起）
    local np=0 i
    for i in $(seq 1 30); do
        np=$(count_burn_procs)
        [ "$np" -ge 1 ] && break
        sleep 0.1
    done
    wait "$pid"
    local rc=$?
    check_rc "4 $name 退出码" 0 "$rc"
    check_eq "4 $name 运行中只有一个进程（扫 /proc/*/cmdline）" 1 "$np"
}

run_two "0,2" t4a "0,2"
check_grep "4 有 #0 日志" "$OUT/t4a.log" "#0"
check_grep "4 有 #2 日志" "$OUT/t4a.log" "#2"
check_grep "4 mock 里芯片 2 有 MatMul" "$OUT/t4a.mock.log" "matmul chip=2"
if grep -qF "#1" "$OUT/t4a.log"; then
    fail_case "4 只指定 0,2，不该出现 #1 的日志"
else
    ok_case "4 未指定的芯片 1 没有参与"
fi

run_two "all" t4b "all"
check_grep "4 all 有 #0 日志" "$OUT/t4b.log" "#0"
check_grep "4 all 有 #7 日志" "$OUT/t4b.log" "#7"
check_grep "4 all 时 8 个芯片都收工" "$OUT/t4b.log" "全部 8 个芯片已收工"

# ---------------------------------------------------------------------------
# 5) SIGTERM：干净退出，mock 里每个芯片 alloc/free 配平
# ---------------------------------------------------------------------------
hr
echo "5) SIGTERM 收尾"
LD_LIBRARY_PATH="$MOCK_LIB" ASCENDCL_LIB= MOCK_LOG="$OUT/t5.mock.log" \
    "$BIN" --devices 0,1 --stats 0 >"$OUT/t5.log" 2>&1 &
pid=$!
sleep 1
kill -TERM "$pid" 2>/dev/null
wait "$pid"
rc=$?
check_rc "5 SIGTERM 后退出码" 0 "$rc"
check_grep "5 输出有收工总结" "$OUT/t5.log" "已收工"
for chip in 0 1; do
    a=$(grep -c "^malloc chip=$chip " "$OUT/t5.mock.log" || true)
    f=$(grep -c "^free chip=$chip " "$OUT/t5.mock.log" || true)
    r=$(grep -c "^reset-device chip=$chip" "$OUT/t5.mock.log" || true)
    check_eq "5 芯片 $chip alloc/free 配平" "$a" "$f"
    if [ "$a" -gt 0 ]; then ok_case "5 芯片 $chip 确实分配过 HBM"; else fail_case "5 芯片 $chip 没有分配记录"; fi
    check_eq "5 芯片 $chip 已 reset device" 1 "$r"
done

# ---------------------------------------------------------------------------
# 6) --diag：逐步检查，走到 diag 完成
# ---------------------------------------------------------------------------
hr
echo "6) --diag"
mock_run "$OUT/t6.mock.log" --diag >"$OUT/t6.log" 2>&1
rc=$?
check_rc "6 --diag 退出码" 0 "$rc"
check_grep "6 输出有 diag 完成" "$OUT/t6.log" "diag 完成"
check_grep "6 每步有 OK" "$OUT/t6.log" "aclrtSetDevice: OK"
check_grep "6 走到 MatMul" "$OUT/t6.log" "aclopExecuteV2(MatMul 1024x1024x1024): OK"

# ---------------------------------------------------------------------------
# 7) 注入设备侧错误：退出码 3 且输出带错误码
# ---------------------------------------------------------------------------
hr
echo "7) MOCK_FAIL_AT=1 注入 507000"
MOCK_FAIL_AT=1 MOCK_FAIL_CODE=507000 LD_LIBRARY_PATH="$MOCK_LIB" ASCENDCL_LIB= \
    MOCK_LOG="$OUT/t7.mock.log" \
    "$BIN" --duration 5 --stats 0 >"$OUT/t7.log" 2>&1
rc=$?
check_rc "7 设备错误退出码" 3 "$rc"
check_grep "7 输出有错误码 507000" "$OUT/t7.log" "507000"
check_grep "7 指出是哪一步失败" "$OUT/t7.log" "aclopExecuteV2"
check_grep "7 mock 记录了粘性失败" "$OUT/t7.mock.log" "sticky=1"

# ---------------------------------------------------------------------------
# 8) --npu-smi：解析外部命令输出（用 shell 脚本假扮 npu-smi）
# ---------------------------------------------------------------------------
hr
echo "8) --npu-smi 解析"
cat >"$OUT/fake-npu-smi.sh" <<'SMI'
#!/bin/sh
cat <<'EOF'
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.0                   Version: 24.1.0                                               |
+------------------------------------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2               | OK            | 88.3        34                0    / 0             |
| 0                         | 0000:C1:00.0  | 96          1234 / 4096       3401 / 65536         |
+===========================+===============+====================================================+
| 1     910B2               | OK            | 92.1        36                0    / 0             |
| 0                         | 0000:01:00.0  | 93          1200 / 4096       3394 / 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name         | Process memory(MB)      |
+===========================+===============+======================+=========================+
| 0       0                 | 12345         | /usr/bin/python3     | 1234                    |
+===========================+===============+======================+=========================+
EOF
SMI
chmod +x "$OUT/fake-npu-smi.sh"
NPU_SMI_BIN="$OUT/fake-npu-smi.sh" mock_run "$OUT/t8.mock.log" \
    --npu-smi --duration 2 --stats 1 >"$OUT/t8.log" 2>&1
rc=$?
check_rc "8 --npu-smi 退出码" 0 "$rc"
check_grep "8 解析出 AICore%" "$OUT/t8.log" "AICore 96%"
check_grep "8 解析出 HBM 已用/总量" "$OUT/t8.log" "HBM 3401/65536 MB"
check_grep "8 解析出功耗" "$OUT/t8.log" "88.3W"
check_grep "8 解析出温度" "$OUT/t8.log" "34C"

# ---------------------------------------------------------------------------
# 9) 越界卡号：退出码 3
# ---------------------------------------------------------------------------
hr
echo "9) 越界卡号"
mock_run "$OUT/t9.mock.log" --devices 9 --duration 1 >"$OUT/t9.log" 2>&1
rc=$?
check_rc "9 越界卡号退出码" 3 "$rc"
check_grep "9 提示超出范围" "$OUT/t9.log" "超出范围"

# ---------------------------------------------------------------------------
# 附加：--version / --help / --mem-gb 固定占用量
# ---------------------------------------------------------------------------
hr
echo "10) 其他参数"
"$BIN" --version >"$OUT/t10a.log" 2>&1
check_rc "10 --version 退出码" 0 $?
check_grep "10 版本号" "$OUT/t10a.log" "gpu_burn_ascend 1.3.0"

"$BIN" --help >"$OUT/t10b.log" 2>&1
check_rc "10 --help 退出码" 0 $?
check_grep "10 帮助里有用法" "$OUT/t10b.log" "用法:"

mock_run "$OUT/t10c.mock.log" --mem-gb 2 --size 512 --duration 1 --stats 0 >"$OUT/t10c.log" 2>&1
rc=$?
check_rc "10 --mem-gb 固定占用量退出码" 0 "$rc"
check_grep "10 --mem-gb 生效" "$OUT/t10c.log" "目标 HBM 占用: 2.00 GiB"
check_grep "10 --mem-gb 实际占用 2 GiB" "$OUT/t10c.log" "已占 HBM: 2.00 GiB"

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
hr
printf '用例断言: %d 通过, %d 失败\n' "$PASSED" "$FAILED"
if [ "$FAILED" -ne 0 ]; then
    printf '失败的断言:%s\n' "$FAILED_CASES"
    echo
    echo '==== 相关日志尾部（完整日志在 $OUT，测试结束会清理）===='
    for f in "$OUT"/*.log; do
        [ -e "$f" ] || continue
        echo "---- $f ----"
        tail -n 15 "$f"
    done
    exit 1
fi
echo '全部通过'
exit 0
