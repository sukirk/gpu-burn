/*
 * mock_libascendcl.c -- 假 libascendcl.so，用于在没有 NPU 的机器上端到端跑
 *                       gpu_burn_ascend 的测试。
 *
 * 思路和 CUDA 版的 tests/mock_libcuda.c 一样：导出和真库同名的符号，把设备、
 * 内存、算子调用都记账，并且严格校验参数（形状、dtype、指针归属），不做「一律
 * 返回成功」的假驱动——那样测出来的只是参数解析，测不出调用顺序和资源释放。
 *
 * 环境变量：
 *   MOCK_CHIPS      可见芯片数，默认 8
 *   MOCK_TOTAL_MB   每芯片 HBM 总量，默认 65536
 *   MOCK_USED_MB    每芯片预置的「别人已占用」量，默认 0
 *   MOCK_SOC        aclrtGetSocName 返回的名字，默认 Ascend910B2
 *   MOCK_FLOPS      模拟算力（FLOP/s），默认 300e12；MatMul 按 2*M*N*K/FLOPS 睡
 *   MOCK_LOG        日志文件（追加写），不设则不写日志
 *   MOCK_FAIL_AT    第 N 次 aclopExecuteV2 注入错误，0=不注入
 *   MOCK_FAIL_CODE  注入的错误码，默认 507000；注入后保持「粘性失败」
 *
 * 编译：gcc -O2 -Wall -Wextra -fPIC -shared -Wl,-soname,libascendcl.so -o libascendcl.so
 */

#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- 和 gpu_burn_ascend.c 里一致的 ABI 声明（假驱动自己声明一份） ---- */

typedef int aclError;
typedef void *aclrtStream;
typedef void *aclTensorDesc;
typedef void *aclDataBuffer;
typedef void *aclopAttr;

#define ACL_SUCCESS 0
#define ACL_FLOAT16 1
#define ACL_FORMAT_ND 2
#define ACL_MEM_MALLOC_HUGE_FIRST 0
#define ACL_HBM_MEM 1

/* 假驱动自己造的错误码，尽量贴近 CANN 真实码段 */
#define ERR_INVALID_PARAM 100000
#define ERR_INVALID_DEVICE 107001
#define ERR_INVALID_HANDLE 107003
#define ERR_INVALID_ADDR 107006
#define ERR_MALLOC 207000
#define ERR_DEFAULT_FAIL 507000

#define MAX_CHIPS 32
#define CHIP_STRIDE 0x1000000000ULL /* 每芯片 64 GiB 的假地址空间 */
#define CHIP_BASE 0x700000000000ULL

#define MAGIC_STREAM 0x5354524dU /* 'STRM' */
#define MAGIC_DESC 0x44455343U   /* 'DESC' */
#define MAGIC_BUF 0x44425546U    /* 'DBUF' */
#define MAGIC_ATTR 0x41545452U   /* 'ATTR' */

/* ---- 内部对象：假 stream / tensor 描述 / data buffer ---- */

typedef struct {
    uint32_t magic;
} FakeStream;

typedef struct {
    uint32_t magic;
    int dtype;
    int ndims;
    long long dims[8];
    int format;
} FakeDesc;

typedef struct {
    uint32_t magic;
    void *data;
    size_t size;
} FakeBuf;

typedef struct {
    uint32_t magic;
    int has[2];    /* transpose_a / transpose_b 是否设置过 */
    uint8_t val[2]; /* 两个属性的值 */
} FakeAttr;

/* ---- 全局状态（除了 t_chip 都在 g_mu 保护下） ---- */

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

static int g_chips;
static size_t g_total_bytes;
static size_t g_used[MAX_CHIPS];
static uint64_t g_off[MAX_CHIPS]; /* 每芯片内已分配的假地址偏移 */

typedef struct {
    int chip;
    uint64_t addr;
    size_t size;
} Block;

static Block *g_blocks;
static int g_nblocks;
static int g_capblocks;

static int g_inited;
static int g_last_err;
static int g_calls_matmul;
static unsigned long long g_fail_at;
static int g_fail_code;
static int g_fail_seen;

static FILE *g_fp;
static int g_fp_tried;

static __thread int t_chip = -1; /* aclrtSetDevice 设置的当前芯片 */

/* ---- 环境变量 / 日志 ---- */

static unsigned long long env_u64(const char *name, unsigned long long def)
{
    const char *s = getenv(name);
    if (!s || !*s)
        return def;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s)
        return def;
    return v;
}

static double env_double(const char *name, double def)
{
    const char *s = getenv(name);
    if (!s || !*s)
        return def;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s)
        return def;
    return v;
}

/* 第一次调用时把环境变量读进来（假驱动没有构造/析构钩子，懒加载最省事） */
static void lib_init(void)
{
    pthread_mutex_lock(&g_mu);
    if (!g_chips) {
        unsigned long long chips = env_u64("MOCK_CHIPS", 8);
        if (chips == 0)
            chips = 8;
        if (chips > MAX_CHIPS)
            chips = MAX_CHIPS;
        g_chips = (int)chips;
        unsigned long long mb = env_u64("MOCK_TOTAL_MB", 65536);
        if (mb == 0)
            mb = 65536;
        g_total_bytes = (size_t)mb * 1024u * 1024u;
        unsigned long long used_mb = env_u64("MOCK_USED_MB", 0);
        for (int i = 0; i < g_chips; i++)
            g_used[i] = (size_t)used_mb * 1024u * 1024u;
        g_fail_at = env_u64("MOCK_FAIL_AT", 0);
        g_fail_code = (int)env_u64("MOCK_FAIL_CODE", ERR_DEFAULT_FAIL);
    }
    pthread_mutex_unlock(&g_mu);
}

/* 加锁写一行日志；没有 MOCK_LOG 就什么都不做（调用方必须还没持有 g_mu） */
static void mock_log(const char *fmt, ...)
{
    lib_init();
    pthread_mutex_lock(&g_mu);
    if (!g_fp && !g_fp_tried) {
        g_fp_tried = 1;
        const char *path = getenv("MOCK_LOG");
        if (path && *path)
            g_fp = fopen(path, "a");
    }
    if (g_fp) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(g_fp, fmt, ap);
        va_end(ap);
        fputc('\n', g_fp);
        fflush(g_fp);
    }
    pthread_mutex_unlock(&g_mu);
}

static aclError set_err(int code)
{
    __atomic_store_n(&g_last_err, code, __ATOMIC_RELAXED);
    return code;
}

/* ---- 块表 ---- */

static Block *block_for(uint64_t addr)
{
    for (int i = 0; i < g_nblocks; i++)
        if (g_blocks[i].addr == addr)
            return &g_blocks[i];
    return NULL;
}

/* 指针是否落在这个芯片已分配的某块里（校验 memset / MatMul 的指针归属） */
static Block *block_containing(int chip, const void *p, size_t size)
{
    uint64_t addr = (uint64_t)(uintptr_t)p;
    for (int i = 0; i < g_nblocks; i++) {
        Block *b = &g_blocks[i];
        if (b->chip != chip)
            continue;
        if (addr >= b->addr && size <= b->size && addr + size <= b->addr + b->size)
            return b;
    }
    return NULL;
}

static void blocks_add(int chip, uint64_t addr, size_t size)
{
    if (g_nblocks == g_capblocks) {
        int cap = g_capblocks ? g_capblocks * 2 : 32;
        Block *nb = realloc(g_blocks, (size_t)cap * sizeof *nb);
        if (!nb)
            abort();
        g_blocks = nb;
        g_capblocks = cap;
    }
    g_blocks[g_nblocks].chip = chip;
    g_blocks[g_nblocks].addr = addr;
    g_blocks[g_nblocks].size = size;
    g_nblocks++;
}

/* ---- 导出符号 ---- */

aclError aclInit(const char *configPath)
{
    (void)configPath;
    lib_init();
    int already;
    pthread_mutex_lock(&g_mu);
    already = g_inited;
    if (!already)
        g_inited = 1;
    pthread_mutex_unlock(&g_mu);
    if (already)
        return set_err(ERR_INVALID_PARAM);
    mock_log("init");
    return ACL_SUCCESS;
}

aclError aclFinalize(void)
{
    lib_init();
    pthread_mutex_lock(&g_mu);
    g_inited = 0;
    pthread_mutex_unlock(&g_mu);
    mock_log("finalize");
    return ACL_SUCCESS;
}

const char *aclGetRecentErrMsg(void)
{
    lib_init();
    static char msg[256];
    int code = __atomic_load_n(&g_last_err, __ATOMIC_RELAXED);
    switch (code) {
    case 507000:
        snprintf(msg, sizeof msg,
                 "EE9999: [Mock] Device error: context lost / device exception "
                 "(模拟设备侧错误，见 MOCK_FAIL_AT)");
        break;
    case 207000:
        snprintf(msg, sizeof msg, "EE9999: [Mock] rtMalloc failed: out of HBM");
        break;
    case 107001:
        snprintf(msg, sizeof msg, "EE9999: [Mock] invalid device id");
        break;
    case 107003:
        snprintf(msg, sizeof msg, "EE9999: [Mock] invalid handle");
        break;
    case 107006:
        snprintf(msg, sizeof msg, "EE9999: [Mock] invalid memory address");
        break;
    case 100000:
        snprintf(msg, sizeof msg, "EE9999: [Mock] invalid parameter");
        break;
    case 0:
        snprintf(msg, sizeof msg, "EE9999: [Mock] no error");
        break;
    default:
        snprintf(msg, sizeof msg, "EE9999: [Mock] error %d", code);
        break;
    }
    return msg;
}

aclError aclrtGetDeviceCount(uint32_t *count)
{
    lib_init();
    if (!count)
        return set_err(ERR_INVALID_PARAM);
    *count = (uint32_t)g_chips;
    mock_log("device-count count=%d", g_chips);
    return ACL_SUCCESS;
}

aclError aclrtSetDevice(int32_t deviceId)
{
    lib_init();
    if (deviceId < 0 || deviceId >= g_chips) {
        mock_log("set-device-invalid id=%d chips=%d", (int)deviceId, g_chips);
        return set_err(ERR_INVALID_DEVICE);
    }
    t_chip = (int)deviceId;
    mock_log("set-device chip=%d", (int)deviceId);
    return ACL_SUCCESS;
}

aclError aclrtResetDevice(int32_t deviceId)
{
    lib_init();
    if (deviceId < 0 || deviceId >= g_chips) {
        mock_log("reset-device-invalid id=%d", (int)deviceId);
        return set_err(ERR_INVALID_DEVICE);
    }
    t_chip = -1;
    mock_log("reset-device chip=%d", (int)deviceId);
    return ACL_SUCCESS;
}

/* 假 stream 就是个带魔数的小句柄 */
aclError aclrtCreateStream(aclrtStream *stream)
{
    lib_init();
    if (!stream)
        return set_err(ERR_INVALID_PARAM);
    if (t_chip < 0) {
        mock_log("create-stream-no-device");
        return set_err(ERR_INVALID_DEVICE);
    }
    FakeStream *s = malloc(sizeof *s);
    if (!s)
        return set_err(ERR_MALLOC);
    s->magic = MAGIC_STREAM;
    *stream = s;
    mock_log("create-stream chip=%d", t_chip);
    return ACL_SUCCESS;
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    if (!stream)
        return set_err(ERR_INVALID_PARAM);
    FakeStream *s = stream;
    if (s->magic != MAGIC_STREAM) {
        mock_log("destroy-stream-invalid ptr=%p", (void *)stream);
        return set_err(ERR_INVALID_HANDLE);
    }
    free(s);
    mock_log("destroy-stream chip=%d", t_chip);
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    if (!stream)
        return set_err(ERR_INVALID_PARAM);
    FakeStream *s = stream;
    if (s->magic != MAGIC_STREAM) {
        mock_log("sync-invalid ptr=%p", (void *)stream);
        return set_err(ERR_INVALID_HANDLE);
    }
    mock_log("sync chip=%d", t_chip);
    return ACL_SUCCESS;
}

aclError aclrtMalloc(void **devPtr, size_t size, int policy)
{
    lib_init();
    if (!devPtr || size == 0)
        return set_err(ERR_INVALID_PARAM);
    if (policy != ACL_MEM_MALLOC_HUGE_FIRST && policy != 1 && policy != 2)
        return set_err(ERR_INVALID_PARAM);
    if (t_chip < 0)
        return set_err(ERR_INVALID_DEVICE);

    int chip = t_chip;
    size_t used_before, used_after;
    uint64_t addr;
    pthread_mutex_lock(&g_mu);
    used_before = g_used[chip];
    if (size > g_total_bytes - used_before || size > CHIP_STRIDE) {
        pthread_mutex_unlock(&g_mu);
        mock_log("malloc-fail chip=%d size=%zu used=%zu/%zu code=%d", chip, size, used_before,
                 g_total_bytes, ERR_MALLOC);
        return set_err(ERR_MALLOC);
    }
    addr = CHIP_BASE + (uint64_t)chip * CHIP_STRIDE + g_off[chip];
    g_off[chip] += (uint64_t)size;
    g_used[chip] += size;
    used_after = g_used[chip];
    blocks_add(chip, addr, size);
    pthread_mutex_unlock(&g_mu);

    *devPtr = (void *)(uintptr_t)addr;
    mock_log("malloc chip=%d addr=0x%llx size=%zu used=%zu/%zu", chip,
             (unsigned long long)addr, size, used_after, g_total_bytes);
    return ACL_SUCCESS;
}

aclError aclrtFree(void *devPtr)
{
    lib_init();
    if (!devPtr)
        return set_err(ERR_INVALID_PARAM);
    uint64_t addr = (uint64_t)(uintptr_t)devPtr;

    int chip, idx;
    size_t size, used_after;
    pthread_mutex_lock(&g_mu);
    Block *b = block_for(addr);
    if (!b) {
        pthread_mutex_unlock(&g_mu);
        mock_log("free-unknown addr=0x%llx code=%d", (unsigned long long)addr, ERR_INVALID_ADDR);
        return set_err(ERR_INVALID_ADDR);
    }
    chip = b->chip;
    size = b->size;
    idx = (int)(b - g_blocks);
    g_blocks[idx] = g_blocks[g_nblocks - 1];
    g_nblocks--;
    g_used[chip] -= size;
    used_after = g_used[chip];
    pthread_mutex_unlock(&g_mu);

    mock_log("free chip=%d addr=0x%llx size=%zu used=%zu/%zu", chip, (unsigned long long)addr, size,
             used_after, g_total_bytes);
    return ACL_SUCCESS;
}

aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count)
{
    lib_init();
    if (!devPtr || count == 0)
        return set_err(ERR_INVALID_PARAM);
    if (t_chip < 0)
        return set_err(ERR_INVALID_DEVICE);

    int chip = t_chip;
    int bad;
    pthread_mutex_lock(&g_mu);
    Block *b = block_containing(chip, devPtr, count);
    bad = (!b || maxCount > b->size);
    pthread_mutex_unlock(&g_mu);

    if (bad) {
        mock_log("memset-invalid chip=%d addr=0x%llx maxCount=%zu count=%zu", chip,
                 (unsigned long long)(uintptr_t)devPtr, maxCount, count);
        return set_err(ERR_INVALID_ADDR);
    }
    mock_log("memset chip=%d addr=0x%llx maxCount=%zu count=%zu value=%d", chip,
             (unsigned long long)(uintptr_t)devPtr, maxCount, count, (int)value);
    return ACL_SUCCESS;
}

aclError aclrtGetMemInfo(int memType, size_t *freeOut, size_t *totalOut)
{
    lib_init();
    if (!freeOut || !totalOut)
        return set_err(ERR_INVALID_PARAM);
    /* 真库各版本用法不一（0/1/2），假驱动全都认，方便被测程序探测 */
    if (memType < 0 || memType > 2)
        return set_err(ERR_INVALID_PARAM);
    if (t_chip < 0)
        return set_err(ERR_INVALID_DEVICE);

    int chip = t_chip;
    size_t used;
    pthread_mutex_lock(&g_mu);
    used = g_used[chip];
    *totalOut = g_total_bytes;
    *freeOut = g_total_bytes - used;
    pthread_mutex_unlock(&g_mu);
    return ACL_SUCCESS;
}

aclError aclrtGetSocName(char *socName, size_t len)
{
    lib_init();
    if (!socName || len < 8)
        return set_err(ERR_INVALID_PARAM);
    const char *soc = getenv("MOCK_SOC");
    if (!soc || !*soc)
        soc = "Ascend910B2";
    snprintf(socName, len, "%s", soc);
    mock_log("soc name=%s", socName);
    return ACL_SUCCESS;
}

aclTensorDesc *aclCreateTensorDesc(int dataType, int numDims, const long long *dims, int format)
{
    lib_init();
    if (dataType != ACL_FLOAT16 || format != ACL_FORMAT_ND)
        return NULL;
    if (numDims < 1 || numDims > 8 || !dims)
        return NULL;

    FakeDesc *d = calloc(1, sizeof *d);
    if (!d)
        return NULL;
    d->magic = MAGIC_DESC;
    d->dtype = dataType;
    d->ndims = numDims;
    d->format = format;
    for (int i = 0; i < numDims; i++) {
        if (dims[i] <= 0) {
            free(d);
            return NULL;
        }
        d->dims[i] = dims[i];
    }
    mock_log("desc-create dtype=%d ndims=%d dim0=%lld dim1=%lld format=%d", dataType, numDims,
             dims[0], numDims > 1 ? dims[1] : 0, format);
    return (aclTensorDesc *)d;
}

void aclDestroyTensorDesc(const aclTensorDesc *desc)
{
    if (!desc)
        return;
    FakeDesc *d = (FakeDesc *)(uintptr_t)desc;
    if (d->magic != MAGIC_DESC) {
        mock_log("desc-destroy-invalid ptr=%p", (const void *)desc);
        return;
    }
    free(d);
    mock_log("desc-destroy");
}

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size)
{
    lib_init();
    if (!data || size == 0)
        return NULL;
    FakeBuf *b = calloc(1, sizeof *b);
    if (!b)
        return NULL;
    b->magic = MAGIC_BUF;
    b->data = data;
    b->size = size;
    mock_log("buf-create addr=0x%llx size=%zu", (unsigned long long)(uintptr_t)data, size);
    return (aclDataBuffer *)b;
}

aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer)
{
    if (!dataBuffer)
        return set_err(ERR_INVALID_PARAM);
    FakeBuf *b = (FakeBuf *)(uintptr_t)dataBuffer;
    if (b->magic != MAGIC_BUF) {
        mock_log("buf-destroy-invalid ptr=%p", (const void *)dataBuffer);
        return set_err(ERR_INVALID_HANDLE);
    }
    free(b);
    mock_log("buf-destroy");
    return ACL_SUCCESS;
}

/* ---- 算子属性：MatMul 的 transpose_a/transpose_b ---- */

aclopAttr *aclopCreateAttr(void)
{
    lib_init();
    FakeAttr *a = calloc(1, sizeof *a);
    if (!a)
        return NULL;
    a->magic = MAGIC_ATTR;
    mock_log("attr-create");
    return (aclopAttr *)a;
}

aclError aclopSetAttrBool(aclopAttr *attr, const char *name, uint8_t value)
{
    if (!attr || !name)
        return set_err(ERR_INVALID_PARAM);
    FakeAttr *a = (FakeAttr *)(uintptr_t)attr;
    if (a->magic != MAGIC_ATTR) {
        mock_log("attr-set-invalid ptr=%p", (void *)attr);
        return set_err(ERR_INVALID_HANDLE);
    }
    int idx;
    if (strcmp(name, "transpose_a") == 0)
        idx = 0;
    else if (strcmp(name, "transpose_b") == 0)
        idx = 1;
    else {
        mock_log("attr-set-unknown name=%s", name);
        return set_err(ERR_INVALID_PARAM);
    }
    a->has[idx] = 1;
    a->val[idx] = value ? 1 : 0;
    mock_log("attr-set name=%s value=%d", name, (int)a->val[idx]);
    return ACL_SUCCESS;
}

void aclopDestroyAttr(const aclopAttr *attr)
{
    if (!attr)
        return;
    FakeAttr *a = (FakeAttr *)(uintptr_t)attr;
    if (a->magic != MAGIC_ATTR) {
        mock_log("attr-destroy-invalid ptr=%p", (const void *)attr);
        return;
    }
    free(a);
    mock_log("attr-destroy");
}

/* ---- 被重点校验的算子调用 ---- */

static aclError mm_invalid(const char *why, int chip)
{
    mock_log("matmul-invalid chip=%d reason=%s", chip, why);
    return set_err(ERR_INVALID_PARAM);
}

aclError aclopExecuteV2(const char *opType, int numInputs, aclTensorDesc *const inputDesc[],
                        aclDataBuffer *const inputs[], int numOutputs,
                        aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
                        aclopAttr *attr, aclrtStream stream)
{
    lib_init();

    int chip = t_chip;
    if (chip < 0)
        return set_err(ERR_INVALID_DEVICE);
    if (!stream || ((FakeStream *)stream)->magic != MAGIC_STREAM)
        return mm_invalid("bad-stream", chip);
    /* MatMul 必须带 transpose_a/transpose_b 两个属性，且都是 false（不转置） */
    if (!attr)
        return mm_invalid("null-attr", chip);
    FakeAttr *fa = (FakeAttr *)(uintptr_t)attr;
    if (fa->magic != MAGIC_ATTR)
        return mm_invalid("bad-attr-magic", chip);
    if (!fa->has[0] || !fa->has[1])
        return mm_invalid("attr-transpose-not-set", chip);
    if (fa->val[0] || fa->val[1])
        return mm_invalid("attr-transpose-true", chip);
    if (!opType || strcmp(opType, "MatMul") != 0)
        return mm_invalid("opType!=MatMul", chip);
    if (numInputs != 2)
        return mm_invalid("numInputs!=2", chip);
    if (numOutputs != 1)
        return mm_invalid("numOutputs!=1", chip);
    if (!inputDesc || !inputs || !outputDesc || !outputs)
        return mm_invalid("null-array", chip);
    if (!inputDesc[0] || !inputDesc[1] || !inputs[0] || !inputs[1] || !outputDesc[0] ||
        !outputs[0])
        return mm_invalid("null-entry", chip);

    FakeDesc *da = (FakeDesc *)(uintptr_t)inputDesc[0];
    FakeDesc *db = (FakeDesc *)(uintptr_t)inputDesc[1];
    FakeDesc *dc = (FakeDesc *)(uintptr_t)outputDesc[0];
    FakeBuf *ba = (FakeBuf *)(uintptr_t)inputs[0];
    FakeBuf *bb = (FakeBuf *)(uintptr_t)inputs[1];
    FakeBuf *bc = (FakeBuf *)(uintptr_t)outputs[0];

    if (da->magic != MAGIC_DESC || db->magic != MAGIC_DESC || dc->magic != MAGIC_DESC)
        return mm_invalid("bad-desc-magic", chip);
    if (ba->magic != MAGIC_BUF || bb->magic != MAGIC_BUF || bc->magic != MAGIC_BUF)
        return mm_invalid("bad-buffer-magic", chip);
    if (da->dtype != ACL_FLOAT16 || db->dtype != ACL_FLOAT16 || dc->dtype != ACL_FLOAT16)
        return mm_invalid("dtype!=FLOAT16", chip);
    if (da->format != ACL_FORMAT_ND || db->format != ACL_FORMAT_ND ||
        dc->format != ACL_FORMAT_ND)
        return mm_invalid("format!=ND", chip);
    if (da->ndims != 2 || db->ndims != 2 || dc->ndims != 2)
        return mm_invalid("ndims!=2", chip);

    long long M = da->dims[0], K = da->dims[1];
    long long K2 = db->dims[0], N = db->dims[1];
    if (M <= 0 || K <= 0 || N <= 0)
        return mm_invalid("bad-dim", chip);
    if (K2 != K)
        return mm_invalid("K-mismatch", chip);
    if (dc->dims[0] != M || dc->dims[1] != N)
        return mm_invalid("output-shape", chip);

    size_t want_a = (size_t)M * (size_t)K * 2;
    size_t want_b = (size_t)K * (size_t)N * 2;
    size_t want_c = (size_t)M * (size_t)N * 2;
    if (ba->size != want_a)
        return mm_invalid("A-size", chip);
    if (bb->size != want_b)
        return mm_invalid("B-size", chip);
    if (bc->size != want_c)
        return mm_invalid("C-size", chip);

    /* 三个指针必须都落在本芯片已分配的块里 */
    int bad_ptr;
    int call_no, sticky, fail_code;
    pthread_mutex_lock(&g_mu);
    bad_ptr = !block_containing(chip, ba->data, ba->size) ||
              !block_containing(chip, bb->data, bb->size) ||
              !block_containing(chip, bc->data, bc->size);
    if (!bad_ptr) {
        g_calls_matmul++;
        if (g_fail_at > 0 && (unsigned long long)g_calls_matmul >= g_fail_at)
            g_fail_seen = 1;
    }
    call_no = g_calls_matmul;
    sticky = g_fail_seen;
    fail_code = g_fail_code;
    pthread_mutex_unlock(&g_mu);

    if (bad_ptr)
        return mm_invalid("ptr-not-allocated", chip);

    if (sticky) {
        mock_log("matmul-fail chip=%d call=%d code=%d sticky=1", chip, call_no, fail_code);
        return set_err(fail_code);
    }

    /* 模拟算力：2*M*N*K 个 FLOP */
    unsigned long long flops =
        2ULL * (unsigned long long)M * (unsigned long long)N * (unsigned long long)K;
    double rate = env_double("MOCK_FLOPS", 300e12);
    if (rate <= 0.0)
        rate = 300e12;
    double sec = (double)flops / rate;
    struct timespec ts;
    ts.tv_sec = (time_t)sec;
    ts.tv_nsec = (long)((sec - (double)ts.tv_sec) * 1e9);
    if (ts.tv_nsec < 0)
        ts.tv_nsec = 0;
    /* 被信号打断就睡完剩下的：让 SIGTERM 测试的收尾时序更接近真机 */
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR)
        ;

    mock_log("matmul chip=%d call=%d M=%lld N=%lld K=%lld flops=%llu sleep_ms=%.3f", chip, call_no,
             M, N, K, flops, sec * 1000.0);
    return ACL_SUCCESS;
}
