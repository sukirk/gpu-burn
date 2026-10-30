/*
 * mock_libcuda.c -- 假 libcuda.so.1，用来在没有 GPU 的机器上端到端验证 gpu_burn。
 *
 * 它不是 CUDA 模拟器：不执行 kernel，只是把 Driver API 的调用序列、参数合法性、
 * 分配/释放、异步时序（用 sleep 模拟 kernel 时长）都做出来，让被测程序走完
 * 「分配显存 -> JIT 载入 PTX -> 标定 -> 循环发射 -> 收信号 -> 释放」全流程。
 *
 * 故意做的严格校验（被测程序写错就会直接失败）：
 *   - cuModuleLoadData 必须收到含 ".entry gpu_burn_kernel" 的 PTX 文本；
 *   - cuLaunchKernel 的 kernelParams 里第 0 个参数必须是一个落在已分配区间内的
 *     设备指针，第 1 个必须是 >0 的迭代数（验证 ctypes 的 void** 封送是对的）；
 *   - cuMemsetD32Async / cuMemcpyDtoDAsync 的目标必须整体落在已分配区间内；
 *   - 句柄必须是本 mock 自己发出的（magic 校验），防止把 NULL/野指针传进来。
 *
 * 编译：gcc -O2 -fPIC -shared -Wl,-soname,libcuda.so.1 -o libcuda.so.1 mock_libcuda.c
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef uint64_t CUdeviceptr;

#define OK 0
#define E_INVALID_VALUE 1
#define E_OUT_OF_MEMORY 2
#define E_NOT_INITIALIZED 3
#define E_INVALID_CONTEXT 201
#define E_INVALID_HANDLE 400
#define E_INVALID_PTX 218
#define E_NOT_SUPPORTED 801

#define MAGIC_STREAM 0x53545245u
#define MAGIC_EVENT 0x45564e54u
#define MAGIC_CTX 0x43545830u

#define MAX_ALLOC 8192
#define MAX_STREAM 64

static int g_inited;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;  /* 多线程调用驱动 */
static pthread_mutex_t g_log_lock = PTHREAD_MUTEX_INITIALIZER;
static __thread int t_dev;                                   /* 本线程当前设备 */
static size_t g_total = 80ull << 30;
static size_t g_used_dev[64];                                /* 每张卡各自的占用 */
static int g_sms = 108;
static int g_devs = 1;               /* MOCK_DEVS：假装机器上有几张卡 */
static const char *g_name = "Mock A100-SXM4-80GB";
static double g_speed = 1.5e12;      /* 线程-迭代数/秒，逼近 A100 FP32 峰值 */
static size_t g_oom_after = 0;       /* 非 0：used 超过它之后一律 OOM */
static int g_fail_launch_at = 0;     /* 非 0：第 N 次 cuLaunchKernel 注入错误 */
static int g_fail_code = 999;        /* 注入的错误码（999 = CUDA_ERROR_UNKNOWN） */
static int g_launch_calls = 0;
static int g_fail_ctx_dev = -1;      /* MOCK_FAIL_CTX_DEV：让指定卡建上下文失败 */
static int g_sticky = 0;             /* 注入后模拟上下文丢失：后续调用也跟着失败 */
static FILE *g_log;

typedef struct { CUdeviceptr ptr; size_t size; int used; int dev; } alloc_t;
static alloc_t g_allocs[MAX_ALLOC];

typedef struct {
    unsigned magic;
    double busy_until;
} mock_stream_t;

typedef struct {
    unsigned magic;
    double t;
} mock_event_t;

typedef struct { unsigned magic; int dev; } mock_ctx_t;

/* stream 归线程所有：每张卡一个线程，各自建各自的 stream */
static __thread mock_stream_t *t_streams[MAX_STREAM];
static __thread int t_nstreams;

/* ------------------------------------------------------------------ */

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void sleep_until(double t)
{
    double d = t - now_s();
    if (d <= 0)
        return;
    if (d > 5.0)
        d = 5.0;
    struct timespec ts;
    ts.tv_sec = (time_t)d;
    ts.tv_nsec = (long)((d - (double)ts.tv_sec) * 1e9);
    nanosleep(&ts, NULL);
}

static void vlog(const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0)
        return;
    pthread_mutex_lock(&g_log_lock);
    if (g_log) {
        fwrite(buf, 1, (size_t)n, g_log);
        fputc('\n', g_log);
        fflush(g_log);
    }
    pthread_mutex_unlock(&g_log_lock);
}

static void env_init_locked(void)
{
    const char *s;
    if ((s = getenv("MOCK_TOTAL_GB")))
        g_total = (size_t)(atof(s) * (double)(1ull << 30));
    if ((s = getenv("MOCK_DEVS"))) {
        g_devs = atoi(s);
        if (g_devs < 1)
            g_devs = 1;
    }
    if ((s = getenv("MOCK_SMS")))
        g_sms = atoi(s);
    if ((s = getenv("MOCK_NAME")))
        g_name = s;
    if ((s = getenv("MOCK_SPEED")))
        g_speed = atof(s);
    if ((s = getenv("MOCK_OOM_AFTER_GB")))
        g_oom_after = (size_t)(atof(s) * (double)(1ull << 30));
    if ((s = getenv("MOCK_USED_GB"))) {   /* 模拟「被别的进程占掉」的显存，每张卡都占 */
        size_t used = (size_t)(atof(s) * (double)(1ull << 30));
        for (int i = 0; i < 64; i++)
            g_used_dev[i] = used;
    }
    if ((s = getenv("MOCK_FAIL_LAUNCH_AT")))
        g_fail_launch_at = atoi(s);
    if ((s = getenv("MOCK_FAIL_CODE")))
        g_fail_code = atoi(s);
    if ((s = getenv("MOCK_FAIL_CTX_DEV")))
        g_fail_ctx_dev = atoi(s);
    if ((s = getenv("MOCK_LOG")))
        g_log = fopen(s, "a");
}

static void env_init(void)
{
    if (g_inited)
        return;
    pthread_mutex_lock(&g_lock);
    if (!g_inited) {
        env_init_locked();
        g_inited = 1;
        pthread_mutex_unlock(&g_lock);
        vlog("mock: init total=%zu sms=%d devs=%d name=%s", g_total, g_sms, g_devs, g_name);
        return;
    }
    pthread_mutex_unlock(&g_lock);
}

static alloc_t *find_alloc(CUdeviceptr ptr, size_t size)
{
    for (int i = 0; i < MAX_ALLOC; i++) {
        alloc_t *a = &g_allocs[i];
        if (!a->used)
            continue;
        if (ptr >= a->ptr && ptr - a->ptr <= a->size && size <= a->size - (size_t)(ptr - a->ptr))
            return a;
    }
    return NULL;
}

static mock_stream_t *as_stream(CUstream s)
{
    mock_stream_t *m = (mock_stream_t *)s;
    if (!m || m->magic != MAGIC_STREAM)
        return NULL;
    return m;
}

static mock_event_t *as_event(CUevent e)
{
    mock_event_t *m = (mock_event_t *)e;
    if (!m || m->magic != MAGIC_EVENT)
        return NULL;
    return m;
}

static void note_sleep(double secs)
{
    if (secs > 0 && secs < 5.0)
        vlog("mock: simulate %.1f ms", secs * 1e3);
}

/* ------------------------------------------------------------------ */
/* Driver API                                                          */
/* ------------------------------------------------------------------ */

CUresult cuInit(unsigned int flags)
{
    env_init();
    vlog("mock: cuInit(%u)", flags);
    (void)flags;
    return OK;
}

CUresult cuDriverGetVersion(int *v)
{
    if (!v)
        return E_INVALID_VALUE;
    *v = 12090;
    return OK;
}

CUresult cuDeviceGetCount(int *n)
{
    if (!n)
        return E_INVALID_VALUE;
    *n = g_devs;
    return OK;
}

CUresult cuDeviceGet(CUdevice *d, int ordinal)
{
    if (!d)
        return E_INVALID_VALUE;
    if (ordinal < 0 || ordinal >= g_devs)
        return E_INVALID_VALUE;
    *d = ordinal;
    return OK;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
{
    if (!name || len <= 0)
        return E_INVALID_VALUE;
    if (dev < 0 || dev >= g_devs)
        return E_INVALID_VALUE;
    if (g_devs > 1)
        snprintf(name, (size_t)len, "%s #%d", g_name, (int)dev);
    else
        snprintf(name, (size_t)len, "%s", g_name);
    return OK;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
    (void)dev;
    if (!bytes)
        return E_INVALID_VALUE;
    *bytes = g_total;
    return OK;
}

CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev)
{
    (void)dev;
    if (!pi)
        return E_INVALID_VALUE;
    switch (attrib) {
    case 1:  *pi = 1024; break;           /* MAX_THREADS_PER_BLOCK */
    case 16: *pi = g_sms; break;          /* MULTIPROCESSOR_COUNT */
    case 75: *pi = 8; break;              /* CC_MAJOR */
    case 76: *pi = 0; break;              /* CC_MINOR */
    default: *pi = 0; break;
    }
    return OK;
}

CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice dev)
{
    (void)flags;
    (void)dev;
    if (!ctx)
        return E_INVALID_VALUE;
    if ((int)dev == g_fail_ctx_dev) {
        vlog("mock: cuCtxCreate(dev=%d) -> 注入失败 %d", (int)dev, g_fail_code);
        return (CUresult)g_fail_code;
    }
    mock_ctx_t *c = (mock_ctx_t *)calloc(1, sizeof(*c));
    c->magic = MAGIC_CTX;
    c->dev = (int)dev;
    t_dev = (int)dev;
    *ctx = c;
    vlog("mock: cuCtxCreate -> %p", (void *)c);
    return OK;
}

CUresult cuCtxDestroy_v2(CUcontext ctx)
{
    mock_ctx_t *c = (mock_ctx_t *)ctx;
    if (!c || c->magic != MAGIC_CTX)
        return E_INVALID_HANDLE;
    c->magic = 0;
    free(c);
    vlog("mock: cuCtxDestroy");
    return OK;
}

CUresult cuCtxSetCurrent(CUcontext ctx)
{
    mock_ctx_t *c = (mock_ctx_t *)ctx;
    if (!c || c->magic != MAGIC_CTX)
        return E_INVALID_CONTEXT;
    t_dev = c->dev;
    return OK;
}

CUresult cuCtxSynchronize(void)
{
    if (g_sticky)
        return g_sticky;
    double t = 0;
    for (int i = 0; i < t_nstreams; i++)
        if (t_streams[i]->busy_until > t)
            t = t_streams[i]->busy_until;
    note_sleep(t - now_s());
    sleep_until(t);
    vlog("mock: cuCtxSynchronize");
    return OK;
}

CUresult cuMemGetInfo_v2(size_t *free_, size_t *total)
{
    if (free_)
        *free_ = g_total - g_used_dev[t_dev];
    if (total)
        *total = g_total;
    return OK;
}

CUresult cuMemAlloc_v2(CUdeviceptr *ptr, size_t size)
{
    if (!ptr)
        return E_INVALID_VALUE;
    if (size == 0)
        return E_INVALID_VALUE;
    size_t cap = g_total;
    if (g_oom_after && cap > g_oom_after)
        cap = g_oom_after;
    size_t *used = &g_used_dev[t_dev];
    pthread_mutex_lock(&g_lock);
    if (*used + size > cap) {
        vlog("mock: cuMemAlloc(%zu) -> OUT_OF_MEMORY (dev=%d used=%zu)", size, t_dev, *used);
        pthread_mutex_unlock(&g_lock);
        return E_OUT_OF_MEMORY;
    }
    for (int i = 0; i < MAX_ALLOC; i++) {
        if (g_allocs[i].used)
            continue;
        g_allocs[i].used = 1;
        g_allocs[i].size = size;
        g_allocs[i].dev = t_dev;
        g_allocs[i].ptr = 0x7f0000000000ull + (uint64_t)i * 0x100000000ull;
        *used += size;
        *ptr = g_allocs[i].ptr;
        pthread_mutex_unlock(&g_lock);
        vlog("mock: cuMemAlloc(%zu) -> [%d] 0x%llx dev=%d used=%zu", size, i,
             (unsigned long long)*ptr, t_dev, *used);
        return OK;
    }
    pthread_mutex_unlock(&g_lock);
    return E_OUT_OF_MEMORY;
}

CUresult cuMemFree_v2(CUdeviceptr ptr)
{
    alloc_t *a = find_alloc(ptr, 0);
    if (!a || a->ptr != ptr)
        return E_INVALID_VALUE;
    pthread_mutex_lock(&g_lock);
    g_used_dev[a->dev] -= a->size;
    size_t left = g_used_dev[a->dev];
    int dev = a->dev;
    a->used = 0;
    pthread_mutex_unlock(&g_lock);
    vlog("mock: cuMemFree(0x%llx) dev=%d used=%zu", (unsigned long long)ptr, dev, left);
    return OK;
}

CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int value, size_t n, CUstream s)
{
    (void)value;
    if (!find_alloc(dst, n * 4)) {
        vlog("mock: cuMemsetD32Async dst=0x%llx n=%zu OUT OF RANGE",
             (unsigned long long)dst, n);
        return E_INVALID_VALUE;
    }
    mock_stream_t *m = as_stream(s);
    if (s && !m)
        return E_INVALID_HANDLE;
    if (m)
        m->busy_until = now_s() + (double)(n * 4) / 1.5e12;
    vlog("mock: cuMemsetD32Async n=%zu (%zu bytes)", n, n * 4);
    return OK;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s)
{
    if (!find_alloc(dst, n) || !find_alloc(src, n)) {
        vlog("mock: cuMemcpyDtoDAsync OUT OF RANGE");
        return E_INVALID_VALUE;
    }
    mock_stream_t *m = as_stream(s);
    if (s && !m)
        return E_INVALID_HANDLE;
    if (m)
        m->busy_until = now_s() + (double)n / 1.5e12;
    vlog("mock: cuMemcpyDtoDAsync %zu bytes", n);
    return OK;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image)
{
    if (!module || !image)
        return E_INVALID_VALUE;
    const char *ptx = (const char *)image;
    if (!strstr(ptx, ".version") || !strstr(ptx, ".entry gpu_burn_kernel") ||
        !strstr(ptx, "fma.rn.f32")) {
        vlog("mock: cuModuleLoadData -> INVALID_PTX (not the expected PTX)");
        return E_INVALID_PTX;
    }
    char ver[32] = "?";
    const char *p = strstr(ptx, ".version");
    sscanf(p, ".version %31s", ver);
    vlog("mock: cuModuleLoadData OK (PTX ISA %s, %zu bytes)", ver, strlen(ptx));
    static int dummy;
    *module = &dummy;
    return OK;
}

CUresult cuModuleUnload(CUmodule module)
{
    if (!module)
        return E_INVALID_HANDLE;
    vlog("mock: cuModuleUnload");
    return OK;
}

CUresult cuModuleGetFunction(CUfunction *fn, CUmodule module, const char *name)
{
    if (!fn || !module || !name)
        return E_INVALID_VALUE;
    if (strcmp(name, "gpu_burn_kernel") != 0) {
        vlog("mock: cuModuleGetFunction(%s) -> NOT FOUND", name);
        return E_INVALID_VALUE;
    }
    static int dummy;
    *fn = &dummy;
    return OK;
}

CUresult cuLaunchKernel(CUfunction fn, unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz,
                        unsigned int shared, CUstream s, void **params, void **extra)
{
    (void)gy; (void)gz; (void)by; (void)bz; (void)shared; (void)extra;
    if (g_sticky)
        return g_sticky;
    g_launch_calls++;
    if (g_fail_launch_at > 0 && g_launch_calls == g_fail_launch_at) {
        g_sticky = g_fail_code;
        vlog("mock: cuLaunchKernel 第 %d 次调用注入设备侧错误 %d（上下文丢失）", g_launch_calls,
             g_fail_code);
        return g_sticky;
    }
    if (!fn || !params)
        return E_INVALID_VALUE;
    if (gx == 0 || bx == 0 || bx > 1024)
        return E_INVALID_VALUE;
    uint64_t out_ptr = *(const uint64_t *)params[0];
    uint32_t iters = *(const uint32_t *)params[1];
    if (!find_alloc(out_ptr, 4)) {
        vlog("mock: cuLaunchKernel param0 0x%llx not inside any allocation (BAD SETUP)",
             (unsigned long long)out_ptr);
        return E_INVALID_VALUE;
    }
    if (iters == 0) {
        vlog("mock: cuLaunchKernel param1 iters=0 (BAD SETUP)");
        return E_INVALID_VALUE;
    }
    mock_stream_t *m = as_stream(s);
    if (!m)
        return E_INVALID_HANDLE;
    double threads = (double)gx * bx;
    double secs = ((double)iters * threads) / g_speed;
    m->busy_until = (m->busy_until > now_s() ? m->busy_until : now_s()) + secs;
    vlog("mock: cuLaunchKernel grid=%u block=%u iters=%u -> %.1f ms", gx, bx, iters, secs * 1e3);
    return OK;
}

CUresult cuStreamCreate(CUstream *stream, unsigned int flags)
{
    (void)flags;
    if (!stream)
        return E_INVALID_VALUE;
    mock_stream_t *m = (mock_stream_t *)calloc(1, sizeof(*m));
    m->magic = MAGIC_STREAM;
    if (t_nstreams < MAX_STREAM)
        t_streams[t_nstreams++] = m;
    *stream = m;
    return OK;
}

CUresult cuStreamDestroy_v2(CUstream s)
{
    mock_stream_t *m = as_stream(s);
    if (!m)
        return E_INVALID_HANDLE;
    for (int i = 0; i < t_nstreams; i++)
        if (t_streams[i] == m)
            t_streams[i] = t_streams[--t_nstreams];
    m->magic = 0;
    free(m);
    return OK;
}

CUresult cuStreamSynchronize(CUstream s)
{
    if (g_sticky)
        return g_sticky;
    mock_stream_t *m = as_stream(s);
    if (!m)
        return E_INVALID_HANDLE;
    sleep_until(m->busy_until);
    return OK;
}

CUresult cuEventCreate(CUevent *ev, unsigned int flags)
{
    (void)flags;
    if (!ev)
        return E_INVALID_VALUE;
    mock_event_t *e = (mock_event_t *)calloc(1, sizeof(*e));
    e->magic = MAGIC_EVENT;
    *ev = e;
    return OK;
}

CUresult cuEventRecord(CUevent ev, CUstream s)
{
    mock_event_t *e = as_event(ev);
    mock_stream_t *m = as_stream(s);
    if (!e)
        return E_INVALID_HANDLE;
    if (!m)
        return E_INVALID_HANDLE;
    e->t = m->busy_until;
    return OK;
}

CUresult cuEventSynchronize(CUevent ev)
{
    mock_event_t *e = as_event(ev);
    if (!e)
        return E_INVALID_HANDLE;
    sleep_until(e->t);
    return OK;
}

CUresult cuEventQuery(CUevent ev)
{
    mock_event_t *e = as_event(ev);
    if (!e)
        return E_INVALID_HANDLE;
    return now_s() >= e->t ? OK : 600; /* CUDA_ERROR_NOT_READY */
}

CUresult cuEventElapsedTime(float *ms, CUevent start, CUevent end)
{
    mock_event_t *a = as_event(start), *b = as_event(end);
    if (!ms || !a || !b)
        return E_INVALID_HANDLE;
    *ms = (float)((b->t - a->t) * 1e3);
    return OK;
}

CUresult cuEventDestroy_v2(CUevent ev)
{
    mock_event_t *e = as_event(ev);
    if (!e)
        return E_INVALID_HANDLE;
    e->magic = 0;
    free(e);
    return OK;
}

CUresult cuGetErrorString(CUresult code, const char **str)
{
    static const char *msgs[] = {"no error", "invalid value", "out of memory",
                                 "not initialized"};
    if (!str)
        return E_INVALID_VALUE;
    switch (code) {
    case 0: *str = msgs[0]; break;
    case 1: *str = msgs[1]; break;
    case 2: *str = msgs[2]; break;
    case 3: *str = msgs[3]; break;
    case 100: *str = "no device"; break;
    case 201: *str = "invalid context"; break;
    case 218: *str = "invalid PTX"; break;
    case 400: *str = "invalid handle"; break;
    case 801: *str = "not supported"; break;
    default: *str = "mock: unknown error"; break;
    }
    return OK;
}
