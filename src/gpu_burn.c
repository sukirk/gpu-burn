/*
 * gpu_burn.c -- 把 CUDA 显存占满，并把 GPU 利用率刷到接近 100%。
 *
 * 只依赖 libc + libdl + libpthread：
 *   - 运行时 dlopen 宿主机提供的 libcuda.so.1，直接调 CUDA Driver API；
 *   - 计算 kernel 以内联 PTX 文本交给驱动 JIT，不需要 nvcc 或预编译的 cubin。
 *
 * 用法见 --help，编译见 Makefile。
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define VERSION "1.1.1"

/* ------------------------------------------------------------------------- */
/* CUDA Driver API 的最小类型/常量定义（避免依赖 cuda.h）                      */
/* ------------------------------------------------------------------------- */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef uint64_t CUdeviceptr;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2

#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

#define MAX_STREAMS 16
#define MAX_DEPTH 64
#define MB (1u << 20)

/* ------------------------------------------------------------------------- */
/* kernel：内联 PTX                                                           */
/* ------------------------------------------------------------------------- */

/*
 * 每个线程做一个固定次数的循环，循环体是 4 条互相独立的 FP32 FMA（加一个递减
 * 计数器和分支）。选 4 条独立 FMA 是为了让每个 SM 的 4 个 warp 调度器都能被占满：
 * 每轮 7 条指令里 4 条是 FMA，正好卡在 FP32 吞吐上，既不空转也不被整数指令拖走。
 * 累加器用 a = 0.5*a + seed 的形式，保证数值有界（不会 Inf/NaN），同时因为系数和
 * seed 都来自寄存器，ptxas 无法把它折叠掉。循环结束每线程写一次全局内存，
 * 防止整个循环被当成死代码删掉。
 */
#define KERNEL_HEADER                                                     \
    "//\n"                                                               \
    "// gpu_burn_kernel -- bounded FP32 FMA spin loop, used only to keep every SM busy.\n" \
    "//\n"                                                               \
    "// Kept as PTX text on purpose: the CUDA driver JITs it (via libcuda.so.1), so the\n" \
    "// program needs no CUDA toolkit, no nvcc and no per-architecture cubin.\n" \
    "//\n"                                                               \
    "//   params: (u64 out_ptr, u32 iters)\n"                            \
    "//   each thread: iters * 4 FMA on 4 independent accumulators, then one store so\n" \
    "//   ptxas cannot prove the loop dead.\n"                           \
    "//\n"

#define KERNEL_SIG                                                       \
    "\n.visible .entry gpu_burn_kernel(\n"                               \
    "    .param .u64 out_ptr,\n"                                         \
    "    .param .u32 iters\n"                                            \
    ")\n{\n"

#define KERNEL_BODY                                                      \
    "\n"                                                                 \
    "    .reg .pred  %p1;\n"                                             \
    "    .reg .b32   %r1, %r2;\n"                                        \
    "    .reg .b64   %rd1, %rd2;\n"                                      \
    "    .reg .f32   %f1, %f2, %f3, %f4, %f5, %f6;\n"                    \
    "\n"                                                                 \
    "    ld.param.u64    %rd1, [out_ptr];\n"                             \
    "    ld.param.u32    %r1,  [iters];\n"                               \
    "\n"                                                                 \
    "    mov.u32         %r2, %tid.x;\n"                                 \
    "    cvt.rn.f32.u32  %f5, %r2;           // per-thread seed\n"         \
    "    mov.f32         %f1, 0f3F800000;    // 1.0\n"                   \
    "    mov.f32         %f2, 0f3F800000;\n"                             \
    "    mov.f32         %f3, 0f3F800000;\n"                             \
    "    mov.f32         %f4, 0f3F800000;\n"                             \
    "    mov.f32         %f6, 0f3F000000;    // 0.5\n"                   \
    "\n"                                                                 \
    "LOOP:\n"                                                            \
    "    fma.rn.f32      %f1, %f1, %f6, %f5;\n"                          \
    "    fma.rn.f32      %f2, %f2, %f6, %f5;\n"                          \
    "    fma.rn.f32      %f3, %f3, %f6, %f5;\n"                          \
    "    fma.rn.f32      %f4, %f4, %f6, %f5;\n"                          \
    "    add.u32         %r1, %r1, -1;\n"                                \
    "    setp.gt.u32     %p1, %r1, 0;\n"                                 \
    "    @%p1 bra        LOOP;\n"                                        \
    "\n"                                                                 \
    "    add.f32         %f1, %f1, %f2;\n"                               \
    "    add.f32         %f3, %f3, %f4;\n"                               \
    "    add.f32         %f1, %f1, %f3;\n"                               \
    "\n"                                                                 \
    "    mad.wide.u32    %rd2, %r2, 4, %rd1;\n"                          \
    "    st.global.f32   [%rd2], %f1;\n"                                 \
    "    ret;\n"

/* 按顺序尝试：正常情况第一个就能被 JIT，后面几个只是给老/新驱动留退路
   （sm_80 + PTX ISA 7.0 是 CUDA 11.0 起就有的组合，A100 必然满足）。*/
typedef struct {
    const char *label;
    const char *isa;
    const char *target;
} KernelVariant;

static const KernelVariant KERNEL_VARIANTS[] = {
    {"sm_80 / PTX ISA 7.0", "7.0", "sm_80"},
    {"sm_80 / PTX ISA 8.0", "8.0", "sm_80"},
    {"sm_70 / PTX ISA 6.4", "6.4", "sm_70"},
};

static const char *DRIVER_CANDIDATES[] = {
    "libcuda.so.1",
    "libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
    "/usr/local/cuda/lib64/libcuda.so.1",
    "/usr/lib/wsl/lib/libcuda.so.1",
};

static const char *NVML_CANDIDATES[] = {
    "libnvidia-ml.so.1",
    "libnvidia-ml.so",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    "/usr/lib64/libnvidia-ml.so.1",
};

#define DRIVER_HINT                                                       \
    "libcuda.so.1 没能加载。这个库由宿主机的 NVIDIA 驱动提供，不在镜像里，需要容器\n" \
    "以 GPU 方式启动才会被注入：\n"                                        \
    "\n"                                                                  \
    "    docker run --gpus all -it <image>              # 需要宿主机装 nvidia-container-toolkit\n" \
    "    docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ...   # 老版本 docker\n" \
    "\n"                                                                  \
    "自检顺序：\n"                                                         \
    "  1) ls -l /dev/nvidia*                          # 有没有设备节点\n"  \
    "  2) ldconfig -p | grep libcuda                  # 有没有驱动库\n"    \
    "  3) nvidia-smi                                  # 宿主机驱动是否正常\n"

/* ------------------------------------------------------------------------- */
/* 小工具                                                                     */
/* ------------------------------------------------------------------------- */

static volatile sig_atomic_t g_stop;
static int g_quiet;
static int g_driver_loaded;
static int g_cleanup_quiet; /* --check 收尾时不要多打一行“已释放” */
static __thread int g_dev_tag = -1;  /* >=0 时日志里带上卡号：多卡模式下每张卡一个线程 */

static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;

/* 多卡模式是几个线程一起打日志：整行先拼好、再一次性写出去，
   否则两行会插到对方中间（"主动收工[gpu_burn 16:23:14 #1] 已释放..."）。 */
static void vlog_line(FILE *fp, const char *pre, const char *fmt, va_list ap)
{
    char buf[1200];
    int n = snprintf(buf, sizeof buf, "%s", pre);
    if (n < 0 || (size_t)n >= sizeof buf)
        n = 0;
    int m = vsnprintf(buf + n, sizeof buf - (size_t)n, fmt, ap);
    size_t len = (size_t)n + (m > 0 ? (size_t)m : 0);
    if (len > sizeof buf - 2)
        len = sizeof buf - 2;
    buf[len++] = '\n';
    pthread_mutex_lock(&g_log_mu);
    fwrite(buf, 1, len, fp);
    fflush(fp);
    pthread_mutex_unlock(&g_log_mu);
}

static void logmsg(const char *fmt, ...)
{
    if (g_quiet)
        return;
    char ts[16], pre[64];
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(ts, sizeof ts, "%H:%M:%S", &tm);
    if (g_dev_tag >= 0)
        snprintf(pre, sizeof pre, "[gpu_burn %s #%d] ", ts, g_dev_tag);
    else
        snprintf(pre, sizeof pre, "[gpu_burn %s] ", ts);
    va_list ap;
    va_start(ap, fmt);
    vlog_line(stdout, pre, fmt, ap);
    va_end(ap);
}

static void errmsg(const char *fmt, ...)
{
    char pre[64];
    if (g_dev_tag >= 0)
        snprintf(pre, sizeof pre, "[gpu_burn #%d] ERR: ", g_dev_tag);
    else
        snprintf(pre, sizeof pre, "[gpu_burn] ERR: ");
    va_list ap;
    va_start(ap, fmt);
    vlog_line(stderr, pre, fmt, ap);
    va_end(ap);
}

static void warnmsg(const char *fmt, ...)
{
    if (g_quiet)
        return;
    char pre[64];
    if (g_dev_tag >= 0)
        snprintf(pre, sizeof pre, "[gpu_burn #%d] WARN: ", g_dev_tag);
    else
        snprintf(pre, sizeof pre, "[gpu_burn] WARN: ");
    va_list ap;
    va_start(ap, fmt);
    vlog_line(stderr, pre, fmt, ap);
    va_end(ap);
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static char *human(char *buf, size_t n, uint64_t v)
{
    static const char *unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double d = (double)v;
    int i = 0;
    while (d >= 1024.0 && i < 4) {
        d /= 1024.0;
        i++;
    }
    if (i == 0)
        snprintf(buf, n, "%llu B", (unsigned long long)v);
    else
        snprintf(buf, n, "%.2f %s", d, unit[i]);
    return buf;
}

static void appendf(char *buf, size_t cap, size_t *len, const char *fmt, ...)
{
    if (*len >= cap - 1)
        return;
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(buf + *len, cap - *len, fmt, ap);
    va_end(ap);
    if (r > 0 && (size_t)r < cap - *len)
        *len += (size_t)r;
}

/* ------------------------------------------------------------------------- */
/* 驱动动态加载                                                               */
/* ------------------------------------------------------------------------- */

static struct {
    void *dll;
    const char *name;
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDriverGetVersion)(int *);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetName)(char *, int, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuDeviceTotalMem)(size_t *, CUdevice);
    CUresult (*cuCtxCreate)(CUcontext *, unsigned, CUdevice);
    CUresult (*cuCtxDestroy)(CUcontext);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuMemGetInfo)(size_t *, size_t *);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemsetD32Async)(CUdeviceptr, unsigned, size_t, CUstream);
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                               unsigned, unsigned, unsigned, CUstream, void **, void **);
    CUresult (*cuStreamCreate)(CUstream *, unsigned);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuGetErrorString)(CUresult, const char **);
} g_cu;

static int sym_required(void *dll, const char *name, void *dst)
{
    void *p = dlsym(dll, name);
    if (!p)
        return 0;
    *(void **)dst = p;
    return 1;
}

/* 先找 _v2 版本（真驱动与 mock 都提供），没有再退回不带版本的名字 */
static int sym_v2(void *dll, const char *name, void *dst)
{
    char buf[128];
    snprintf(buf, sizeof buf, "%s_v2", name);
    if (sym_required(dll, buf, dst))
        return 1;
    return sym_required(dll, name, dst);
}

#define BIND(field, name)                                                  \
    do {                                                                   \
        if (!sym_v2(dll, name, (void *)&g_cu.field)) {                     \
            missing = name;                                                \
            goto fail;                                                     \
        }                                                                  \
    } while (0)

static int load_driver(void)
{
    char errs[512] = "";
    for (size_t i = 0; i < sizeof DRIVER_CANDIDATES / sizeof *DRIVER_CANDIDATES; i++) {
        const char *cand = DRIVER_CANDIDATES[i];
        void *dll = dlopen(cand, RTLD_NOW | RTLD_GLOBAL);
        if (!dll) {
            size_t used = strlen(errs);
            snprintf(errs + used, sizeof errs - used, "%s；", dlerror());
            continue;
        }
        const char *missing = NULL;
        memset(&g_cu, 0, sizeof g_cu);
        g_cu.dll = dll;
        g_cu.name = cand;

        BIND(cuInit, "cuInit");
        BIND(cuDriverGetVersion, "cuDriverGetVersion");
        BIND(cuDeviceGetCount, "cuDeviceGetCount");
        BIND(cuDeviceGet, "cuDeviceGet");
        BIND(cuDeviceGetName, "cuDeviceGetName");
        BIND(cuDeviceGetAttribute, "cuDeviceGetAttribute");
        BIND(cuDeviceTotalMem, "cuDeviceTotalMem");
        BIND(cuCtxCreate, "cuCtxCreate");
        BIND(cuCtxDestroy, "cuCtxDestroy");
        BIND(cuCtxSetCurrent, "cuCtxSetCurrent");
        BIND(cuCtxSynchronize, "cuCtxSynchronize");
        BIND(cuMemGetInfo, "cuMemGetInfo");
        BIND(cuMemAlloc, "cuMemAlloc");
        BIND(cuMemFree, "cuMemFree");
        BIND(cuMemsetD32Async, "cuMemsetD32Async");
        BIND(cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync");
        BIND(cuModuleLoadData, "cuModuleLoadData");
        BIND(cuModuleUnload, "cuModuleUnload");
        BIND(cuModuleGetFunction, "cuModuleGetFunction");
        BIND(cuLaunchKernel, "cuLaunchKernel");
        BIND(cuStreamCreate, "cuStreamCreate");
        BIND(cuStreamDestroy, "cuStreamDestroy");
        BIND(cuStreamSynchronize, "cuStreamSynchronize");
        BIND(cuGetErrorString, "cuGetErrorString");

        g_driver_loaded = 1;
        logmsg("驱动库: %s", cand);
        return 1;
    fail:
        dlclose(dll);
        g_cu.dll = NULL;
        logmsg("WARN: %s 里缺符号 %s", cand, missing ? missing : "?");
    }
    errmsg("找不到可用的 libcuda.so.1");
    if (errs[0])
        fprintf(stderr, "[gpu_burn] ERR: 尝试过的失败原因: %s\n", errs);
    fputs(DRIVER_HINT, stderr);
    return 0;
}

/* CUDA 错误码带上数字：驱动对某些码只回 "unknown"，没有数字就无从查起 */
static const char *cu_err(CUresult rc)
{
    static char bufs[8][192];
    static int slot;
    char *buf = bufs[slot = (slot + 1) & 7];
    const char *s = NULL;
    if (!g_cu.cuGetErrorString || g_cu.cuGetErrorString(rc, &s) != CUDA_SUCCESS || !s)
        s = "unknown";
    snprintf(buf, sizeof bufs[0], "%s (rc=%d)", s, (int)rc);
    return buf;
}

/* 设备侧错误（非法访问 / 上下文丢失）的提示，省得用户对着 unknown 发呆 */
static void cuda_fail(const char *what, CUresult rc)
{
    errmsg("%s: %s", what, cu_err(rc));
    if ((int)rc == 999 || (int)rc == 719 || (int)rc == 700)
        errmsg("提示: 这是设备侧错误（kernel 非法访问 / 上下文丢失 / 驱动异常），"
               "加 --diag 逐步检查能找到失败的那一步");
}

/* ------------------------------------------------------------------------- */
/* 运行期状态                                                                 */
/* ------------------------------------------------------------------------- */

typedef struct {
    CUdeviceptr ptr;
    size_t size;
} Chunk;

#define MAX_DEVICES 32

typedef struct {
    int device;
    int devs[MAX_DEVICES];   /* 要刷的卡号列表（--devices 展开之后） */
    int ndevs;
    const char *devices_spec; /* --devices 的原始字符串，NULL 表示没给 */
    double mem_gb;      /* <0 表示用 mem_pct */
    double mem_pct;
    double duration;
    double stats;
    double target_ms;
    long iters_opt;
    int blocks_per_sm;
    int threads;
    int streams;
    int queue_depth;
    int churn;
    int churn_mb;
    int check;
    int diag;
} Args;

typedef struct Runtime {
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUstream streams[MAX_STREAMS];
    int nstreams;
    int inflight[MAX_STREAMS];
    int depth;
    Chunk *chunks;
    int nchunks;
    int capchunks;
    size_t got;
    CUdeviceptr sink;
    size_t sink_bytes;
    int block;
    int grid;
    uint32_t iters;
    unsigned long long launches;
    int sms;
    int max_threads;
    double started;
    int closed;
    pthread_t churn_tid;
    int churn_started;
    int nvml_ok;
    int failed;     /* 本卡失败次数：只影响本线程，不拖累别的卡 */
    int churn_stop; /* 通知 churn 线程收工（每张卡一个 churn 线程） */
} Runtime;

/* 一张卡一个线程，卡的状态放线程本地：CUDA 上下文是绑线程的，每张卡在自己
   线程里建上下文、分配显存、发 kernel，互不干扰；进程退出时所有卡一起释放。 */
static __thread Runtime R;

/* NVML 是可选增强：拿不到就当没有，只影响统计行里多几项 */
typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef struct {
    unsigned long long total;
    unsigned long long free_;
    unsigned long long used;
} nvmlMemory_t;

typedef struct {
    void *dll;
    int (*Init)(void);
    int (*Shutdown)(void);
    int (*GetHandleByIndex)(unsigned int, void **);
    int (*GetUtilization)(void *, nvmlUtilization_t *);
    int (*GetMemoryInfo)(void *, nvmlMemory_t *);
    int (*GetTemperature)(void *, int, unsigned int *);
    int (*GetPowerUsage)(void *, unsigned int *);
    void *dev;
} NvmlLib;

static __thread NvmlLib g_nvml;

static int load_nvml(void)
{
    for (size_t i = 0; i < sizeof NVML_CANDIDATES / sizeof *NVML_CANDIDATES; i++) {
        void *dll = dlopen(NVML_CANDIDATES[i], RTLD_NOW | RTLD_GLOBAL);
        if (!dll)
            continue;
        memset(&g_nvml, 0, sizeof g_nvml);
        g_nvml.dll = dll;
        if (!sym_required(dll, "nvmlInit_v2", (void *)&g_nvml.Init) ||
            !sym_required(dll, "nvmlShutdown", (void *)&g_nvml.Shutdown) ||
            !sym_v2(dll, "nvmlDeviceGetHandleByIndex", (void *)&g_nvml.GetHandleByIndex) ||
            !sym_required(dll, "nvmlDeviceGetUtilizationRates", (void *)&g_nvml.GetUtilization) ||
            !sym_required(dll, "nvmlDeviceGetMemoryInfo", (void *)&g_nvml.GetMemoryInfo) ||
            !sym_required(dll, "nvmlDeviceGetTemperature", (void *)&g_nvml.GetTemperature) ||
            !sym_required(dll, "nvmlDeviceGetPowerUsage", (void *)&g_nvml.GetPowerUsage)) {
            dlclose(dll);
            memset(&g_nvml, 0, sizeof g_nvml);
            continue;
        }
        return 1;
    }
    return 0;
}

static void cleanup_all(void);

/* ------------------------------------------------------------------------- */
/* 显存占用                                                                   */
/* ------------------------------------------------------------------------- */

static void add_chunk(CUdeviceptr p, size_t s)
{
    if (R.nchunks == R.capchunks) {
        int cap = R.capchunks ? R.capchunks * 2 : 64;
        Chunk *nc = realloc(R.chunks, (size_t)cap * sizeof *nc);
        if (!nc) {
            errmsg("内存不足，无法记录显存块");
            exit(3);
        }
        R.chunks = nc;
        R.capchunks = cap;
    }
    R.chunks[R.nchunks].ptr = p;
    R.chunks[R.nchunks].size = s;
    R.nchunks++;
}

/* 按块分配，单块失败就减半重试，尽量多占 */
static void alloc_vram(size_t want)
{
    const size_t chunk = (size_t)1 << 30;
    R.got = 0;
    while (R.got < want) {
        size_t size = chunk;
        if (size > want - R.got)
            size = want - R.got;
        for (;;) {
            CUdeviceptr p = 0;
            CUresult rc = g_cu.cuMemAlloc(&p, size);
            if (rc == CUDA_SUCCESS) {
                add_chunk(p, size);
                R.got += size;
                break;
            }
            if (rc != CUDA_ERROR_OUT_OF_MEMORY || size <= 32u * MB) {
                logmsg("显存分配到 %s 后停住（%s）", human((char[32]){0}, 32, R.got), cu_err(rc));
                return;
            }
            size /= 2;
        }
        if (R.nchunks % 8 == 0)
            logmsg("  已占 %s", human((char[32]){0}, 32, R.got));
    }
}

/* 把每块显存 memset 一遍：确认真的能用，同时让显存计数立刻反映出来 */
static void touch_vram(void)
{
    size_t total = 0;
    char b1[32], b2[32];
    for (int i = 0; i < R.nchunks; i++) {
        CUresult rc = g_cu.cuMemsetD32Async(R.chunks[i].ptr, 0, R.chunks[i].size / 4,
                                            R.streams[0]);
        if (rc != CUDA_SUCCESS) {
            errmsg("写入显存失败: %s", cu_err(rc));
            cleanup_all();
            R.failed = 1;
            return;
        }
        total += R.chunks[i].size;
        if ((i + 1) % 8 == 0 || i + 1 == R.nchunks) {
            rc = g_cu.cuStreamSynchronize(R.streams[0]);
            if (rc != CUDA_SUCCESS) {
                cuda_fail("写入显存后同步失败", rc);
                cleanup_all();
                R.failed = 1;
                return;
            }
            logmsg("  已写入 %s / %s", human(b1, sizeof b1, total),
                   human(b2, sizeof b2, R.got));
        }
    }
}

/* ------------------------------------------------------------------------- */
/* kernel 载入 / 发射 / 标定                                                  */
/* ------------------------------------------------------------------------- */

static int build_ptx(size_t idx, char *buf, size_t cap)
{
    const KernelVariant *v = &KERNEL_VARIANTS[idx];
    int n = snprintf(buf, cap, "%s.version %s\n.target %s\n.address_size 64\n%s%s}\n",
                     KERNEL_HEADER, v->isa, v->target, KERNEL_SIG, KERNEL_BODY);
    return (n > 0 && (size_t)n < cap) ? n : 0;
}

static int load_kernel(void)
{
    char ptx[4096];
    for (size_t i = 0; i < sizeof KERNEL_VARIANTS / sizeof *KERNEL_VARIANTS; i++) {
        const KernelVariant *v = &KERNEL_VARIANTS[i];
        if (!build_ptx(i, ptx, sizeof ptx)) {
            errmsg("PTX 文本超出缓冲区");
            return 0;
        }
        CUresult rc = g_cu.cuModuleLoadData(&R.mod, ptx);
        if (rc == CUDA_SUCCESS) {
            logmsg("kernel: 内联 PTX JIT 成功（%s）", v->label);
            break;
        }
        R.mod = NULL;
        logmsg("WARN: JIT %s 失败（%s），换下一个版本", v->label, cu_err(rc));
    }
    if (!R.mod) {
        errmsg("所有 PTX 版本都不能被驱动接受，检查驱动版本是否支持 CUDA 11+");
        return 0;
    }
    CUresult rc = g_cu.cuModuleGetFunction(&R.fn, R.mod, "gpu_burn_kernel");
    if (rc != CUDA_SUCCESS) {
        errmsg("找不到 kernel gpu_burn_kernel: %s", cu_err(rc));
        return 0;
    }
    return 1;
}

static CUresult launch(CUstream s, uint32_t iters)
{
    CUdeviceptr out = R.sink;
    void *params[2];
    params[0] = &out;
    params[1] = &iters;
    return g_cu.cuLaunchKernel(R.fn, (unsigned)R.grid, 1, 1, (unsigned)R.block, 1, 1, 0, s,
                               params, NULL);
}

/* 用 host 时钟 + stream 同步给一次 kernel 计时：不碰 event API，
   驱动面小一层，某些虚拟化/共享环境下 event 计时会直接报错。 */
static double timed_kernel_ms(uint32_t iters, CUresult *err)
{
    double t0 = now_s();
    CUresult rc = launch(R.streams[0], iters);
    if (rc == CUDA_SUCCESS)
        rc = g_cu.cuStreamSynchronize(R.streams[0]);
    double ms = (now_s() - t0) * 1000.0;
    if (err)
        *err = rc;
    return ms;
}

/* 试跑几次，把每线程迭代数标定到目标时长（换卡不用改参数） */
static uint32_t calibrate(double target_ms)
{
    double probe = 20000.0;
    for (int round = 0; round < 4; round++) {
        CUresult rc = CUDA_SUCCESS;
        double measured = timed_kernel_ms((uint32_t)probe, &rc);
        float ms = (float)measured;
        if (rc != CUDA_SUCCESS) {
            cuda_fail("标定失败", rc);
            cleanup_all();
            R.failed = 1;
            return 0;
        }
        if (ms >= 8.0f) {
            double it = probe * target_ms / (double)ms;
            if (it < 1000.0)
                it = 1000.0;
            if (it > 2e9)
                it = 2e9;
            logmsg("标定: %d 次迭代耗时 %.1f ms -> 目标 %.0f ms 用 %.0f 次迭代", (int)probe,
                   (double)ms, target_ms, it);
            return (uint32_t)it;
        }
        if (ms <= 0.0f) {
            probe *= 4.0; /* 计时器精度不够，加大量 */
            continue;
        }
        double m = target_ms / (double)ms;
        probe *= (m > 2.0 ? m : 2.0);
        if (probe < 1000.0)
            probe = 1000.0;
        if (probe > 2e9)
            probe = 2e9;
    }
    logmsg("标定: 计时不稳，退化用 %.0f 次迭代", probe);
    return (uint32_t)probe;
}

/* ------------------------------------------------------------------------- */
/* 显存带宽 churn                                                             */
/* ------------------------------------------------------------------------- */

typedef struct {
    CUdeviceptr a;
    CUdeviceptr b;
    size_t n;
    Runtime *owner; /* churn 是独立线程，看不到发起它的那张卡的线程本地状态 */
} ChurnArg;

static void *churn_main(void *arg)
{
    ChurnArg *ca = (ChurnArg *)arg;
    CUstream s = NULL;
    if (g_cu.cuCtxSetCurrent(ca->owner->ctx) != CUDA_SUCCESS) {
        warnmsg("churn 线程无法绑定上下文");
        return NULL;
    }
    if (g_cu.cuStreamCreate(&s, 0) != CUDA_SUCCESS)
        return NULL;
    int idx = 0;
    while (!g_stop && !ca->owner->churn_stop) {
        CUdeviceptr dst = (idx % 2 == 0) ? ca->a : ca->b;
        CUdeviceptr src = (idx % 2 == 0) ? ca->b : ca->a;
        if (g_cu.cuMemcpyDtoDAsync(dst, src, ca->n, s) != CUDA_SUCCESS) {
            warnmsg("churn 拷贝失败，线程退出");
            break;
        }
        idx++;
        if (idx % 2 == 0 && g_cu.cuStreamSynchronize(s) != CUDA_SUCCESS) {
            warnmsg("churn 同步失败，线程退出");
            break;
        }
    }
    if (s) {
        g_cu.cuStreamSynchronize(s);
        g_cu.cuStreamDestroy(s);
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */
/* 统计行                                                                     */
/* ------------------------------------------------------------------------- */

static void print_stats(const Args *a, double elapsed)
{
    char line[512];
    char b1[32], b2[32];
    size_t len = 0;
    (void)a;
    line[0] = '\0';
    appendf(line, sizeof line, &len, "运行 %6.1fs | kernel 发射 %llu 次", elapsed, R.launches);
    size_t free_ = 0, total = 0;
    if (g_cu.cuMemGetInfo(&free_, &total) == CUDA_SUCCESS)
        appendf(line, sizeof line, &len, " | 显存 %s/%s",
                human(b1, sizeof b1, total - free_), human(b2, sizeof b2, total));
    if (R.nvml_ok) {
        nvmlUtilization_t u;
        if (g_nvml.GetUtilization(g_nvml.dev, &u) == 0) {
            appendf(line, sizeof line, &len, " | GPU 利用率 %u%%", u.gpu);
            appendf(line, sizeof line, &len, " | 显存带宽 %u%%", u.memory);
        }
        unsigned temp = 0;
        if (g_nvml.GetTemperature(g_nvml.dev, 0, &temp) == 0)
            appendf(line, sizeof line, &len, " | %uC", temp);
        unsigned pw = 0;
        if (g_nvml.GetPowerUsage(g_nvml.dev, &pw) == 0)
            appendf(line, sizeof line, &len, " | %.0fW", pw / 1000.0);
    }
    logmsg("%s", line);
}

/* ------------------------------------------------------------------------- */
/* 清理                                                                       */
/* ------------------------------------------------------------------------- */

static void cleanup_all(void)
{
    if (R.closed)
        return;
    R.closed = 1;
    R.churn_stop = 1; /* 只停本卡的 churn 线程，别的卡的线程还在跑 */
    int had_ctx = R.ctx != NULL;
    if (!g_driver_loaded)
        return;

    if (R.churn_started) {
        pthread_join(R.churn_tid, NULL);
        R.churn_started = 0;
    }
    if (R.ctx) {
        if (g_cu.cuCtxSynchronize() != CUDA_SUCCESS)
            warnmsg("退出前 ctx sync 失败（可能上下文已丢失）");
    }
    for (int i = 0; i < R.nchunks; i++)
        g_cu.cuMemFree(R.chunks[i].ptr);
    R.nchunks = 0;
    free(R.chunks);
    R.chunks = NULL;
    R.capchunks = 0;
    if (R.sink) {
        g_cu.cuMemFree(R.sink);
        R.sink = 0;
    }
    for (int i = 0; i < R.nstreams; i++)
        if (R.streams[i]) {
            g_cu.cuStreamDestroy(R.streams[i]);
            R.streams[i] = NULL;
        }
    R.nstreams = 0;
    if (R.mod) {
        g_cu.cuModuleUnload(R.mod);
        R.mod = NULL;
    }
    if (R.ctx) {
        g_cu.cuCtxDestroy(R.ctx);
        R.ctx = NULL;
    }
    if (R.nvml_ok) {
        g_nvml.Shutdown();
        R.nvml_ok = 0;
    }
    if (!g_cleanup_quiet && had_ctx)
        logmsg("已释放显存与 CUDA 上下文");
}

/* ------------------------------------------------------------------------- */
/* 准备 / 运行                                                                */
/* ------------------------------------------------------------------------- */

static void *xcalloc(size_t n, size_t s)
{
    void *p = calloc(n, s);
    if (!p) {
        errmsg("内存分配失败");
        exit(3);
    }
    return p;
}

static void setup(const Args *a)
{
    CUresult rc = g_cu.cuInit(0);
    if (rc != CUDA_SUCCESS) {
        errmsg("cuInit 失败: %s", cu_err(rc));
        R.failed = 1;
        return;
    }
    int drv = 0;
    if (g_cu.cuDriverGetVersion(&drv) == CUDA_SUCCESS)
        logmsg("驱动 API 版本: %d.%d", drv / 1000, (drv % 1000) / 10);

    int count = 0;
    rc = g_cu.cuDeviceGetCount(&count);
    if (rc != CUDA_SUCCESS || count <= 0) {
        errmsg("看不到任何 GPU 设备（cuDeviceGetCount=%d）", count);
        R.failed = 1;
        return;
    }
    logmsg("可见设备数: %d", count);
    if (a->device < 0 || a->device >= count) {
        errmsg("--device %d 超出范围（共 %d 张卡）", a->device, count);
        R.failed = 1;
        return;
    }
    CUdevice dev = 0;
    g_cu.cuDeviceGet(&dev, a->device);

    char name[256] = "";
    if (g_cu.cuDeviceGetName(name, (int)sizeof name, dev) == CUDA_SUCCESS && name[0]) {
        /* ok */
    }
    int cc_major = 0, cc_minor = 0;
    g_cu.cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    g_cu.cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    g_cu.cuDeviceGetAttribute(&R.sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    g_cu.cuDeviceGetAttribute(&R.max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    logmsg("设备 #%d: %s  (sm_%d%d, %d SM)", a->device, name[0] ? name : "未知", cc_major,
           cc_minor, R.sms);
    if (R.max_threads <= 0)
        R.max_threads = 1024;

    rc = g_cu.cuCtxCreate(&R.ctx, 0, dev);
    if (rc != CUDA_SUCCESS) {
        errmsg("cuCtxCreate 失败: %s", cu_err(rc));
        R.failed = 1;
        return;
    }

    size_t free_ = 0, total = 0;
    rc = g_cu.cuMemGetInfo(&free_, &total);
    if (rc != CUDA_SUCCESS) {
        errmsg("cuMemGetInfo 失败: %s", cu_err(rc));
        R.failed = 1;
        return;
    }
    char b1[32], b2[32];
    logmsg("显存: 总 %s / 空闲 %s", human(b1, sizeof b1, total), human(b2, sizeof b2, free_));

    size_t want;
    if (a->mem_gb >= 0.0)
        want = (size_t)(a->mem_gb * (double)(1ull << 30));
    else
        want = (size_t)((double)free_ * a->mem_pct);
    size_t reserve = 256u * MB;
    size_t cap = (free_ > reserve) ? free_ - reserve : 0;
    if (want > cap)
        want = cap;
    if (want < 64u * MB) {
        errmsg("可占显存太少（空闲 %s），检查是否有别的进程占着",
               human(b1, sizeof b1, free_));
        R.failed = 1;
        return;
    }
    logmsg("计划占用: %s（%.0f%% 空闲显存）", human(b1, sizeof b1, want),
           free_ ? 100.0 * (double)want / (double)free_ : 0.0);

    rc = g_cu.cuStreamCreate(&R.streams[0], 0);
    if (rc != CUDA_SUCCESS) {
        errmsg("创建 stream 失败: %s", cu_err(rc));
        R.failed = 1;
        return;
    }
    R.nstreams = 1;

    /* 先把 kernel 输出缓冲（几 KB）拿到手，再去吃大块显存：万一显存被占满，
       至少 kernel 还能跑，程序不会因为“差 4KB”整体失败。 */
    R.sink_bytes = (size_t)(a->threads > 1024 ? a->threads : 1024) * 4;
    rc = g_cu.cuMemAlloc(&R.sink, R.sink_bytes);
    if (rc != CUDA_SUCCESS) {
        errmsg("无法分配 kernel 输出缓冲: %s", cu_err(rc));
        R.failed = 1;
        return;
    }

    double t0 = now_s();
    alloc_vram(want);
    if (R.got == 0) {
        errmsg("显存分配失败，一块都没拿到");
        R.failed = 1;
        return;
    }
    touch_vram();
    if (R.failed)
        return;
    logmsg("显存就绪: %s，耗时 %.1fs", human(b1, sizeof b1, R.got), now_s() - t0);

    if (!load_kernel()) {
        R.failed = 1;
        return;
    }

    R.block = a->threads;
    if (R.block > R.max_threads)
        R.block = R.max_threads;
    if (R.block < 32)
        R.block = 32;
    R.block -= R.block % 32;
    R.grid = R.sms * a->blocks_per_sm;
    if (R.grid < 1)
        R.grid = 1;

    if (a->iters_opt > 0)
        R.iters = (uint32_t)a->iters_opt;
    else
        R.iters = calibrate(a->target_ms);
    if (R.failed)
        return;
    logmsg("发射配置: grid=%d  block=%d  streams=%d  iters=%u", R.grid, R.block, a->streams,
           R.iters);

    for (int i = 1; i < a->streams; i++) {
        rc = g_cu.cuStreamCreate(&R.streams[i], 0);
        if (rc != CUDA_SUCCESS) {
            errmsg("创建 stream 失败: %s", cu_err(rc));
            R.failed = 1;
            return;
        }
        R.nstreams = i + 1;
    }
    R.depth = a->queue_depth >= 2 ? a->queue_depth : 2;
    if (R.depth > MAX_DEPTH)
        R.depth = MAX_DEPTH;
    for (int i = 0; i < R.nstreams; i++)
        R.inflight[i] = 0;

    /* NVML 只是锦上添花：拿不到就别打那几个字段 */
    if (load_nvml() && g_nvml.Init() == 0 &&
        g_nvml.GetHandleByIndex((unsigned)a->device, &g_nvml.dev) == 0) {
        R.nvml_ok = 1;
    } else if (g_nvml.dll) {
        dlclose(g_nvml.dll);
        memset(&g_nvml, 0, sizeof g_nvml);
        logmsg("NVML 不可用，统计行里没有温度/功耗/利用率（不影响刷 GPU）");
    }

    if (a->churn) {
        Chunk *ca = NULL, *cb = NULL;
        for (int i = 0; i < R.nchunks; i++) {
            if (R.chunks[i].size < 64u * MB)
                continue;
            if (!ca)
                ca = &R.chunks[i];
            else if (!cb) {
                cb = &R.chunks[i];
                break;
            }
        }
        if (!ca || !cb) {
            logmsg("WARN: 显存块不足 2 块（每块 >=64MiB），--churn 跳过了");
        } else {
            size_t n = ca->size < cb->size ? ca->size : cb->size;
            size_t limit = (size_t)a->churn_mb * MB;
            if (n > limit)
                n = limit;
            n -= n % 4;
            ChurnArg *arg = xcalloc(1, sizeof *arg);
            arg->a = ca->ptr;
            arg->b = cb->ptr;
            arg->n = n;
            arg->owner = &R;
            logmsg("显存带宽 churn 已启动（每轮 %s）", human(b1, sizeof b1, n));
            if (pthread_create(&R.churn_tid, NULL, churn_main, arg) == 0)
                R.churn_started = 1;
            else
                logmsg("WARN: churn 线程起不来");
        }
    }
}

static void run_loop(const Args *a)
{
    R.started = now_s();
    double next_stats = R.started + (a->stats > 0.0 ? a->stats : 1e18);
    while (!g_stop) {
        for (int i = 0; i < R.nstreams; i++) {
            CUresult rc = launch(R.streams[i], R.iters);
            if (rc != CUDA_SUCCESS) {
                if (g_stop)
                    break;
                cuda_fail("kernel 发射失败", rc);
                cleanup_all();
                R.failed = 1;
                return;
            }
            R.launches++;
            R.inflight[i]++;
            /* 每个 stream 排满 depth 个 kernel 就等这一批跑完：GPU 永远有活干，
               同时不会把无限的 kernel 堆在驱动队列里。 */
            if (R.inflight[i] >= R.depth) {
                rc = g_cu.cuStreamSynchronize(R.streams[i]);
                if (rc != CUDA_SUCCESS && !g_stop) {
                    cuda_fail("kernel 执行失败", rc);
                    cleanup_all();
                    R.failed = 1;
                    return;
                }
                R.inflight[i] = 0;
            }
        }
        double now = now_s();
        if (a->duration > 0.0 && now - R.started >= a->duration) {
            logmsg("到达 --duration %.0fs，主动收工", a->duration);
            break;
        }
        if (now >= next_stats) {
            print_stats(a, now - R.started);
            next_stats = now + a->stats;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* --diag：把整条链路拆成一步一步，哪一步炸就停在哪一步（真机排障用）          */
/* ------------------------------------------------------------------------- */

static void diag_err(int step, const char *what, CUresult rc)
{
    errmsg("diag %d) %s 失败: %s", step, what, cu_err(rc));
    if ((int)rc == 999 || (int)rc == 719 || (int)rc == 700)
        errmsg("提示: 设备侧错误。上面最后一条 OK 之后的那次调用出问题了；"
               "可以再试 --threads 128 --blocks-per-sm 1 --iters 2000");
}

#define DIAG_STEP(no, what, expr)                                              \
    do {                                                                       \
        CUresult _rc = (expr);                                                \
        if (_rc != CUDA_SUCCESS) {                                            \
            diag_err((no), (what), _rc);                                      \
            cleanup_all();                                                    \
            exit(3);                                                          \
        }                                                                     \
        logmsg("diag %d) %s: OK", (no), (what));                              \
    } while (0)

static void diag(const Args *a)
{
    int drv = 0;
    if (g_cu.cuDriverGetVersion(&drv) == CUDA_SUCCESS)
        logmsg("驱动 API 版本: %d.%d", drv / 1000, (drv % 1000) / 10);

    DIAG_STEP(1, "cuInit", g_cu.cuInit(0));

    int count = 0;
    DIAG_STEP(2, "cuDeviceGetCount", g_cu.cuDeviceGetCount(&count));
    logmsg("     可见设备数: %d", count);
    if (count <= 0) {
        errmsg("diag: 没有可见的 GPU");
        exit(3);
    }
    if (a->device < 0 || a->device >= count) {
        errmsg("diag: --device %d 超出范围（共 %d 张卡）", a->device, count);
        exit(3);
    }
    CUdevice dev = 0;
    DIAG_STEP(3, "cuDeviceGet", g_cu.cuDeviceGet(&dev, a->device));

    char name[256] = "";
    if (g_cu.cuDeviceGetName(name, (int)sizeof name, dev) != CUDA_SUCCESS)
        name[0] = '\0';
    int cc_major = 0, cc_minor = 0;
    g_cu.cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    g_cu.cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    g_cu.cuDeviceGetAttribute(&R.sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    g_cu.cuDeviceGetAttribute(&R.max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    if (R.max_threads <= 0)
        R.max_threads = 1024;
    logmsg("diag 4) 设备属性: %s sm_%d%d %d SM maxThreads/block=%d", name[0] ? name : "未知",
           cc_major, cc_minor, R.sms, R.max_threads);

    DIAG_STEP(5, "cuCtxCreate", g_cu.cuCtxCreate(&R.ctx, 0, dev));
    DIAG_STEP(6, "cuStreamCreate", g_cu.cuStreamCreate(&R.streams[0], 0));
    R.nstreams = 1;

    R.sink_bytes = 4096;
    DIAG_STEP(7, "cuMemAlloc 4KiB 输出缓冲",
              g_cu.cuMemAlloc(&R.sink, R.sink_bytes));

    size_t probe = 64u * MB;
    CUdeviceptr p = 0;
    DIAG_STEP(8, "cuMemAlloc 64MiB",
              g_cu.cuMemAlloc(&p, probe));
    add_chunk(p, probe);

    DIAG_STEP(9, "cuMemsetD32Async 64MiB",
              g_cu.cuMemsetD32Async(p, 0, probe / 4, R.streams[0]));
    DIAG_STEP(10, "cuStreamSynchronize（等 memset 落地）",
              g_cu.cuStreamSynchronize(R.streams[0]));

    if (!load_kernel()) {
        errmsg("diag 11) PTX JIT 失败");
        cleanup_all();
        exit(3);
    }
    logmsg("diag 11) PTX JIT（cuModuleLoadData + cuModuleGetFunction）: OK");

    R.block = 32;
    R.grid = R.sms > 0 ? R.sms : 1;
    DIAG_STEP(12, "小规模 launch（grid=SM block=32 iters=1000）",
              launch(R.streams[0], 1000));
    DIAG_STEP(13, "cuStreamSynchronize（小规模 kernel）",
              g_cu.cuStreamSynchronize(R.streams[0]));

    R.block = a->threads > R.max_threads ? R.max_threads : a->threads;
    R.block -= R.block % 32;
    R.grid = (R.sms > 0 ? R.sms : 1) * (a->blocks_per_sm > 0 ? a->blocks_per_sm : 1);
    double t0 = now_s();
    DIAG_STEP(14, "满规模 launch（grid=SM*blocks-per-sm iters=20000）",
              launch(R.streams[0], 20000));
    DIAG_STEP(15, "cuStreamSynchronize（满规模 kernel）",
              g_cu.cuStreamSynchronize(R.streams[0]));
    logmsg("     满规模 kernel 耗时 %.1f ms（grid=%d block=%d）", (now_s() - t0) * 1000.0,
           R.grid, R.block);

    cleanup_all();
    logmsg("diag 完成：从驱动到 kernel 这条链路可以跑");
}

#undef DIAG_STEP

/* ------------------------------------------------------------------------- */
/* --check：不常驻，只验证「驱动可见 / 能分配显存 / PTX 能 JIT / kernel 能跑」  */
/* ------------------------------------------------------------------------- */

static void self_check(const Args *a)
{
    CUresult rc = g_cu.cuInit(0);
    if (rc != CUDA_SUCCESS) {
        errmsg("cuInit 失败: %s", cu_err(rc));
        exit(3);
    }
    int drv = 0;
    if (g_cu.cuDriverGetVersion(&drv) == CUDA_SUCCESS)
        logmsg("驱动 API 版本: %d.%d", drv / 1000, (drv % 1000) / 10);
    int count = 0;
    rc = g_cu.cuDeviceGetCount(&count);
    if (rc != CUDA_SUCCESS || count <= 0) {
        errmsg("看不到任何 GPU 设备");
        exit(3);
    }
    logmsg("可见设备数: %d", count);
    CUdevice dev = 0;
    g_cu.cuDeviceGet(&dev, a->device < count ? a->device : 0);
    char name[256] = "";
    g_cu.cuDeviceGetName(name, (int)sizeof name, dev);
    int major = 0, minor = 0;
    g_cu.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    g_cu.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    g_cu.cuDeviceGetAttribute(&R.sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    g_cu.cuDeviceGetAttribute(&R.max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    logmsg("设备 #%d: %s  (sm_%d%d, %d SM)", a->device, name, major, minor, R.sms);

    char b1[32], b2[32];
    size_t free_ = 0, total = 0;
    g_cu.cuMemGetInfo(&free_, &total);
    logmsg("显存: 总 %s / 空闲 %s", human(b1, sizeof b1, total), human(b2, sizeof b2, free_));

    rc = g_cu.cuCtxCreate(&R.ctx, 0, dev);
    if (rc != CUDA_SUCCESS) {
        errmsg("cuCtxCreate 失败: %s", cu_err(rc));
        exit(3);
    }
    rc = g_cu.cuStreamCreate(&R.streams[0], 0);
    if (rc != CUDA_SUCCESS) {
        errmsg("创建 stream 失败: %s", cu_err(rc));
        exit(3);
    }
    R.nstreams = 1;

    size_t probe_bytes = 256u * MB;
    CUdeviceptr p = 0;
    rc = g_cu.cuMemAlloc(&p, probe_bytes);
    if (rc != CUDA_SUCCESS) {
        errmsg("分配 256MiB 显存失败: %s", cu_err(rc));
        exit(3);
    }
    add_chunk(p, probe_bytes);
    R.sink = p + (probe_bytes / 2);
    rc = g_cu.cuMemsetD32Async(p, 0, probe_bytes / 4, R.streams[0]);
    if (rc != CUDA_SUCCESS) {
        errmsg("写入显存失败: %s", cu_err(rc));
        exit(3);
    }
    g_cu.cuStreamSynchronize(R.streams[0]);
    logmsg("分配 + 写入 256MiB 测试显存: OK");

    if (!load_kernel()) {
        R.failed = 1;
        return;
    }

    R.block = a->threads;
    if (R.block > R.max_threads)
        R.block = R.max_threads;
    R.block -= R.block % 32;
    R.grid = R.sms * a->blocks_per_sm;
    uint32_t iters = calibrate(50.0);

    double ms = timed_kernel_ms(iters, &rc);
    if (rc != CUDA_SUCCESS) {
        cuda_fail("试跑 kernel 失败", rc);
        exit(3);
    }
    logmsg("kernel 试跑: OK，grid=%d block=%d iters=%u，耗时 %.1f ms", R.grid, R.block, iters,
           ms);

    if (load_nvml() && g_nvml.Init() == 0 &&
        g_nvml.GetHandleByIndex((unsigned)a->device, &g_nvml.dev) == 0) {
        R.nvml_ok = 1;
        nvmlUtilization_t u;
        nvmlMemory_t m;
        unsigned temp = 0, pw = 0;
        if (g_nvml.GetUtilization(g_nvml.dev, &u) == 0) {
            g_nvml.GetMemoryInfo(g_nvml.dev, &m);
            g_nvml.GetTemperature(g_nvml.dev, 0, &temp);
            g_nvml.GetPowerUsage(g_nvml.dev, &pw);
            logmsg("NVML: OK  gpu_util=%u mem_util=%u power_w=%.1f temp_c=%u vram_total=%llu "
                   "vram_used=%llu",
                   u.gpu, u.memory, pw / 1000.0, temp, m.total, m.used);
        }
    } else {
        logmsg("NVML 不可用（不影响刷 GPU，只是没有温度/功耗/利用率）");
    }

    /* 用同一个 cleanup 路径收尾，顺带验证资源释放 */
    R.sink = 0; /* 试跑用的输出缓冲指向 probe 块内部，别单独 free */
    g_cleanup_quiet = 1;
    cleanup_all();
    logmsg("自检通过：这个容器可以在当前这张卡上刷起来");
}

/* ------------------------------------------------------------------------- */
/* main                                                                       */
/* ------------------------------------------------------------------------- */

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

/* --device / --devices 解析成具体卡号。返回 0 成功，-1 失败（已打错误） */
static int resolve_devices(Args *a, int count)
{
    a->ndevs = 1;
    a->devs[0] = a->device;
    if (!a->devices_spec)
        return 0;

    if (strcmp(a->devices_spec, "all") == 0) {
        int n = count < MAX_DEVICES ? count : MAX_DEVICES;
        a->ndevs = n;
        for (int i = 0; i < n; i++)
            a->devs[i] = i;
        return 0;
    }

    a->ndevs = 0;
    const char *p = a->devices_spec;
    while (*p) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) {
            errmsg("--devices 解析不了: \"%s\"（写 0,1,2 或 all）", a->devices_spec);
            return -1;
        }
        if (v < 0 || v >= count) {
            errmsg("--devices 里的 %ld 超出范围（本机共 %d 张卡）", v, count);
            return -1;
        }
        int dup = 0;
        for (int i = 0; i < a->ndevs; i++)
            if (a->devs[i] == (int)v)
                dup = 1;
        if (!dup && a->ndevs < MAX_DEVICES)
            a->devs[a->ndevs++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ')
            p++;
    }
    if (a->ndevs == 0) {
        errmsg("--devices 没给任何卡号");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* 多卡：一个进程，每张卡一个线程                                             */
/*                                                                            */
/* 不在 fork 出来的子进程里建上下文：有些池化平台按进程分配 GPU 会话，子进程     */
/* 继承不了父进程的会话，重新申请就失败（bootstrap: Fail to init session）。     */
/* 所以走单进程多线程：一个进程一个会话，每张卡一个线程，各自建上下文、各自刷、   */
/* 各自释放。                                                                  */
/* 收尾也天然干净：进程一退（哪怕 kill -9）所有卡的显存都由驱动回收，不会留孤儿  */
/* 进程；收到 SIGINT/SIGTERM 时每个线程都走正常释放流程。                       */
/* ------------------------------------------------------------------------- */

typedef struct {
    const Args *a;
    int dev;
    int code;
} Worker;

static void *worker_main(void *p)
{
    Worker *w = (Worker *)p;
    Args a = *w->a; /* 每张卡一份自己的参数副本，改 device 不互相踩 */
    a.device = w->dev;
    if (w->a->ndevs > 1)
        g_dev_tag = w->dev; /* 多卡时日志带卡号，单卡保持原样 */

    if (w->a->diag) {
        diag(&a);
        w->code = 0;
        return NULL;
    }
    if (w->a->check) {
        self_check(&a);
        w->code = 0;
        return NULL;
    }
    setup(&a);
    if (!R.failed)
        run_loop(&a);
    cleanup_all();
    w->code = R.failed ? 3 : 0;
    return NULL;
}

static int run_multi(Args *a)
{
    logmsg("多卡模式: 目标卡 %d 张，每张卡一个线程（单进程，只有一个 CUDA 会话）", a->ndevs);

    Worker ws[MAX_DEVICES];
    pthread_t tids[MAX_DEVICES];
    int n = 0;
    for (int i = 0; i < a->ndevs; i++) {
        Worker *w = &ws[n];
        w->a = a;
        w->dev = a->devs[i];
        w->code = 3;
        if (pthread_create(&tids[n], NULL, worker_main, w) == 0) {
            n++;
        } else {
            errmsg("#%d 线程起不来，这张卡跳过", a->devs[i]);
        }
    }

    for (int i = 0; i < n; i++)
        pthread_join(tids[i], NULL);

    int worst = 0, bad = 0;
    for (int i = 0; i < n; i++) {
        if (ws[i].code != 0)
            bad++;
        if (ws[i].code > worst)
            worst = ws[i].code;
    }
    if (worst == 0)
        logmsg("全部 %d 张卡已收工，显存都已释放", n);
    else
        errmsg("%d 张卡里有 %d 张没能正常刷（错误见上面带卡号的行）", n, bad);
    return worst;
}

static void usage(FILE *out)
{
    fprintf(out,
            "用法: gpu_burn [选项]\n"
            "\n"
            "占满 CUDA 显存并把 GPU 利用率刷到接近 100%%（不依赖 PyTorch / CUDA Toolkit）\n"
            "\n"
            "选项:\n"
            "  --device N          GPU 序号（默认 0），等价于 --devices N\n"
            "  --devices LIST      要刷的卡号，逗号分隔或 all（如 0,1,2 / all）\n"
            "  --mem-gb GB         固定占用多少 GiB 显存（默认按 --mem-pct 算）\n"
            "  --mem-pct P         占空闲显存的比例，0~1（默认 0.60）\n"
            "  --duration SEC      运行多少秒后自动退出，0=一直跑（默认 0）\n"
            "  --stats SEC         多少秒打一次状态，0=不打（默认 5）\n"
            "  --target-ms MS      单个 kernel 的目标时长（默认 120）\n"
            "  --iters N           每线程迭代数，0=自动标定（默认 0）\n"
            "  --blocks-per-sm N   每个 SM 排多少个 block（默认 8）\n"
            "  --threads N         每个 block 的线程数（默认 256）\n"
            "  --streams N         并发 stream 数（默认 1）\n"
            "  --queue-depth N     每个 stream 最多预排的 kernel 数（默认 4）\n"
            "  --churn             另起线程做 D2D 拷贝，把显存带宽也刷起来\n"
            "  --churn-mb MB       churn 单次拷贝大小（默认 512）\n"
            "  --check             只做环境自检和短时试跑，然后退出\n"
            "  --diag              逐步检查驱动/CUDA 链路，打印每一步的结果（排障用）\n"
            "  --quiet             只打错误\n"
            "  --version           打印版本\n"
            "  -h, --help          打印本页\n"
            "\n"
            "示例:\n"
            "  gpu_burn                         # 占 60%% 空闲显存，一直刷，5 秒报一次\n"
            "  gpu_burn --mem-gb 70             # 固定占 70GiB\n"
            "  gpu_burn --duration 1800         # 刷 30 分钟\n"
            "  gpu_burn --churn --stats 2       # 顺带刷显存带宽\n"
            "  gpu_burn --devices all           # 每张卡一个子进程，全部刷起来\n"
            "  gpu_burn --devices 0,2           # 只刷 0 号和 2 号卡\n"
            "  gpu_burn --check                 # 只自检，不常驻\n");
}

int main(int argc, char **argv)
{
    Args a;
    memset(&a, 0, sizeof a);
    a.device = 0;
    a.mem_gb = -1.0;
    a.mem_pct = 0.60;
    a.duration = 0.0;
    a.stats = 5.0;
    a.target_ms = 120.0;
    a.iters_opt = 0;
    a.blocks_per_sm = 8;
    a.threads = 256;
    a.streams = 1;
    a.queue_depth = 4;
    a.churn = 0;
    a.churn_mb = 512;
    a.check = 0;

    static const struct option opt[] = {
        {"device", required_argument, 0, 'd'},
        {"devices", required_argument, 0, 'E'},
        {"mem-gb", required_argument, 0, 'g'},
        {"mem-pct", required_argument, 0, 'p'},
        {"duration", required_argument, 0, 'D'},
        {"stats", required_argument, 0, 's'},
        {"target-ms", required_argument, 0, 't'},
        {"iters", required_argument, 0, 'i'},
        {"blocks-per-sm", required_argument, 0, 'b'},
        {"threads", required_argument, 0, 'T'},
        {"streams", required_argument, 0, 'S'},
        {"queue-depth", required_argument, 0, 'q'},
        {"churn", no_argument, 0, 'c'},
        {"churn-mb", required_argument, 0, 'C'},
        {"check", no_argument, 0, 'k'},
        {"diag", no_argument, 0, 'X'},
        {"quiet", no_argument, 0, 'Q'},
        {"dump-ptx", no_argument, 0, 'P'},
        {"version", no_argument, 0, 'V'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0},
    };

    int c;
    while ((c = getopt_long(argc, argv, "h", opt, NULL)) != -1) {
        switch (c) {
        case 'd': a.device = atoi(optarg); break;
        case 'E': a.devices_spec = optarg; break;
        case 'g': a.mem_gb = atof(optarg); break;
        case 'p': a.mem_pct = atof(optarg); break;
        case 'D': a.duration = atof(optarg); break;
        case 's': a.stats = atof(optarg); break;
        case 't': a.target_ms = atof(optarg); break;
        case 'i': a.iters_opt = atol(optarg); break;
        case 'b': a.blocks_per_sm = atoi(optarg); break;
        case 'T': a.threads = atoi(optarg); break;
        case 'S': a.streams = atoi(optarg); break;
        case 'q': a.queue_depth = atoi(optarg); break;
        case 'c': a.churn = 1; break;
        case 'C': a.churn_mb = atoi(optarg); break;
        case 'k': a.check = 1; break;
        case 'X': a.diag = 1; break;
        case 'Q': g_quiet = 1; break;
        case 'P': {
            static char buf[4096];
            int n = build_ptx(0, buf, sizeof buf);
            if (n <= 0) {
                errmsg("PTX 文本超出缓冲区");
                return 3;
            }
            fwrite(buf, 1, (size_t)n, stdout);
            return 0;
        }
        case 'V':
            printf("gpu_burn %s\n", VERSION);
            return 0;
        case 'h': usage(stdout); return 0;
        default: usage(stderr); return 2;
        }
    }
    if (a.streams < 1)
        a.streams = 1;
    if (a.streams > MAX_STREAMS)
        a.streams = MAX_STREAMS;
    if (a.mem_pct <= 0.0 || a.mem_pct > 1.0)
        a.mem_pct = 0.60;
    if (a.threads < 32)
        a.threads = 32;
    if (a.threads > 1024)
        a.threads = 1024;
    if (a.duration < 0.0)
        a.duration = 0.0;

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);

    logmsg("gpu_burn %s，pid %d", VERSION, (int)getpid());
    if (!load_driver())
        return 2;

    /* 先把卡数问出来，"--devices all" 要据此展开 */
    int devcount = 0;
    if (g_cu.cuInit(0) == CUDA_SUCCESS)
        g_cu.cuDeviceGetCount(&devcount);
    if (devcount <= 0) {
        errmsg("看不到任何 GPU 设备（cuDeviceGetCount=%d）", devcount);
        return 3;
    }
    if (resolve_devices(&a, devcount) != 0)
        return 3;

    /* --diag / --check 本来就是一张卡一张卡看，串行跑，不用起线程 */
    if (a.ndevs > 1 && (a.diag || a.check)) {
        for (int i = 0; i < a.ndevs; i++) {
            a.device = a.devs[i];
            g_dev_tag = a.devs[i];
            if (a.diag)
                diag(&a);
            else
                self_check(&a);
            memset(&R, 0, sizeof R); /* 下一张卡从干净状态开始 */
            memset(&g_nvml, 0, sizeof g_nvml);
        }
        return 0;
    }
    if (a.ndevs > 1)
        return run_multi(&a);
    a.device = a.devs[0];

    if (a.diag) {
        diag(&a);
        return 0;
    }
    if (a.check) {
        self_check(&a);
        return 0;
    }
    setup(&a);
    if (!R.failed)
        run_loop(&a);
    cleanup_all();
    return R.failed ? 3 : 0;
}
