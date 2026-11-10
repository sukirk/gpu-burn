/*
 * gpu_burn_ascend.c -- 昇腾 NPU 版「刷利用率」程序（单文件，不依赖 CANN 头文件）
 *
 * 设计目标：在装了 CANN 运行时的 910B2 机器上，把每张卡的 AICore 利用率和 HBM
 * 占用刷起来，同时不引入任何编译期依赖：ACL 的类型/常量/函数原型全部自己声明，
 * 运行时 dlopen("libascendcl.so") 再 dlsym 绑定。这样在没装 CANN 的开发机上也能
 * 编译，在容器里只要驱动库和 /dev/davinci* 挂进来就能跑。
 *
 * 和 CUDA 版 gpu_burn 保持同样的代码结构：
 *   参数解析 -> 日志 -> ACL 绑定 -> setup -> run_loop -> 统计 -> cleanup
 *
 * 多芯片必须是「一个进程 + 每芯片一个线程」：池化平台按进程分配 NPU 会话，
 * 子进程重新申请会失败，所以绝不用 fork。
 *
 * 编译：make（产物 ./gpu_burn_ascend）
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define VERSION "1.2.0"

#define KB (1u << 10)
#define MB (1u << 20)
#define GB (1u << 30)
#define MAX_DEVICES 32

/* ------------------------------------------------------------------------- */
/* ACL 类型 / 常量 / 函数原型                                                  */
/*                                                                            */
/* 全部按 CANN 文档的 ABI 自己声明：aclError 就是 int，句柄都是不透明指针。      */
/* 只用到刷利用率需要的一小撮 API。                                            */
/* ------------------------------------------------------------------------- */

typedef int aclError;
typedef void *aclrtStream;
typedef void *aclTensorDesc;
typedef void *aclDataBuffer;
typedef void *aclopAttr;

#define ACL_SUCCESS 0
#define ACL_FLOAT16 1              /* aclDataType: 半精度 */
#define ACL_FORMAT_ND 2            /* aclFormat: ND */
#define ACL_MEM_MALLOC_HUGE_FIRST 0 /* aclrtMemMallocPolicy */
#define ACL_HBM_MEM 1              /* aclrtMemType: HBM（老版本里也有 0/2 的写法，运行时探测） */

/* ------------------------------------------------------------------------- */
/* 小工具                                                                     */
/* ------------------------------------------------------------------------- */

static volatile sig_atomic_t g_stop;
static int g_quiet;
static int g_cleanup_quiet; /* --check/--diag 收尾时不要多打一行“已释放” */
static __thread int g_dev_tag = -1; /* >=0 时日志里带卡号：多芯片模式下每芯片一个线程 */

static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;

/* 多芯片是几个线程一起打日志：整行先拼好、一次性写出去，否则两行会插到对方中间 */
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

/* 统一的前缀：[gpu_burn_ascend HH:MM:SS] 或 [gpu_burn_ascend HH:MM:SS #N]，错误再多一层 */
static void make_prefix(char *pre, size_t cap, const char *level)
{
    char ts[16];
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(ts, sizeof ts, "%H:%M:%S", &tm);
    if (g_dev_tag >= 0)
        snprintf(pre, cap, "[gpu_burn_ascend %s #%d] %s", ts, g_dev_tag, level);
    else
        snprintf(pre, cap, "[gpu_burn_ascend %s] %s", ts, level);
}

static void logmsg(const char *fmt, ...)
{
    if (g_quiet)
        return;
    char pre[96];
    make_prefix(pre, sizeof pre, "");
    va_list ap;
    va_start(ap, fmt);
    vlog_line(stdout, pre, fmt, ap);
    va_end(ap);
}

static void errmsg(const char *fmt, ...)
{
    char pre[96];
    make_prefix(pre, sizeof pre, "ERR: ");
    va_list ap;
    va_start(ap, fmt);
    vlog_line(stderr, pre, fmt, ap);
    va_end(ap);
}

static void warnmsg(const char *fmt, ...)
{
    if (g_quiet)
        return;
    char pre[96];
    make_prefix(pre, sizeof pre, "WARN: ");
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
    if (cap == 0 || *len >= cap - 1)
        return;
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(buf + *len, cap - *len, fmt, ap);
    va_end(ap);
    if (r > 0 && (size_t)r < cap - *len)
        *len += (size_t)r;
}

/* ------------------------------------------------------------------------- */
/* 容器/环境提示                                                              */
/* ------------------------------------------------------------------------- */

#define LIB_HINT                                                               \
    "[gpu_burn_ascend] 提示: 需要 CANN 运行时库 libascendcl.so。容器里通常要挂宿主机的\n" \
    "[gpu_burn_ascend]       /usr/local/Ascend/driver 和 /usr/local/Ascend/ascend-toolkit，\n" \
    "[gpu_burn_ascend]       设备节点也要挂（/dev/davinci*），并让 LD_LIBRARY_PATH 包含：\n" \
    "[gpu_burn_ascend]         /usr/local/Ascend/ascend-toolkit/latest/lib64\n" \
    "[gpu_burn_ascend]         /usr/local/Ascend/driver/lib64\n"               \
    "[gpu_burn_ascend]       也可以用 ASCENDCL_LIB=/绝对路径/libascendcl.so 直接指定。\n"

#define NODE_HINT                                                              \
    "[gpu_burn_ascend] 提示: 容器里看不到 /dev/davinci*，需要把 NPU 设备节点挂进来，例如：\n" \
    "[gpu_burn_ascend]   docker run --rm -it \\\n"                              \
    "[gpu_burn_ascend]     --device=/dev/davinci0 --device=/dev/davinci_manager \\\n" \
    "[gpu_burn_ascend]     --device=/dev/devmm_svm --device=/dev/hisi_hdc \\\n" \
    "[gpu_burn_ascend]     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\\n" \
    "[gpu_burn_ascend]     -v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit \\\n" \
    "[gpu_burn_ascend]     -e LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64 \\\n" \
    "[gpu_burn_ascend]     <镜像> gpu_burn_ascend --check\n"                     \
    "[gpu_burn_ascend]   K8s 池化平台还需确认 pod 申请到了 NPU 资源（如 huawei.com/Ascend910）。\n"

/* 统计一下 /dev 下有没有 davinci 节点，只用于把「环境不对」和「ACL 调用失败」区分开 */
static int count_davinci_nodes(char *out, size_t cap)
{
    DIR *d = opendir("/dev");
    if (!d)
        return 0;
    int n = 0;
    size_t len = 0;
    struct dirent *e;
    if (cap)
        out[0] = '\0';
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "davinci", 7) != 0)
            continue;
        n++;
        if (out && len + 1 < cap)
            len += (size_t)snprintf(out + len, cap - len, "%s%s", len ? " " : "", e->d_name);
    }
    closedir(d);
    return n;
}

/* ------------------------------------------------------------------------- */
/* ACL 动态加载                                                               */
/* ------------------------------------------------------------------------- */

static struct {
    void *dll;
    const char *name;
    aclError (*aclInit)(const char *configPath);
    aclError (*aclFinalize)(void);
    const char *(*aclGetRecentErrMsg)(void);
    aclError (*aclrtGetDeviceCount)(uint32_t *count);
    aclError (*aclrtSetDevice)(int32_t deviceId);
    aclError (*aclrtResetDevice)(int32_t deviceId);
    aclError (*aclrtCreateStream)(aclrtStream *stream);
    aclError (*aclrtDestroyStream)(aclrtStream stream);
    aclError (*aclrtSynchronizeStream)(aclrtStream stream);
    aclError (*aclrtMalloc)(void **devPtr, size_t size, int policy);
    aclError (*aclrtFree)(void *devPtr);
    aclError (*aclrtMemset)(void *devPtr, size_t maxCount, int32_t value, size_t count);
    aclError (*aclrtGetMemInfo)(int memType, size_t *freeOut, size_t *totalOut);
    aclError (*aclrtGetSocName)(char *socName, size_t len);
    aclTensorDesc *(*aclCreateTensorDesc)(int dataType, int numDims, const int64_t *dims, int format);
    void (*aclDestroyTensorDesc)(const aclTensorDesc *desc);
    aclDataBuffer *(*aclCreateDataBuffer)(void *data, size_t size);
    aclError (*aclDestroyDataBuffer)(const aclDataBuffer *dataBuffer);
    aclError (*aclopExecuteV2)(const char *opType, int numInputs,
                               aclTensorDesc *const inputDesc[], aclDataBuffer *const inputs[],
                               int numOutputs, aclTensorDesc *const outputDesc[],
                               aclDataBuffer *const outputs[], aclopAttr *attr,
                               aclrtStream stream);
    /* MatMul 的默认属性（transpose_a/transpose_b=false）得靠 aclopAttr 传；
       这三个符号老版本库可能没有，缺了就退回 attr=NULL（多数版本能跑，个别版本
       会报 attr 未设置，--diag 能看出来） */
    aclopAttr *(*aclopCreateAttr)(void);
    aclError (*aclopSetAttrBool)(aclopAttr *attr, const char *name, uint8_t value);
    void (*aclopDestroyAttr)(const aclopAttr *attr);
} g_acl;

static int sym_required(void *dll, const char *name, void *dst)
{
    void *p = dlsym(dll, name);
    if (!p)
        return 0;
    *(void **)dst = p;
    return 1;
}

/* CANN 装在哪里：先试常见路径，再试环境变量指出的位置。ASCENDCL_LIB 可整体覆盖。 */
static int load_acl(void)
{
    static const char *cands[] = {
        "libascendcl.so",
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
        "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/libascendcl.so",
        "/usr/local/Ascend/latest/lib64/libascendcl.so",
        "/usr/local/Ascend/nnae/latest/fwkacllib/lib64/libascendcl.so",
        "/usr/local/Ascend/driver/lib64/libascendcl.so",
    };
    char dyn[4][512];
    const char *e;
    int ndyn = 0;
    if ((e = getenv("ASCEND_TOOLKIT_HOME")) != NULL && *e) {
        snprintf(dyn[ndyn], sizeof dyn[0], "%s/lib64/libascendcl.so", e);
        ndyn++;
    }
    if ((e = getenv("ASCEND_HOME_PATH")) != NULL && *e) {
        snprintf(dyn[ndyn], sizeof dyn[0], "%s/lib64/libascendcl.so", e);
        ndyn++;
    }
    if ((e = getenv("ASCEND_OPP_PATH")) != NULL && *e) {
        /* 一般是 .../opp，库在上一级 */
        snprintf(dyn[ndyn], sizeof dyn[0], "%s/../lib64/libascendcl.so", e);
        ndyn++;
    }

    const char *override = getenv("ASCENDCL_LIB");
    char errs[1024] = "";

    /* 先尽力把驱动侧的库用 RTLD_GLOBAL 载进来，libascendcl 的 DT_NEEDED 才能解析 */
    dlopen("/usr/local/Ascend/driver/lib64/libascend_hal.so", RTLD_NOW | RTLD_GLOBAL);

    for (int pass = 0; pass < 2; pass++) {
        int total = 0;
        const char *one = NULL;
        const char *list[16];
        if (pass == 0 && override && *override) {
            one = override;
            total = 1;
        }
        if (pass == 1 && (!override || !*override)) {
            total = (int)(sizeof cands / sizeof *cands);
            for (int i = 0; i < total; i++)
                list[i] = cands[i];
            int n = total;
            for (int i = 0; i < ndyn; i++)
                list[n++] = dyn[i];
            total = n;
        }
        if (one)
            list[0] = one;
        if (total == 0)
            continue;

        for (int i = 0; i < total; i++) {
            const char *cand = list[i];
            void *dll = dlopen(cand, RTLD_NOW | RTLD_GLOBAL);
            if (!dll) {
                const char *de = dlerror();
                size_t used = strlen(errs);
                snprintf(errs + used, sizeof errs - used, "%s；", de ? de : "dlopen 失败");
                continue;
            }
            const char *missing = NULL;
            memset(&g_acl, 0, sizeof g_acl);
            g_acl.dll = dll;
            g_acl.name = cand;

#define BIND(field, name)                                                      \
    do {                                                                       \
        if (!sym_required(dll, name, (void *)&g_acl.field)) {                  \
            missing = name;                                                    \
            goto fail;                                                         \
        }                                                                      \
    } while (0)

            BIND(aclInit, "aclInit");
            BIND(aclFinalize, "aclFinalize");
            BIND(aclGetRecentErrMsg, "aclGetRecentErrMsg");
            BIND(aclrtGetDeviceCount, "aclrtGetDeviceCount");
            BIND(aclrtSetDevice, "aclrtSetDevice");
            BIND(aclrtResetDevice, "aclrtResetDevice");
            BIND(aclrtCreateStream, "aclrtCreateStream");
            BIND(aclrtDestroyStream, "aclrtDestroyStream");
            BIND(aclrtSynchronizeStream, "aclrtSynchronizeStream");
            BIND(aclrtMalloc, "aclrtMalloc");
            BIND(aclrtFree, "aclrtFree");
            BIND(aclrtMemset, "aclrtMemset");
            BIND(aclCreateTensorDesc, "aclCreateTensorDesc");
            BIND(aclDestroyTensorDesc, "aclDestroyTensorDesc");
            BIND(aclCreateDataBuffer, "aclCreateDataBuffer");
            BIND(aclDestroyDataBuffer, "aclDestroyDataBuffer");
            BIND(aclopExecuteV2, "aclopExecuteV2");

            /* 这几个是可选增强：老版本没有也照样能刷 */
            sym_required(dll, "aclrtGetMemInfo", (void *)&g_acl.aclrtGetMemInfo);
            sym_required(dll, "aclrtGetSocName", (void *)&g_acl.aclrtGetSocName);
            {
                int have_attr = sym_required(dll, "aclopCreateAttr",
                                             (void *)&g_acl.aclopCreateAttr) &&
                                sym_required(dll, "aclopSetAttrBool",
                                             (void *)&g_acl.aclopSetAttrBool) &&
                                sym_required(dll, "aclopDestroyAttr",
                                             (void *)&g_acl.aclopDestroyAttr);
                if (!have_attr) {
                    g_acl.aclopCreateAttr = NULL;
                    g_acl.aclopSetAttrBool = NULL;
                    g_acl.aclopDestroyAttr = NULL;
                }
            }

#undef BIND

            logmsg("ACL 库: %s", cand);
            return 1;
        fail:
            dlclose(dll);
            g_acl.dll = NULL;
            logmsg("WARN: %s 里缺符号 %s", cand, missing ? missing : "?");
        }
    }

    errmsg("找不到可用的 libascendcl.so");
    if (errs[0])
        fprintf(stderr, "[gpu_burn_ascend] ERR: 尝试过的失败原因: %s\n", errs);
    fputs(LIB_HINT, stderr);
    return 0;
}

/* 错误码带上 aclGetRecentErrMsg 的文案，单看数字没法查 */
static const char *acl_str(aclError rc)
{
    static char bufs[8][384];
    static int slot;
    char *buf = bufs[slot = (slot + 1) & 7];
    const char *s = NULL;
    if (g_acl.aclGetRecentErrMsg)
        s = g_acl.aclGetRecentErrMsg();
    if (!s || !*s)
        s = "（aclGetRecentErrMsg 没有文案）";
    snprintf(buf, sizeof bufs[0], "%s (rc=%d)", s, (int)rc);
    return buf;
}

static void acl_fail(const char *what, aclError rc)
{
    errmsg("%s 失败: %s", what, acl_str(rc));
    if (rc == 507000 || rc == 507008 || rc == 507033 || rc == 107002 || rc == 107003)
        errmsg("提示: 这是设备侧错误（设备异常 / 上下文丢失 / 资源不足），"
               "加 --diag 能逐步定位是哪一次 ACL 调用出的问题");
}

/* ------------------------------------------------------------------------- */
/* npu-smi info 解析（--npu-smi 用）                                          */
/*                                                                            */
/* npu-smi 24.1.0 每张 NPU 占两行：                                            */
/*   | 0     910B2   | OK | 88.3  34  0 / 0           |   <- 板卡行：功耗/温度 */
/*   | 0             | 0000:C1:00.0 | 0  0/0 3401/65536 | <- 芯片行：AICore/HBM */
/* 字段位置在不同小版本上偶有漂移，所以按「第 2 列有没有 Bus-Id 冒号」判定行类型， */
/* 再按数字出现顺序取值，不做死板的列号匹配。                                   */
/* ------------------------------------------------------------------------- */

typedef struct {
    int npu;   /* 板卡号 */
    int chip;  /* 芯片号 */
    int dev;   /* 按 npu-smi 输出顺序编号的 acl device id（多芯板卡上是 npu*每板芯数+chip） */
    int aicore;
    long hbm_used;
    long hbm_total;
    int temp;
    double power;
    int has_chip;
    int has_temp;
    int has_power;
} NpuSmiEntry;

#define MAX_NPU_SMI 64

static NpuSmiEntry g_smi[MAX_NPU_SMI];
static int g_smi_n;
static int g_smi_chip_seen; /* 已经遇到几个芯片行，用来推断 acl 的 device id */
static int g_smi_ok;
static int g_smi_warned;
static double g_smi_at = -1e9;
static pthread_mutex_t g_smi_mu = PTHREAD_MUTEX_INITIALIZER;

/* 把一行里的数字按出现顺序抠出来（'/' 当分隔符），返回个数 */
static int numbers_in(const char *s, double *out, int max)
{
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        int start = isdigit((unsigned char)*p) ||
                    ((*p == '-' || *p == '+') &&
                     (isdigit((unsigned char)p[1]) || p[1] == '.')) ||
                    (*p == '.' && isdigit((unsigned char)p[1]));
        if (start) {
            char *end = NULL;
            double d = strtod(p, &end);
            if (end == p) {
                p++;
                continue;
            }
            out[n++] = d;
            p = end;
        } else {
            p++;
        }
    }
    return n;
}

static char *trim(char *s)
{
    while (*s && isspace((unsigned char)*s))
        s++;
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1]))
        s[--n] = '\0';
    return s;
}

static NpuSmiEntry *smi_entry_for(int npu, int chip)
{
    for (int i = 0; i < g_smi_n; i++)
        if (g_smi[i].npu == npu && g_smi[i].chip == chip)
            return &g_smi[i];
    if (g_smi_n >= MAX_NPU_SMI)
        return NULL;
    NpuSmiEntry *e = &g_smi[g_smi_n++];
    memset(e, 0, sizeof *e);
    e->npu = npu;
    e->chip = chip;
    return e;
}

static void parse_smi_line(char *line, int *pending_npu)
{
    char *fields[8];
    int nf = 0;
    char *p = line;
    while (nf < (int)(sizeof fields / sizeof *fields)) {
        char *q = strchr(p, '|');
        if (!q)
            break;
        *q = '\0';
        fields[nf++] = trim(p);
        p = q + 1;
    }
    if (nf < 4 || !fields[1][0] || !fields[2][0] || !fields[3][0])
        return;

    if (strchr(fields[2], ':') != NULL) {
        /* 芯片行：第 1 列是芯片号，第 4 列是 AICore% 内存用量 HBM 用量 */
        double v[8];
        int n = numbers_in(fields[3], v, 8);
        double id[2];
        int ni = numbers_in(fields[1], id, 2);
        if (n < 5 || ni < 1 || *pending_npu < 0)
            return;
        NpuSmiEntry *e = smi_entry_for(*pending_npu, (int)id[0]);
        if (!e || e->has_chip)
            return;
        e->dev = g_smi_chip_seen++;
        e->aicore = (int)v[0];
        e->hbm_used = (long)v[3];
        e->hbm_total = (long)v[4];
        e->has_chip = 1;
    } else if (strstr(fields[2], "OK") || strstr(fields[2], "Warn") ||
               strstr(fields[2], "Alarm") || strstr(fields[2], "Critical") ||
               strstr(fields[2], "Unknown")) {
        /* 板卡行：第 1 列开头是 NPU 号，第 4 列是 功耗 温度 大页使用/总数。
           表头行（Name/Health）和进程表（Process id）不会命中上面的健康状态词。 */
        double v[8];
        int n = numbers_in(fields[3], v, 8);
        double id[2];
        int ni = numbers_in(fields[1], id, 2);
        if (ni < 1)
            return;
        int npu = (int)id[0];
        if (npu < 0 || npu >= MAX_NPU_SMI)
            return;
        *pending_npu = npu;
        NpuSmiEntry *e = smi_entry_for(npu, 0);
        if (!e)
            return;
        if (n >= 2) {
            e->power = v[0];
            e->temp = (int)v[1];
            e->has_power = 1;
            e->has_temp = 1;
        }
    }
}

/* 跑一次 npu-smi info 并解析。返回 1 表示至少解析出一行芯片信息。 */
static int npu_smi_refresh(void)
{
    const char *cmd = getenv("NPU_SMI_BIN");
    if (!cmd || !*cmd)
        cmd = "npu-smi info";
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        warnmsg("跑不了 \"%s\"（%s）：统计行里不带 npu-smi 数据", cmd, strerror(errno));
        return 0;
    }
    g_smi_n = 0;
    g_smi_chip_seen = 0;
    char line[1024];
    int pending_npu = -1;
    while (fgets(line, sizeof line, fp) != NULL)
        parse_smi_line(line, &pending_npu);
    int status = pclose(fp);
    int chips = 0;
    for (int i = 0; i < g_smi_n; i++)
        if (g_smi[i].has_chip)
            chips++;
    if (chips == 0 && !g_smi_warned) {
        g_smi_warned = 1;
        warnmsg("\"%s\" 没解析出芯片行（退出码 %d）：--npu-smi 数据缺失，不影响刷利用率", cmd,
                status);
    }
    return chips > 0;
}

/* 取某张卡的 npu-smi 数据。带 1 秒缓存：多张卡同时打统计时只跑一次命令。 */
static int npu_smi_lookup(int dev, NpuSmiEntry *out)
{
    int found = 0;
    pthread_mutex_lock(&g_smi_mu);
    double now = now_s();
    if (now - g_smi_at >= 0.9) {
        g_smi_ok = npu_smi_refresh();
        g_smi_at = now;
    }
    if (g_smi_ok) {
        for (int i = 0; i < g_smi_n; i++) {
            if (g_smi[i].dev == dev && g_smi[i].has_chip) {
                *out = g_smi[i];
                found = 1;
                break;
            }
        }
    }
    pthread_mutex_unlock(&g_smi_mu);
    return found;
}

/* ------------------------------------------------------------------------- */
/* 运行期状态                                                                 */
/* ------------------------------------------------------------------------- */

typedef struct {
    void *ptr;
    size_t size;
} Chunk;

typedef struct {
    int device;
    int devs[MAX_DEVICES]; /* --devices 展开后的芯片列表 */
    int ndevs;
    const char *devices_spec;
    double mem_gb;  /* >0 表示用 --mem-gb */
    double mem_pct; /* 百分比 0~100 */
    double duration;
    double stats;
    long size;
    int queue;
    int check;
    int diag;
    int npu_smi;
} Args;

typedef struct Runtime {
    int dev;
    int dev_set;
    aclrtStream stream;
    Chunk *chunks;
    int nchunks;
    int capchunks;
    void *a;
    void *b;
    void *c;
    aclTensorDesc *desc_a;
    aclTensorDesc *desc_b;
    aclTensorDesc *desc_c;
    aclDataBuffer *buf_a;
    aclDataBuffer *buf_b;
    aclDataBuffer *buf_c;
    aclopAttr *attr;
    unsigned long long launches;
    size_t hbm_bytes;
    double started;
    int failed;
    int closed;
} Runtime;

/* 一张卡一个线程，卡的状态放线程本地：ACL 的 device 上下文是绑线程的 */
static __thread Runtime R;

/* aclrtGetMemInfo 的 memType 常量在 CANN 各版本里不完全一致，探测一次记住能用的 */
static __thread int t_meminfo_type = -2;

/* ------------------------------------------------------------------------- */
/* HBM 占用                                                                   */
/* ------------------------------------------------------------------------- */

static int add_chunk(void *p, size_t size)
{
    if (R.nchunks == R.capchunks) {
        int cap = R.capchunks ? R.capchunks * 2 : 16;
        Chunk *c = realloc(R.chunks, (size_t)cap * sizeof *c);
        if (!c) {
            errmsg("记录 HBM 块失败（宿主内存不足）");
            return -1;
        }
        R.chunks = c;
        R.capchunks = cap;
    }
    R.chunks[R.nchunks].ptr = p;
    R.chunks[R.nchunks].size = size;
    R.nchunks++;
    return 0;
}

/* 逐块吃 HBM：1 GiB 一块，OOM 就把块大小减半重试，减到 32 MiB 还拿不到就收手，
   已经拿到的量保留（占不满总比一块都占不到强）。 */
static int alloc_hbm(size_t want)
{
    size_t block = GB;
    size_t got = 0;
    int halved = 0;

    if (want == 0)
        return 0;

    while (got < want && !g_stop) {
        size_t ask = (want - got < block) ? (want - got) : block;
        void *p = NULL;
        aclError rc = g_acl.aclrtMalloc(&p, ask, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc == ACL_SUCCESS) {
            rc = g_acl.aclrtMemset(p, ask, 0, ask);
            if (rc != ACL_SUCCESS) {
                acl_fail("aclrtMemset（HBM 占位块）", rc);
                g_acl.aclrtFree(p);
                return -1;
            }
            if (add_chunk(p, ask) != 0) {
                g_acl.aclrtFree(p);
                return -1;
            }
            got += ask;
            continue;
        }
        if (block > 32u * MB) {
            block /= 2;
            if (block < 32u * MB)
                block = 32u * MB;
            halved = 1;
            continue;
        }
        break; /* 32 MiB 都拿不到就别硬试了，保留已拿到的量 */
    }
    if (halved) {
        char bb[32];
        logmsg("HBM 块大小减到 %s 才分配成功", human(bb, sizeof bb, block));
    }
    R.hbm_bytes = got;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* MatMul 资源                                                                */
/* ------------------------------------------------------------------------- */

static int probe_meminfo(size_t *free_out, size_t *total_out)
{
    if (!g_acl.aclrtGetMemInfo)
        return 0;
    if (t_meminfo_type == -1)
        return 0;
    if (t_meminfo_type >= 0) {
        size_t f = 0, t = 0;
        if (g_acl.aclrtGetMemInfo(t_meminfo_type, &f, &t) == ACL_SUCCESS && t > 0 && f <= t) {
            *free_out = f;
            *total_out = t;
            return 1;
        }
        return 0;
    }
    static const int try_types[] = {ACL_HBM_MEM, 0, 2};
    for (size_t i = 0; i < sizeof try_types / sizeof *try_types; i++) {
        size_t f = 0, t = 0;
        if (g_acl.aclrtGetMemInfo(try_types[i], &f, &t) == ACL_SUCCESS && t > 0 && f <= t) {
            t_meminfo_type = try_types[i];
            *free_out = f;
            *total_out = t;
            return 1;
        }
    }
    t_meminfo_type = -1;
    return 0;
}

static int matmul_buffer(const char *name, void **out, size_t bytes)
{
    void *p = NULL;
    aclError rc = g_acl.aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        acl_fail(name, rc);
        return -1;
    }
    rc = g_acl.aclrtMemset(p, bytes, 0, bytes);
    if (rc != ACL_SUCCESS) {
        acl_fail(name, rc);
        g_acl.aclrtFree(p);
        return -1;
    }
    if (add_chunk(p, bytes) != 0) {
        g_acl.aclrtFree(p);
        return -1;
    }
    *out = p;
    return 0;
}

static int matmul_alloc(long n, size_t *bytes_out)
{
    if (n < 1) {
        errmsg("--size 必须是正整数");
        return -1;
    }
    uint64_t nn = (uint64_t)n;
    if (nn > (uint64_t)(SIZE_MAX / 2) / nn) {
        errmsg("--size %ld 太大：矩阵字节数溢出 size_t", n);
        return -1;
    }
    size_t bytes = (size_t)(2 * nn * nn); /* FP16 每个元素 2 字节 */
    char hb[32];
    logmsg("MatMul 矩阵: M=N=K=%ld，FP16，A/B/C 各 %s", n, human(hb, sizeof hb, bytes));
    if (matmul_buffer("aclrtMalloc(MatMul A)", &R.a, bytes) != 0)
        return -1;
    if (matmul_buffer("aclrtMalloc(MatMul B)", &R.b, bytes) != 0)
        return -1;
    if (matmul_buffer("aclrtMalloc(MatMul C)", &R.c, bytes) != 0)
        return -1;
    *bytes_out = bytes;
    return 0;
}

static int matmul_build(long n, size_t bytes)
{
    int64_t dims[2] = {(int64_t)n, (int64_t)n};
    R.desc_a = g_acl.aclCreateTensorDesc(ACL_FLOAT16, 2, dims, ACL_FORMAT_ND);
    R.desc_b = g_acl.aclCreateTensorDesc(ACL_FLOAT16, 2, dims, ACL_FORMAT_ND);
    R.desc_c = g_acl.aclCreateTensorDesc(ACL_FLOAT16, 2, dims, ACL_FORMAT_ND);
    if (!R.desc_a || !R.desc_b || !R.desc_c) {
        errmsg("aclCreateTensorDesc 返回 NULL（宿主内存不足？）");
        return -1;
    }
    R.buf_a = g_acl.aclCreateDataBuffer(R.a, bytes);
    R.buf_b = g_acl.aclCreateDataBuffer(R.b, bytes);
    R.buf_c = g_acl.aclCreateDataBuffer(R.c, bytes);
    if (!R.buf_a || !R.buf_b || !R.buf_c) {
        errmsg("aclCreateDataBuffer 返回 NULL（宿主内存不足？）");
        return -1;
    }
    return 0;
}

/* MatMul 默认不转置，但 aclopExecuteV2 这条接口要求把属性显式塞进 aclopAttr；
   库里有 aclopCreateAttr 就建一个，没有就传 NULL由驱动用默认值。 */
static void matmul_attr(void)
{
    if (!g_acl.aclopCreateAttr)
        return;
    R.attr = g_acl.aclopCreateAttr();
    if (!R.attr) {
        warnmsg("aclopCreateAttr 返回 NULL，MatMul 将不带属性执行");
        return;
    }
    aclError rc = g_acl.aclopSetAttrBool(R.attr, "transpose_a", 0);
    if (rc == ACL_SUCCESS)
        rc = g_acl.aclopSetAttrBool(R.attr, "transpose_b", 0);
    if (rc != ACL_SUCCESS) {
        warnmsg("设置 MatMul 的 transpose 属性失败: %s（继续跑）", acl_str(rc));
        g_acl.aclopDestroyAttr(R.attr);
        R.attr = NULL;
    }
}

static int matmul_prepare(long n)
{
    size_t bytes = 0;
    if (matmul_alloc(n, &bytes) != 0)
        return -1;
    if (matmul_build(n, bytes) != 0)
        return -1;
    matmul_attr();
    return 0;
}

static aclError matmul_exec(void)
{
    aclTensorDesc *ins[2];
    aclDataBuffer *inb[2];
    aclTensorDesc *outs[1];
    aclDataBuffer *outb[1];
    ins[0] = R.desc_a;
    ins[1] = R.desc_b;
    inb[0] = R.buf_a;
    inb[1] = R.buf_b;
    outs[0] = R.desc_c;
    outb[0] = R.buf_c;
    return g_acl.aclopExecuteV2("MatMul", 2, ins, inb, 1, outs, outb, R.attr, R.stream);
}

static void matmul_release(void)
{
    if (R.attr) {
        g_acl.aclopDestroyAttr(R.attr);
        R.attr = NULL;
    }
    if (R.buf_a) {
        g_acl.aclDestroyDataBuffer(R.buf_a);
        R.buf_a = NULL;
    }
    if (R.buf_b) {
        g_acl.aclDestroyDataBuffer(R.buf_b);
        R.buf_b = NULL;
    }
    if (R.buf_c) {
        g_acl.aclDestroyDataBuffer(R.buf_c);
        R.buf_c = NULL;
    }
    if (R.desc_a) {
        g_acl.aclDestroyTensorDesc(R.desc_a);
        R.desc_a = NULL;
    }
    if (R.desc_b) {
        g_acl.aclDestroyTensorDesc(R.desc_b);
        R.desc_b = NULL;
    }
    if (R.desc_c) {
        g_acl.aclDestroyTensorDesc(R.desc_c);
        R.desc_c = NULL;
    }
}

/* ------------------------------------------------------------------------- */
/* 清理                                                                       */
/* ------------------------------------------------------------------------- */

static void cleanup_chip(void)
{
    if (R.closed)
        return;
    R.closed = 1;
    int had = (R.stream != NULL) || (R.nchunks > 0);
    if (!g_acl.dll)
        return;

    if (R.stream) {
        /* 退出前把 stream 排干；上下文已经丢了就别纠结返回值 */
        g_acl.aclrtSynchronizeStream(R.stream);
    }
    matmul_release();
    for (int i = 0; i < R.nchunks; i++)
        if (R.chunks[i].ptr)
            g_acl.aclrtFree(R.chunks[i].ptr);
    R.nchunks = 0;
    free(R.chunks);
    R.chunks = NULL;
    R.capchunks = 0;
    if (R.stream) {
        g_acl.aclrtDestroyStream(R.stream);
        R.stream = NULL;
    }
    if (R.dev_set) {
        g_acl.aclrtResetDevice(R.dev);
        R.dev_set = 0;
    }
    if (!g_cleanup_quiet && had)
        logmsg("已释放 HBM 与设备上下文");
}

/* ------------------------------------------------------------------------- */
/* setup                                                                      */
/* ------------------------------------------------------------------------- */

static void setup(const Args *a)
{
    char soc[256] = "";
    char b1[32], b2[32];
    aclError rc;

    R.dev = a->device;
    rc = g_acl.aclrtSetDevice(a->device);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtSetDevice", rc);
        R.failed = 1;
        return;
    }
    R.dev_set = 1;

    if (g_acl.aclrtGetSocName && g_acl.aclrtGetSocName(soc, sizeof soc) == ACL_SUCCESS && soc[0])
        logmsg("设备 #%d: %s", a->device, soc);
    else
        logmsg("设备 #%d", a->device);

    size_t free_ = 0, total_ = 0;
    int have_mem = probe_meminfo(&free_, &total_);
    if (have_mem)
        logmsg("HBM: 总 %s / 空闲 %s（memType=%d）", human(b1, sizeof b1, total_),
               human(b2, sizeof b2, free_), t_meminfo_type);
    else
        warnmsg("aclrtGetMemInfo 不可用：不做 HBM 统计，--mem-pct 按 64 GiB 估算");

    rc = g_acl.aclrtCreateStream(&R.stream);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtCreateStream", rc);
        R.failed = 1;
        return;
    }

    size_t want;
    if (a->mem_gb > 0.0) {
        double gb = a->mem_gb > 1024.0 ? 1024.0 : a->mem_gb;
        want = (size_t)(gb * (double)GB);
        logmsg("目标 HBM 占用: %s（--mem-gb %.2f）", human(b1, sizeof b1, want), a->mem_gb);
    } else {
        size_t base = have_mem ? free_ : (size_t)65536 * MB;
        want = (size_t)((double)base * a->mem_pct);
        logmsg("目标 HBM 占用: %s（--mem-pct %.0f%% of %s）", human(b1, sizeof b1, want),
               a->mem_pct * 100.0, human(b2, sizeof b2, base));
    }

    if (alloc_hbm(want) != 0) {
        R.failed = 1;
        return;
    }
    logmsg("已占 HBM: %s（%d 块）", human(b1, sizeof b1, R.hbm_bytes), R.nchunks);
    if (R.hbm_bytes < want && !g_stop) {
        char bg[32], bw[32];
        warnmsg("HBM 只拿到 %s / 目标 %s：继续跑，但占不满（--mem-gb/--mem-pct 调小点，"
                "或先确认没有别的进程占着 HBM）",
                human(bg, sizeof bg, R.hbm_bytes), human(bw, sizeof bw, want));
    }

    if (matmul_prepare(a->size) != 0) {
        R.failed = 1;
        return;
    }

    rc = g_acl.aclrtSynchronizeStream(R.stream);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtSynchronizeStream（初始化后同步）", rc);
        R.failed = 1;
        return;
    }
    logmsg("就绪: MatMul %ldx%ldx%ld，异步队列深度 %d", a->size, a->size, a->size, a->queue);
}

/* ------------------------------------------------------------------------- */
/* 统计行                                                                     */
/* ------------------------------------------------------------------------- */

static void print_stats(const Args *a, double elapsed)
{
    char line[768];
    char b1[32], b2[32];
    size_t len = 0;
    line[0] = '\0';

    double n = (double)a->size;
    double flops = 2.0 * n * n * n * (double)R.launches;
    double tflops = elapsed > 0.0 ? flops / elapsed / 1e12 : 0.0;

    appendf(line, sizeof line, &len, "运行 %6.1fs | MatMul 发射 %llu 次 | 累计 %.1f TFLOP | 实测 %.1f TFLOPS",
            elapsed, R.launches, flops / 1e12, tflops);

    size_t free_ = 0, total_ = 0;
    if (probe_meminfo(&free_, &total_))
        appendf(line, sizeof line, &len, " | HBM 已占 %s/%s", human(b1, sizeof b1, R.hbm_bytes),
                human(b2, sizeof b2, total_));

    if (a->npu_smi) {
        NpuSmiEntry e;
        if (npu_smi_lookup(R.dev, &e)) {
            appendf(line, sizeof line, &len, " | npu-smi: AICore %d%% | HBM %ld/%ld MB", e.aicore,
                    e.hbm_used, e.hbm_total);
            if (e.has_power)
                appendf(line, sizeof line, &len, " | %.1fW", e.power);
            if (e.has_temp)
                appendf(line, sizeof line, &len, " | %dC", e.temp);
        }
    }
    logmsg("%s", line);
}

/* ------------------------------------------------------------------------- */
/* 运行循环：每芯片在一个 stream 上连续发 MatMul，队列深度到顶就同步一次        */
/* ------------------------------------------------------------------------- */

static void run_loop(const Args *a)
{
    R.started = now_s();
    double next_stats = R.started + (a->stats > 0.0 ? a->stats : 1e18);

    while (!g_stop) {
        for (int i = 0; i < a->queue && !g_stop; i++) {
            aclError rc = matmul_exec();
            if (rc != ACL_SUCCESS) {
                if (g_stop)
                    break;
                acl_fail("aclopExecuteV2(MatMul)", rc);
                R.failed = 1;
                return;
            }
            R.launches++;
        }
        aclError rc = g_acl.aclrtSynchronizeStream(R.stream);
        if (rc != ACL_SUCCESS) {
            if (g_stop)
                break;
            acl_fail("aclrtSynchronizeStream（等 MatMul 完成）", rc);
            R.failed = 1;
            return;
        }
        double now = now_s();
        if (a->duration > 0.0 && now - R.started >= a->duration) {
            logmsg("到达 --duration %.0fs，主动收工", a->duration);
            break;
        }
        if (a->stats > 0.0 && now >= next_stats) {
            print_stats(a, now - R.started);
            while (next_stats <= now)
                next_stats += a->stats;
        }
    }
    if (g_stop)
        logmsg("收到退出信号，收尾");
}

/* ------------------------------------------------------------------------- */
/* --diag：把整条链路拆成一步一步，哪一步炸就停在哪一步（真机排障用）          */
/* ------------------------------------------------------------------------- */

static int g_init_rc;   /* 主流程 aclInit 的返回码 */
static int g_init_done; /* 主流程是否已成功 aclInit */

static void diag_fail(int step, const char *what, aclError rc)
{
    errmsg("diag %d) %s 失败: rc=%d", step, what, (int)rc);
    const char *s = g_acl.aclGetRecentErrMsg ? g_acl.aclGetRecentErrMsg() : NULL;
    if (s && *s)
        errmsg("aclGetRecentErrMsg: %s", s);
    errmsg("diag 停在第 %d 步：%s。上面的返回码就是 ACL 的原始错误码", step, what);
}

#define DIAG_STEP(no, what, expr)                                              \
    do {                                                                       \
        aclError _rc = (expr);                                                 \
        if (_rc != ACL_SUCCESS) {                                              \
            diag_fail((no), (what), _rc);                                      \
            g_cleanup_quiet = 1;                                               \
            cleanup_chip();                                                    \
            exit(3);                                                           \
        }                                                                      \
        logmsg("diag %d) %s: OK", (no), (what));                               \
    } while (0)

static void diag(const Args *a)
{
    char b1[32], b2[32];
    logmsg("== diag: 逐个 ACL 调用检查（设备 #%d）==", a->device);

    if (g_init_rc != ACL_SUCCESS) {
        diag_fail(1, "aclInit", g_init_rc);
        exit(3);
    }
    logmsg("diag 1) aclInit: OK%s", g_init_done ? "（主流程已执行）" : "");

    uint32_t count = 0;
    DIAG_STEP(2, "aclrtGetDeviceCount", g_acl.aclrtGetDeviceCount(&count));
    logmsg("     可见设备数: %u", count);

    DIAG_STEP(3, "aclrtSetDevice", g_acl.aclrtSetDevice(a->device));
    R.dev = a->device;
    R.dev_set = 1;

    char soc[256] = "";
    if (!g_acl.aclrtGetSocName)
        logmsg("diag 4) aclrtGetSocName: 符号缺失，跳过（不影响刷）");
    else if (g_acl.aclrtGetSocName(soc, sizeof soc) != ACL_SUCCESS || !soc[0])
        logmsg("diag 4) aclrtGetSocName: 调用失败，跳过（不影响刷）");
    else
        logmsg("diag 4) aclrtGetSocName: OK（%s）", soc);

    size_t free_ = 0, total_ = 0;
    if (probe_meminfo(&free_, &total_))
        logmsg("diag 5) aclrtGetMemInfo: OK（空闲 %s / 总 %s，memType=%d）",
               human(b1, sizeof b1, free_), human(b2, sizeof b2, total_), t_meminfo_type);
    else
        logmsg("diag 5) aclrtGetMemInfo: 不可用，跳过（--mem-pct 将按 64 GiB 估算）");

    DIAG_STEP(6, "aclrtCreateStream", g_acl.aclrtCreateStream(&R.stream));

    void *probe = NULL;
    DIAG_STEP(7, "aclrtMalloc 64MiB", g_acl.aclrtMalloc(&probe, 64u * MB, ACL_MEM_MALLOC_HUGE_FIRST));
    if (add_chunk(probe, 64u * MB) != 0) {
        g_acl.aclrtFree(probe);
        exit(3);
    }
    DIAG_STEP(8, "aclrtMemset 64MiB", g_acl.aclrtMemset(probe, 64u * MB, 0, 64u * MB));
    DIAG_STEP(9, "aclrtSynchronizeStream（等 memset 落地）",
              g_acl.aclrtSynchronizeStream(R.stream));

    if (matmul_alloc(1024, &(size_t){0}) != 0) {
        errmsg("diag 10) 分配 MatMul A/B/C 失败");
        g_cleanup_quiet = 1;
        cleanup_chip();
        exit(3);
    }
    logmsg("diag 10) aclrtMalloc MatMul A/B/C（1024x1024 FP16）: OK");

    if (matmul_build(1024, (size_t)2 * (size_t)1024 * (size_t)1024) != 0) {
        errmsg("diag 11) 创建 MatMul 描述符/数据缓冲失败");
        g_cleanup_quiet = 1;
        cleanup_chip();
        exit(3);
    }
    matmul_attr();
    if (g_acl.aclopCreateAttr)
        logmsg("diag 11) aclCreateTensorDesc + aclCreateDataBuffer + aclopAttr: OK");
    else
        logmsg("diag 11) aclCreateTensorDesc + aclCreateDataBuffer: OK（库里没有 aclopCreateAttr，"
               "MatMul 不带属性执行）");

    DIAG_STEP(12, "aclopExecuteV2(MatMul 1024x1024x1024)", matmul_exec());
    DIAG_STEP(13, "aclrtSynchronizeStream（等 MatMul 完成）",
              g_acl.aclrtSynchronizeStream(R.stream));

    g_cleanup_quiet = 1;
    cleanup_chip();
    logmsg("diag 14) 释放 HBM / stream / device: OK");
    logmsg("diag 完成：从 aclInit 到 MatMul 这条链路可以跑");
}

#undef DIAG_STEP

/* ------------------------------------------------------------------------- */
/* --check：不常驻，只验证「设备可见 / 能分配 HBM / MatMul 能跑」              */
/* ------------------------------------------------------------------------- */

static void self_check(const Args *a)
{
    char b1[32], b2[32];
    aclError rc;

    if (g_init_rc != ACL_SUCCESS) {
        acl_fail("aclInit", g_init_rc);
        exit(3);
    }
    logmsg("gpu_burn_ascend %s 自检开始（设备 #%d）", VERSION, a->device);

    uint32_t count = 0;
    rc = g_acl.aclrtGetDeviceCount(&count);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtGetDeviceCount", rc);
        exit(3);
    }
    logmsg("可见设备数: %u", count);
    if (a->device < 0 || (uint32_t)a->device >= count) {
        errmsg("--device %d 超出范围（共 %u 个芯片）", a->device, count);
        exit(3);
    }

    rc = g_acl.aclrtSetDevice(a->device);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtSetDevice", rc);
        exit(3);
    }
    R.dev = a->device;
    R.dev_set = 1;

    char soc[256] = "";
    if (g_acl.aclrtGetSocName && g_acl.aclrtGetSocName(soc, sizeof soc) == ACL_SUCCESS && soc[0])
        logmsg("SoC 名称: %s", soc);
    else
        logmsg("SoC 名称: 未知（aclrtGetSocName 不可用）");

    size_t free_ = 0, total_ = 0;
    if (probe_meminfo(&free_, &total_))
        logmsg("HBM: 总 %s / 空闲 %s", human(b1, sizeof b1, total_), human(b2, sizeof b2, free_));
    else
        logmsg("HBM: 用量不可见（aclrtGetMemInfo 不可用）");

    rc = g_acl.aclrtCreateStream(&R.stream);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtCreateStream", rc);
        exit(3);
    }

    void *p = NULL;
    size_t probe_bytes = 256u * MB;
    rc = g_acl.aclrtMalloc(&p, probe_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtMalloc 256MiB", rc);
        cleanup_chip();
        exit(3);
    }
    if (add_chunk(p, probe_bytes) != 0) {
        g_acl.aclrtFree(p);
        exit(3);
    }
    rc = g_acl.aclrtMemset(p, probe_bytes, 0, probe_bytes);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtMemset 256MiB", rc);
        g_cleanup_quiet = 1;
        cleanup_chip();
        exit(3);
    }
    rc = g_acl.aclrtSynchronizeStream(R.stream);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtSynchronizeStream（256MiB 写入）", rc);
        g_cleanup_quiet = 1;
        cleanup_chip();
        exit(3);
    }
    logmsg("分配 + 写入 256MiB 测试 HBM: OK");

    if (matmul_prepare(2048) != 0) {
        errmsg("自检失败：MatMul 2048 资源准备失败");
        g_cleanup_quiet = 1;
        cleanup_chip();
        exit(3);
    }
    rc = matmul_exec();
    if (rc != ACL_SUCCESS) {
        acl_fail("aclopExecuteV2(MatMul 2048)", rc);
        g_cleanup_quiet = 1;
        cleanup_chip();
        exit(3);
    }
    rc = g_acl.aclrtSynchronizeStream(R.stream);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtSynchronizeStream（MatMul 2048）", rc);
        g_cleanup_quiet = 1;
        cleanup_chip();
        exit(3);
    }
    logmsg("MatMul 2048x2048x2048 试跑: OK");

    g_cleanup_quiet = 1;
    cleanup_chip();
    logmsg("自检通过：这张卡可以刷起来（aclInit → 设备 → HBM 读写 → MatMul 全链路正常）");
}

/* ------------------------------------------------------------------------- */
/* 多芯片：一个进程，每芯片一个线程                                           */
/*                                                                            */
/* 池化平台按进程分配 NPU 会话，fork 出来的子进程申请不到，所以必须单进程多线程。 */
/* 每个线程自己 aclrtSetDevice + 建 stream + 分配自己的 HBM；某芯片失败只标记它   */
/* 自己，其它芯片继续。进程退出（哪怕 kill -9）时驱动会回收所有 device 资源。    */
/* ------------------------------------------------------------------------- */

/* --device / --devices 解析成具体芯片号。返回 0 成功，-1 失败（已打错误） */
static int resolve_devices(Args *a, uint32_t count)
{
    a->ndevs = 1;
    a->devs[0] = a->device >= 0 ? a->device : 0;
    if (!a->devices_spec) {
        if ((uint32_t)a->devs[0] >= count) {
            errmsg("--device %d 超出范围（共 %u 个芯片）", a->devs[0], count);
            return -1;
        }
        return 0;
    }

    if (strcmp(a->devices_spec, "all") == 0) {
        int n = count < MAX_DEVICES ? (int)count : MAX_DEVICES;
        if ((uint32_t)n < count)
            warnmsg("芯片数 %u 超过上限 %d，只刷前 %d 个", count, MAX_DEVICES, n);
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
        if (v < 0 || (uint32_t)v >= count) {
            errmsg("--devices 里的 %ld 超出范围（本机共 %u 个芯片）", v, count);
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
        errmsg("--devices 没给任何芯片号");
        return -1;
    }
    return 0;
}

typedef struct {
    const Args *a;
    int dev;
    int code;
} Worker;

static void *worker_main(void *p)
{
    Worker *w = (Worker *)p;
    Args a = *w->a; /* 每个芯片一份参数副本，改 device 不互相踩 */
    a.device = w->dev;
    if (w->a->ndevs > 1)
        g_dev_tag = w->dev; /* 多芯片时日志带 #N，单芯片保持原样 */

    setup(&a);
    if (!R.failed)
        run_loop(&a);
    cleanup_chip();
    w->code = R.failed ? 3 : 0;
    return NULL;
}

static int run_multi(Args *a)
{
    logmsg("多芯片模式: 目标 %d 个芯片，每芯片一个线程（单进程，只有一个 ACL 会话）", a->ndevs);

    Worker ws[MAX_DEVICES];
    pthread_t tids[MAX_DEVICES];
    int n = 0;
    for (int i = 0; i < a->ndevs && i < MAX_DEVICES; i++) {
        ws[n].a = a;
        ws[n].dev = a->devs[i];
        ws[n].code = 3;
        if (pthread_create(&tids[n], NULL, worker_main, &ws[n]) == 0) {
            n++;
        } else {
            errmsg("#%d 线程起不来，这个芯片跳过", a->devs[i]);
        }
    }

    for (int i = 0; i < n; i++)
        pthread_join(tids[i], NULL);

    int worst = 0, bad = 0;
    if (n < a->ndevs)
        worst = 3;
    for (int i = 0; i < n; i++) {
        if (ws[i].code != 0)
            bad++;
        if (ws[i].code > worst)
            worst = ws[i].code;
    }
    if (worst == 0)
        logmsg("全部 %d 个芯片已收工，HBM 都已释放", n);
    else
        errmsg("%d 个芯片里有 %d 个没能正常刷（错误见上面带 # 卡号的行）", n, bad);
    return worst;
}

/* ------------------------------------------------------------------------- */
/* 收尾 / 信号                                                                */
/* ------------------------------------------------------------------------- */

static void acl_finalize(void)
{
    if (!g_acl.aclFinalize)
        return;
    aclError rc = g_acl.aclFinalize();
    if (rc != ACL_SUCCESS)
        warnmsg("aclFinalize 返回 rc=%d（进程马上退出，可以忽略）", (int)rc);
}

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

/* ------------------------------------------------------------------------- */
/* 帮助                                                                       */
/* ------------------------------------------------------------------------- */

static void usage(FILE *out)
{
    fprintf(out,
            "用法: gpu_burn_ascend [选项]\n"
            "\n"
            "占满昇腾 NPU 的 HBM 并把 AICore 利用率刷到接近 100%%（不依赖 CANN 头文件，\n"
            "运行时 dlopen libascendcl.so；单进程多线程，每芯片一个线程）。\n"
            "\n"
            "选项:\n"
            "  --device N          芯片号（默认 0），等价于 --devices N\n"
            "  --devices LIST      要刷的芯片，逗号分隔或 all（如 0,1,2 / all）\n"
            "  --mem-gb GB         每芯片固定占用多少 GiB HBM（默认按 --mem-pct）\n"
            "  --mem-pct P         占每芯片空闲 HBM 的比例（默认 0.60 = 60%%；写 60 也认）\n"
            "  --duration SEC      运行多少秒后自动退出，0=一直跑（默认 0）\n"
            "  --stats SEC         每多少秒打一行统计，0=不打（默认 5）\n"
            "  --size N            MatMul 的 M=N=K，默认 8192（FP16 时每个矩阵 2*N*N 字节）\n"
            "  --queue N           每芯片的异步队列深度（默认 2）：排 N 个 MatMul 同步一次\n"
            "  --check             单芯片自检（设备数/SoC/256MiB 读写/一次 MatMul），成功退出 0\n"
            "  --diag              逐步检查 ACL 链路，每步打印返回码（排障用）\n"
            "  --npu-smi           统计行里附上 npu-smi 解析出的 AICore%%/HBM/温度/功耗\n"
            "                      （命令可用环境变量 NPU_SMI_BIN 覆盖，默认 \"npu-smi info\"）\n"
            "  --quiet             只打错误\n"
            "  --version           打印版本\n"
            "  -h, --help          打印本页\n"
            "\n"
            "退出码:\n"
            "  0  正常结束\n"
            "  2  环境不满足（找不到 libascendcl.so / 没有可见芯片 / 容器缺设备节点）\n"
            "  3  参数错误或 ACL 调用失败（会打印数字错误码和 aclGetRecentErrMsg）\n"
            "\n"
            "示例:\n"
            "  gpu_burn_ascend --check                  # 先看这张卡能不能刷\n"
            "  gpu_burn_ascend --diag                   # 刷不起来时逐步定位\n"
            "  gpu_burn_ascend                          # 0 号卡，占 60%% 空闲 HBM，一直刷\n"
            "  gpu_burn_ascend --devices all            # 8 张卡一起刷（一个进程）\n"
            "  gpu_burn_ascend --devices 0,2 --duration 1800 --npu-smi\n"
            "  gpu_burn_ascend --mem-gb 40 --size 8192 --queue 2\n");
}

/* ------------------------------------------------------------------------- */
/* main                                                                       */
/* ------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    Args a;
    memset(&a, 0, sizeof a);
    a.device = -1;
    a.mem_gb = -1.0;
    a.mem_pct = 0.60; /* 与 CUDA 版一致：0.6 = 60% */
    a.duration = 0.0;
    a.stats = 5.0;
    a.size = 8192;
    a.queue = 2;
    a.check = 0;
    a.diag = 0;
    a.npu_smi = 0;

    enum {
        OPT_DEVICE = 1000,
        OPT_DEVICES,
        OPT_MEM_GB,
        OPT_MEM_PCT,
        OPT_DURATION,
        OPT_STATS,
        OPT_SIZE,
        OPT_QUEUE,
        OPT_CHECK,
        OPT_DIAG,
        OPT_NPU_SMI,
        OPT_QUIET,
        OPT_VERSION,
        OPT_HELP,
    };
    static const struct option opt[] = {
        {"device", required_argument, 0, OPT_DEVICE},
        {"devices", required_argument, 0, OPT_DEVICES},
        {"mem-gb", required_argument, 0, OPT_MEM_GB},
        {"mem-pct", required_argument, 0, OPT_MEM_PCT},
        {"duration", required_argument, 0, OPT_DURATION},
        {"stats", required_argument, 0, OPT_STATS},
        {"size", required_argument, 0, OPT_SIZE},
        {"queue", required_argument, 0, OPT_QUEUE},
        {"check", no_argument, 0, OPT_CHECK},
        {"diag", no_argument, 0, OPT_DIAG},
        {"npu-smi", no_argument, 0, OPT_NPU_SMI},
        {"quiet", no_argument, 0, OPT_QUIET},
        {"version", no_argument, 0, OPT_VERSION},
        {"help", no_argument, 0, OPT_HELP},
        {0, 0, 0, 0},
    };

    int c;
    while ((c = getopt_long(argc, argv, "h", opt, NULL)) != -1) {
        switch (c) {
        case OPT_DEVICE: a.device = atoi(optarg); break;
        case OPT_DEVICES: a.devices_spec = optarg; break;
        case OPT_MEM_GB: a.mem_gb = atof(optarg); break;
        case OPT_MEM_PCT: a.mem_pct = atof(optarg); break;
        case OPT_DURATION: a.duration = atof(optarg); break;
        case OPT_STATS: a.stats = atof(optarg); break;
        case OPT_SIZE: a.size = atol(optarg); break;
        case OPT_QUEUE: a.queue = atoi(optarg); break;
        case OPT_CHECK: a.check = 1; break;
        case OPT_DIAG: a.diag = 1; break;
        case OPT_NPU_SMI: a.npu_smi = 1; break;
        case OPT_QUIET: g_quiet = 1; break;
        case OPT_VERSION:
            printf("gpu_burn_ascend %s\n", VERSION);
            return 0;
        case OPT_HELP: usage(stdout); return 0;
        default: usage(stderr); return 3;
        }
    }
    if (a.mem_gb > 1024.0) {
        errmsg("--mem-gb %.0f 不合理（单芯片最大按 1024 GiB 处理）", a.mem_gb);
        return 3;
    }
    /* 和 CUDA 版 gpu_burn 保持同一套语义：0~1 是比例（0.6 = 60%）；
       顺手也认 60 这种写法，免得同一个参数名两个程序两种含义 */
    if (a.mem_pct > 1.0 && a.mem_pct <= 100.0) {
        warnmsg("--mem-pct %.4g > 1，按百分比理解（%.0f%%）；下次可以直接写 %.2f",
                a.mem_pct, a.mem_pct, a.mem_pct / 100.0);
        a.mem_pct /= 100.0;
    }
    if (a.mem_pct <= 0.0 || a.mem_pct > 1.0) {
        errmsg("--mem-pct 必须在 (0, 1] 之间（0.6 = 60%%，和 gpu_burn 一样）");
        return 3;
    }
    if (a.duration < 0.0 || a.stats < 0.0) {
        errmsg("--duration / --stats 不能为负数");
        return 3;
    }
    if (a.size < 16) {
        warnmsg("--size %ld 太小，按 16 处理", a.size);
        a.size = 16;
    }
    if (a.size > 32768) {
        warnmsg("--size %ld 太大，按 32768 处理（再大 A/B/C 三块矩阵会占光 HBM）", a.size);
        a.size = 32768;
    }
    if (a.queue < 1)
        a.queue = 1;
    if (a.queue > 64)
        a.queue = 64;

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);

    logmsg("gpu_burn_ascend %s，pid %d", VERSION, (int)getpid());

    if (!load_acl())
        return 2;

    char nodes[256];
    int nnodes = count_davinci_nodes(nodes, sizeof nodes);

    g_init_rc = g_acl.aclInit(NULL);
    if (g_init_rc != ACL_SUCCESS) {
        acl_fail("aclInit", g_init_rc);
        if (nnodes == 0) {
            fputs(NODE_HINT, stderr);
            return 2; /* 环境不满足：容器里没挂设备节点 */
        }
        return 3; /* 设备节点在，但 ACL 初始化失败：按 ACL 错误处理 */
    }
    g_init_done = 1;

    uint32_t devcount = 0;
    aclError rc = g_acl.aclrtGetDeviceCount(&devcount);
    if (rc != ACL_SUCCESS) {
        acl_fail("aclrtGetDeviceCount", rc);
        if (nnodes == 0)
            fputs(NODE_HINT, stderr);
        acl_finalize();
        return nnodes == 0 ? 2 : 3;
    }
    if (devcount == 0) {
        errmsg("没有可见的 NPU 芯片（aclrtGetDeviceCount=0）");
        fputs(NODE_HINT, stderr);
        acl_finalize();
        return 2;
    }
    logmsg("可见芯片数: %u", devcount);
    if (nnodes == 0)
        warnmsg("宿主 /dev 下没有 davinci 节点，但 ACL 能看到 %u 个芯片（可能是容器内映射方式特殊）",
                devcount);

    if (resolve_devices(&a, devcount) != 0) {
        acl_finalize();
        return 3;
    }

    /* --diag / --check 本来就一张一张看，串行跑，不用起线程 */
    if (a.diag || a.check) {
        for (int i = 0; i < a.ndevs; i++) {
            a.device = a.devs[i];
            g_dev_tag = (a.ndevs > 1) ? a.device : -1;
            memset(&R, 0, sizeof R);
            R.dev = -1;
            t_meminfo_type = -2;
            if (a.diag)
                diag(&a);
            else
                self_check(&a);
        }
        acl_finalize();
        return 0;
    }

    int code;
    if (a.ndevs > 1) {
        code = run_multi(&a);
    } else {
        a.device = a.devs[0];
        setup(&a);
        if (!R.failed)
            run_loop(&a);
        cleanup_chip();
        if (!R.failed)
            logmsg("全部 1 个芯片已收工，HBM 都已释放");
        code = R.failed ? 3 : 0;
    }
    acl_finalize();
    return code;
}
