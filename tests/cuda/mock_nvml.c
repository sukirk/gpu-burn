/*
 * mock_nvml.c -- 假 libnvidia-ml.so.1，配合 mock_libcuda 验证 --stats 的 NVML 分支。
 * 返回值固定但形状正确（结构体布局与真 NVML 一致），用来确认 ctypes/结构体解析没写错。
 *
 * 编译：gcc -O2 -fPIC -shared -Wl,-soname,libnvidia-ml.so.1 -o libnvidia-ml.so.1 mock_nvml.c
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

static int g_inited;
static unsigned g_device;

int nvmlInit_v2(void)
{
    const char *s = getenv("MOCK_DEVS");
    g_inited = s ? atoi(s) : 8; /* 这台"机器"上有几张卡，和假驱动保持一致 */
    if (g_inited < 1)
        g_inited = 1;
    return 0;
}

int nvmlShutdown(void)
{
    g_inited = 0;
    return 0;
}

int nvmlDeviceGetHandleByIndex_v2(unsigned int index, void **dev)
{
    if (!dev)
        return 3; /* NVML_ERROR_INVALID_ARGUMENT */
    if (!g_inited)
        return 1; /* NVML_ERROR_UNINITIALIZED */
    if (index >= (unsigned)g_inited)
        return 3;
    g_device = index;
    *dev = &g_device;
    return 0;
}

int nvmlDeviceGetUtilizationRates(void *dev, nvmlUtilization_t *u)
{
    if (!dev || !u)
        return 3;
    u->gpu = 99;
    u->memory = 41;
    return 0;
}

int nvmlDeviceGetMemoryInfo(void *dev, nvmlMemory_t *m)
{
    if (!dev || !m)
        return 3;
    m->total = 80ull << 30;
    m->free = 12ull << 30;
    m->used = m->total - m->free;
    return 0;
}

int nvmlDeviceGetTemperature(void *dev, int sensor, unsigned int *val)
{
    if (!dev || !val)
        return 3;
    (void)sensor;
    *val = 71;
    return 0;
}

int nvmlDeviceGetPowerUsage(void *dev, unsigned int *mw)
{
    if (!dev || !mw)
        return 3;
    *mw = 398500; /* 398.5 W */
    return 0;
}

int nvmlDeviceGetName(void *dev, char *name, unsigned int len)
{
    if (!dev || !name)
        return 3;
    snprintf(name, len, "Mock A100-SXM4-80GB");
    return 0;
}

int nvmlSystemGetDriverVersion(char *buf, unsigned int len)
{
    snprintf(buf, len, "550.54.15-mock");
    return 0;
}
