# gpu_burn
#
#   make           编译 ./gpu_burn（只依赖 libc / libdl / libpthread）
#   make check     编译后用 tests/mock 下的假驱动跑端到端测试（不需要 GPU）
#   make mocks     只编译 tests/mock 下的假驱动
#   make clean
#
# 想编出能在老发行版（如 ubuntu:20.04 / glibc 2.31）上跑的二进制，请在老环境里编译：
#   docker run --rm -v "$PWD":/work -w /work ubuntu:20.04 \
#       bash -c "apt-get update -qq && apt-get install -y -qq gcc libc6-dev make && make"

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=gnu11
CFLAGS  += -fstack-protector-strong -D_FORTIFY_SOURCE=2
LDFLAGS += -Wl,-z,relro,-z,now
LDLIBS  := -ldl -lpthread

BIN  := gpu_burn
SRC  := src/gpu_burn.c
MOCK := tests/mock

.PHONY: all check mocks clean

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

mocks: $(MOCK)/libcuda.so.1 $(MOCK)/libnvidia-ml.so.1

$(MOCK)/libcuda.so.1: tests/mock_libcuda.c | $(MOCK)
	$(CC) -O2 -Wall -Wextra -fPIC -shared -Wl,-soname,libcuda.so.1 -o $@ $<

$(MOCK)/libnvidia-ml.so.1: tests/mock_nvml.c | $(MOCK)
	$(CC) -O2 -Wall -Wextra -fPIC -shared -Wl,-soname,libnvidia-ml.so.1 -o $@ $<

$(MOCK):
	mkdir -p $@

check: $(BIN) mocks
	BIN=./$(BIN) bash tests/run_tests.sh

clean:
	rm -f $(BIN) tests/.mock.log
	rm -rf tests/.out $(MOCK)/*.so.1
