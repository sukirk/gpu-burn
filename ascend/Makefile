# 昇腾 NPU 版 gpu_burn 的构建文件。
#
# 只依赖 libc / libdl / libpthread：CANN 的库运行时才 dlopen，编译期不需要任何
# Ascend 头文件，所以在没装 CANN 的开发机上也能 make 出来。
#
#   make           -> ./gpu_burn_ascend
#   make mocks     -> tests/mock/libascendcl.so（假驱动，仅测试用）
#   make check     -> 跑 tests/run_tests.sh
#   make clean     -> 只删本目录的构建产物

CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=gnu11
CFLAGS += -fstack-protector-strong -D_FORTIFY_SOURCE=2
LDFLAGS += -Wl,-z,relro,-z,now
LDLIBS := -ldl -lpthread

BIN := gpu_burn_ascend
MOCK_DIR := tests/mock
MOCK_SO := $(MOCK_DIR)/libascendcl.so

.PHONY: all mocks check clean

all: $(BIN)

$(BIN): gpu_burn_ascend.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

# 假驱动：导出和 libascendcl.so 同名的符号，测试时用 LD_LIBRARY_PATH 顶替真库
mocks: $(MOCK_SO)

# 假驱动是 .gitignore 里的产物：从别的机器/容器拷过来的副本可能要求更高的 glibc，
# 而它比源码新时 make 不会重编，测试就会莫名其妙 dlopen 失败。所以每次都重编（0.3 秒）。
FORCE:
.PHONY: FORCE
$(MOCK_SO): tests/mock_libascendcl.c FORCE
	@mkdir -p $(MOCK_DIR)
	$(CC) -O2 -Wall -Wextra -fPIC -shared -Wl,-soname,libascendcl.so -o $@ $<

check: $(BIN) mocks
	BIN=./$(BIN) bash tests/run_tests.sh

clean:
	rm -f $(BIN) $(MOCK_SO) tests/.mock.log tests/.npu-smi.mock
	rm -rf tests/.out
