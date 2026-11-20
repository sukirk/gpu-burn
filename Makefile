# gpu_burn
#
#   make           编译 ./gpu_burn（只依赖 libc / libdl / libpthread）
#   make check     编译后用 tests/cuda/mock 下的假驱动跑端到端测试（不需要 GPU）
#   make mocks     只编译 tests/cuda/mock 下的假驱动
#   make check-py-cuda    跑 CUDA Python 版（PyTorch + cuBLAS）的行为测试，用假 torch，不需要 GPU
#   make check-py-ascend  跑昇腾 Python 版（torch_npu）的行为测试，用假 torch，不需要 NPU
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
MOCK := tests/cuda/mock

.PHONY: all check mocks clean check-py-cuda check-py-ascend

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

mocks: $(MOCK)/libcuda.so.1 $(MOCK)/libnvidia-ml.so.1

# 假驱动是 .gitignore 里的产物：从别的机器/容器拷过来的副本可能要求更高的 glibc，
# 而它比源码新时 make 不会重编，测试就会莫名其妙 dlopen 失败。所以每次都重编（不到 1 秒）。
FORCE:
.PHONY: FORCE

$(MOCK)/libcuda.so.1: tests/cuda/mock_libcuda.c FORCE | $(MOCK)
	$(CC) -O2 -Wall -Wextra -fPIC -shared -Wl,-soname,libcuda.so.1 -o $@ $<

$(MOCK)/libnvidia-ml.so.1: tests/cuda/mock_nvml.c FORCE | $(MOCK)
	$(CC) -O2 -Wall -Wextra -fPIC -shared -Wl,-soname,libnvidia-ml.so.1 -o $@ $<

$(MOCK):
	mkdir -p $@

check: $(BIN) mocks
	BIN=./$(BIN) bash tests/cuda/run_tests.sh

check-py-cuda:
	bash tests/python/test_cuda_py.sh

check-py-ascend:
	bash tests/python/test_ascend_py.sh

clean:
	rm -f $(BIN) tests/.mock.log
	rm -rf tests/.out $(MOCK)/*.so.1
