FLAGS = -std=c++11 -fopenmp -O3
NVCC_FLAGS = -std=c++11 -arch=sm_61 -Xcompiler="-fopenmp" -O3 -lineinfo -ccbin=g++-7 --default-stream per-thread
DEBUG = -g -G
NVCC = /usr/local/cuda/bin/nvcc

HEADERS = \
	src/cuBool_gpu.cuh \
	src/bit_vector_kernels.cuh \
	src/float_kernels.cuh \
	src/cuBool_cpu.h \
	src/config.h \
	src/io_and_allocation.hpp \
	src/updates_and_measures.cuh \
	src/helper/args_parser.h \
	src/helper/rngpu.hpp \
	src/helper/cuda_helpers.cuh


all: cuBool

cuBool: src/cuBool_main.cu $(HEADERS)
	$(NVCC) src/cuBool_main.cu $(NVCC_FLAGS) -o cuBool

# cpu: src/cuBool_main.cu $(HEADERS)
# 	$(CC) src/cuBool_main.cu $(FLAGS) -o cuBool

debug: NVCC_FLAGS += $(DEBUG)
debug: cuBool

clean:
	rm -f cuBool test_distance

test_distance: source_tests/test_cuBool_distance.cu $(HEADERS)
	$(NVCC) source_tests/test_cuBool_distance.cu $(NVCC_FLAGS) -o test_distance
