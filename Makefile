FLAGS = -std=c++11 -fopenmp -O3
NVCC_FLAGS = -std=c++11 -arch=sm_61 -Xcompiler="-fopenmp" -O3 -lineinfo -ccbin=g++-7
DEBUG = -g -G

HEADERS = \
	src/CuBin_gpu.cuh \
	src/bit_vector_kernels.cuh \
	src/float_kernels.cuh \
	src/CuBin_cpu.h \
	src//config.h \
	src//io_and_allocation.hpp \
	src//updates_and_measures.cuh \
	src/helper/args_parser.h \
	src/helper/rngpu.hpp \
	src/helper/cuda_helpers.cuh


all: CuBin

CuBin: src/CuBin_main.cu $(HEADERS)
	nvcc src/CuBin_main.cu $(NVCC_FLAGS) -o CuBin

# cpu: src/CuBin_main.cu $(HEADERS)
# 	$(CC) src/CuBin_main.cu $(FLAGS) -o CuBin

debug: NVCC_FLAGS += $(DEBUG)
debug: CuBin test_distance

clean:
	rm -f CuBin test_distance

test_distance: source_tests/test_cubin_distance.cu $(HEADERS)
	nvcc source_tests/test_cubin_distance.cu $(NVCC_FLAGS) -o test_distance
