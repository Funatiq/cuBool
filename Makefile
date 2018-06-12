FLAGS = -std=c++11 -fopenmp -O3
NVCC_FLAGS = -std=c++11 -arch=sm_61 -Xcompiler="-fopenmp" -O3 -lineinfo -ccbin=g++-7
DEBUG = -g -G

HEADERS = \
	source/CuBin_gpu.cuh \
	source/CuBin_cpu.h \
	source/helper/config.h \
	source/helper/cuda_helpers.cuh \
	source/helper/rngpu.hpp \
	source/helper/io_and_allocation.hpp \
	source/helper/args_parser.h \
	source/helper/updates_and_measures.cuh


all: CuBin

CuBin: source/CuBin_main.cu $(HEADERS)
	nvcc source/CuBin_main.cu $(NVCC_FLAGS) -o CuBin

# cpu: source/CuBin_main.cu $(HEADERS)
# 	$(CC) source/CuBin_main.cu $(FLAGS) -o CuBin

debug: NVCC_FLAGS += $(DEBUG)
debug: CuBin test_distance

clean:
	rm -f CuBin test_distance

test_distance: source_tests/test_cubin_distance.cu $(HEADERS)
	nvcc source_tests/test_cubin_distance.cu $(NVCC_FLAGS) -o test_distance
