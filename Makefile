FLAGS = -std=c++11 -arch=sm_61 -Xcompiler="-fopenmp" -O3 -lineinfo
DEBUG = -g -G

HEADERS = \
	source/CuBin_gpu.cuh \
	source/CuBin_cpu.h \
	source/helper/config.h \
	source/helper/cuda_helpers.cuh \
	source/helper/rngpu.hpp \
	source/helper/io_and_allocation.hpp \
	source/helper/args_parser.h

all: CuBin

CuBin: source/CuBin_main.cu $(HEADERS)
	nvcc source/CuBin_main.cu $(FLAGS) -o CuBin

debug: FLAGS += $(DEBUG)
debug: CuBin test_distance

clean:
	rm -f CuBin test_distance

test_distance: source_tests/test_cubin_distance.cu $(HEADERS)
	nvcc source_tests/test_cubin_distance.cu $(FLAGS) -o test_distance
