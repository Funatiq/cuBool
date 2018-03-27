FLAGS = -std=c++11 -arch=sm_61 -Xcompiler="-fopenmp" -O3 -lineinfo
DEBUG = -g -G

HEADERS = \
	source/CuBin_final.cuh \
	source/CuBin_gpu.cuh \
	source/CuBin_cpu.cuh \
	source/helper/config.h \
	source/helper/cuda_helpers.cuh \
	source/helper/rngpu.hpp \
	source/helper/matrix_mult.cuh \
	source/helper/io_and_allocation.hpp

all: CuBin

CuBin: source/CuBin_final.cu $(HEADERS)
	nvcc source/CuBin_final.cu $(FLAGS) -o CuBin $(MAKROS)

perf: MAKROS=-DPERF
perf: CuBin

test: MAKROS=-DTEST
test: CuBin

perf-test: MAKROS=-DPERF -DTEST
perf-test: CuBin

clean:
	rm -f CuBin

