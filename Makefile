FLAGS = -std=c++11 -arch=sm_61 -Xcompiler="-fopenmp" -O3 -lineinfo
DEBUG = -g -G

HEADERS = \
	source/CuBin_gpu.cuh \
	source/helper/config.h \
	source/helper/cuda_helpers.cuh \
	source/helper/rngpu.hpp \
	source/helper/io_and_allocation.hpp \
	source/helper/args_parser.h

MORE_HEADERS = source/helper/matrix_mult.cuh

all: CuBin CuBin_final

CuBin: source/CuBin_main.cu $(HEADERS)
	nvcc source/CuBin_main.cu $(FLAGS) -o CuBin $(MAKROS)

debug: FLAGS += $(DEBUG)
debug: CuBin

CuBin_final: source/CuBin_final.cu $(HEADERS) $(MORE_HEADERS)
	nvcc source/CuBin_final.cu $(FLAGS) -o CuBin_final $(MAKROS)

clean:
	rm -f CuBin CuBin_final

