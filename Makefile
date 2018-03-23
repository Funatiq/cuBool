FLAGS = -std=c++11 -arch=sm_61 -Xcompiler="-fopenmp" -O3

CuBin:
	nvcc source/CuBin_final.cu $(FLAGS) -o CuBin $(MAKROS)

perf: MAKROS=-DPERF
perf: CuBin

test: MAKROS=-DTEST
test: CuBin

perf-test: MAKROS=-DPERF -DTEST
perf-test: CuBin

clean:
	rm -f CuBin

