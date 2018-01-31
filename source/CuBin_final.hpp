#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/types.h>
#include <math.h>
#include <time.h>
#include "helper/rngpu.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <curand_kernel.h>
#include <float.h>
#include <cuda_profiler_api.h>
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <fstream>

#define BLOCKSPERGRID 1024
#define TILE_WIDTH 32

#ifndef THREADSPERBLOCK
#define THREADSPERBLOCK 1024
#endif
#ifndef DIM_PARAM
#define DIM_PARAM 20
#endif
#ifndef CPUITERATIONS
#define CPUITERATIONS 10
#endif
#ifndef INITIALIZATIONMODE
#define INITIALIZATIONMODE 2
#endif

// Makros taken from Christian Hundt
// https://github.com/gravitino/cudahelpers
#define CUERR { \
        cudaError_t cudaerr; \
        if ((cudaerr = cudaGetLastError()) != cudaSuccess){ \
            printf("CUDA ERROR: \"%s\" at LINE %d.\n", cudaGetErrorString(cudaerr), __LINE__); \
        } \
}

#define TIMERSTART(label)                                                    \
        cudaEvent_t start##label, stop##label;                               \
        float time##label;                                                   \
        cudaEventCreate(&start##label);                                      \
        cudaEventCreate(&stop##label);                                       \
        cudaEventRecord(start##label, 0);

#define TIMERSTOP(label)                                                     \
        cudaEventRecord(stop##label, 0);                                     \
        cudaEventSynchronize(stop##label);                                   \
        cudaEventElapsedTime(&time##label, start##label, stop##label);       \
        printf("#%f ms (%s)\n", time##label, #label);

void readInputFileData(uint32_t**, uint32_t**, int*, int*, double*, std::string);

void readInputFileTSV(uint32_t**, uint32_t**, int*, int*, double*, std::string);

//void readInputFileMovieLens(uint32_t**, uint32_t**, int*, int *, double*, string);

bool endsWith(const std::string&, const std::string&);

// void initializeTextureMemory(uint32_t**, int, int);

void initializeFactors(uint32_t**, uint32_t**, uint32_t**, uint32_t**, int, int, float, fast_kiss_state32_t*);

void computeStartError(uint32_t*, uint32_t*, uint32_t*, int, int, int**, int*);

void checkDistance(uint32_t*, uint32_t*, uint32_t*, int, int);

void aftertestGPU(uint32_t*, uint32_t*, uint32_t*, int, int);

void writeToFiles(uint32_t*, uint32_t*, int, int);

void CPUcomputation(uint32_t*, uint32_t*, uint32_t*, int, int, int, uint32_t, int, float, int);

void CPUvectorMatrixMultCompareRow(uint32_t*, uint32_t*, uint32_t*, int, int, int, int*, fast_kiss_state32_t*, int);

void CPUvectorMatrixMultCompareCol(uint32_t*, uint32_t*, uint32_t*, int, int, int, int*, fast_kiss_state32_t*, int);

void aftertestCPU(uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t*, int, int);

__global__ void vectorMatrixMultCompareRow(uint32_t*, uint32_t*, uint32_t*, int, int, int, int*, uint32_t, float);

__global__ void vectorMatrixMultCompareCol(uint32_t*, uint32_t*, uint32_t*, int, int, int, int*, uint32_t, float);

__global__ void computeFullError(uint32_t*, uint32_t*, uint32_t*, int, int, int*);

__global__ void matrixMultiply(uint32_t*, uint32_t*, uint32_t*, int, int);

__global__ void matrixMultiplyInt(int*, int*, uint32_t*, int, int, int);

__inline__ __device__ int warpReduceSum(int);