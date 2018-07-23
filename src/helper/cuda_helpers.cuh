#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <iostream>
#include <cstdint>

#include "../config.h"

#ifndef __CUDACC__
    #include <chrono>
#endif

#ifndef __CUDACC__
    #define TIMERSTART(label)                                                  \
        std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
        a##label = std::chrono::system_clock::now();
#else
    #define TIMERSTART(label)                                                  \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);
#endif

#ifndef __CUDACC__
    #define TIMERSTOP(label)                                                   \
        b##label = std::chrono::system_clock::now();                           \
        std::chrono::duration<double> delta##label = b##label-a##label;        \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << delta##label.count()  << "s" << std::endl;
#else
    #define TIMERSTOP(label)                                                   \
            cudaEventRecord(stop##label, 0);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
            std::cout << "TIMING: " << time##label << " ms (" << #label << ")" \
                      << std::endl;
#endif


#ifdef __CUDACC__
    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }

    // transfer constants
    // #define H2D (cudaMemcpyHostToDevice)
    // #define D2H (cudaMemcpyDeviceToHost)
    // #define H2H (cudaMemcpyHostToHost)
    // #define D2D (cudaMemcpyDeviceToDevice)
#endif

#ifdef __CUDACC__
    #define HOST_DEVICE_QUALIFIER __host__ __device__
    // #define DEVICE_QUALIFIER __device__
#else
    #define HOST_DEVICE_QUALIFIER 
    // #define DEVICE_QUALIFIER
#endif

// safe division
#ifndef SDIV
    #define SDIV(x,y)(((x)+(y)-1)/(y))
#endif

// floor to next multiple of y
#ifndef FLOOR
    #define FLOOR(x,y)(((x)/(y)*(y))
#endif

#define FULLMASK 0xffffffff

#ifdef __CUDACC__
    template<typename T>
    __inline__ __device__
    T warpReduceSum(T val, const unsigned width = warpSize) {
        for (int offset = width / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULLMASK, val, offset);
        return val;
    }

    template<typename T>
    __inline__ __device__
    T blockReduceSum(T val, T* reductionArray) {
        const int lane = threadIdx.x % warpSize;
        const int wid = threadIdx.x / warpSize;
        val = warpReduceSum(val);
        if (lane == 0) reductionArray[wid] = val;
        __syncthreads();
        if (wid == 0) {
            // val = (threadIdx.x < blockDim.x / warpSize) ? reductionArray[lane] : 0;
            val = (threadIdx.x < WARPSPERBLOCK) ? reductionArray[lane] : 0;
            val = warpReduceSum(val, WARPSPERBLOCK);
        }
        return val;
    }
#endif


#endif
