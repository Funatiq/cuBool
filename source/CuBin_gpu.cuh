#ifndef CUBIN_GPU_CUH
#define CUBIN_GPU_CUH

#include <vector>
#include <iostream>
#include <limits>
#include <type_traits>

#include "helper/config.h"
#include "helper/rngpu.hpp"
#include "helper/updates_and_measures.cuh"
#include "helper/cuda_helpers.cuh"

using std::cout;
using std::endl;

// init kernels ---------------------------------------------------------------
template<typename bit_vector_t>
__global__
void initFactor(bit_vector_t * Ab,
                const int height,
                const uint8_t factorDim,
                const uint32_t seed, 
                const float threshold)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < height) {
        fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + tid);
        const int randDepth = -log2f(threshold)+1;

        bit_vector_t Ai = ~bit_vector_t(0) >> (32-factorDim);
        for(int d=0; d<randDepth; ++d)
            Ai &= fast_kiss32(state);

        Ab[tid] = Ai;
    }
}

__global__
void initFactor(float * A,
                const int height,
                const uint8_t factorDim,
                const uint32_t seed, 
                const float threshold)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpLane = threadIdx.x % warpSize;

    if(warpId < height) {
        fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + tid);
        const int i = warpId;
        const int j = warpLane;

        A[i * factorDim + j] = j < factorDim ? fast_kiss32(state) < threshold : 0;
    }
}

// distance kernels ---------------------------------------------------------------
template<typename bit_vector_t>
__global__
void computeDistanceRows(const bit_vector_t * __restrict__ Ab,
                         const bit_vector_t * __restrict__ Bb,
                         const bit_vector_t * __restrict__ Cb, 
                         const int height, const int width,
                         const int padded_width,
                         const uint8_t factorDim,
                         const int inverse_density,
                         int *global_error)
{
    const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ int reductionArray[WARPSPERBLOCK];
    
    const int i = warpId;
    int error_thread = 0;
    if (i < height) {
        const bit_vector_t A_i = Ab[i];

        for (int j = warpLane; j < width; j += warpSize) {
            const int product = (A_i & Bb[j]) ? 1 : 0;

            const int vecId = i / 32 * padded_width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            error_thread += error_measure(product, C_ij, inverse_density);
        }
    }
    
    const int error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

    if (threadIdx.x == 0)
        atomicAdd(global_error, error_block);
}

template<typename bit_vector_t>
__global__
void computeDistanceRowsShared(const bit_vector_t * __restrict__ Ab,
                     const bit_vector_t * __restrict__ Bb,
                     const bit_vector_t * __restrict__ Cb, 
                     const int height,
                     const int width,
                     const int padded_width,
                     const uint8_t factorDim,
                     const int inverse_density,
                     int *global_error)
{
    __shared__ bit_vector_t B_block[ 32 * WARPSPERBLOCK ];
    __shared__ bit_vector_t C_block[ 32 * WARPSPERBLOCK ];

    // const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;

    const int blockSize = WARPSPERBLOCK*32;

    __shared__ int reductionArray[WARPSPERBLOCK];
    
    const int i = warpId;
    const bit_vector_t A_i = i < height ? Ab[i] : 0;

    const int vecRow = i / 32;
    const int vecFirst = vecRow * padded_width;
    const int vecLane = i % 32;
    const int col_in_tile = warpLane;
    const int padded_width_blocks = SDIV(width, blockSize) * blockSize;
    int error_thread = 0;
    for (int j = threadIdx.x; j < padded_width_blocks; j += blockSize) {
        B_block[threadIdx.x] = (j < width) ? Bb[j] : 0;
        C_block[threadIdx.x] = (j < width) ? Cb[vecFirst + j] : 0;
        __syncthreads();

        if(i < height) {
            #pragma unroll
            for(int w = 0; w < WARPSPERBLOCK; ++w) {
                const bit_vector_t B_j = B_block[w*warpSize + warpLane];

                // const int vecId = vecFirst + j/(blockSize)*(blockSize) + w*warpSize + warpLane;
                // const int C_ij = (Cb[vecId] >> vecLane) & 1;
                const int C_ij = (C_block[w*warpSize + col_in_tile] >> vecLane) & 1;

                const int product = (B_j & A_i) ? 1 : 0;

                error_thread += error_measure(product, C_ij, inverse_density);
            }
        }
        __syncthreads();
    }

    const int error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

    if (threadIdx.x == 0)
        atomicAdd(global_error, error_block);
}

__global__
void computeDistanceRowsShared(const float * __restrict__ A,
                         const float * __restrict__ B,
                         const uint32_t * __restrict__ Cb, 
                         const int height, const int width,
                         const int padded_width,
                         const uint8_t factorDim,
                         const int inverse_density,
                         int *global_error)
{
    // const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ int reductionArray[WARPSPERBLOCK];

    __shared__ float B_block[CHUNK_SIZE][32];
    __shared__ uint32_t C_block[CHUNK_SIZE];

    const uint32_t dim_mask = FULLMASK >> (32 - factorDim);
    
    const int i = warpId;
    const int k = warpLane;
    const bool A_i_k = A[i*warpSize + k] > 0.5f;

    const int vecRow = i / 32;
    const int vecFirst = vecRow * padded_width;
    const int vecLane = i % 32;
    int error_warp = 0;
    for (int j_chunk = 0; j_chunk < padded_width; j_chunk += CHUNK_SIZE) {
        #pragma unroll
        for(int j_local = warpIdIntern; j_local < CHUNK_SIZE; j_local += WARPSPERBLOCK) {
            const int j = j_chunk + j_local;
            B_block[j_local][k] = j < width ? B[j * warpSize + k] : 0;
        }
        if(threadIdx.x < CHUNK_SIZE) {
            const int vecId = vecFirst + j_chunk;
            C_block[threadIdx.x] = Cb[vecId + threadIdx.x];
        }
        __syncthreads();

        if (i < height) {
            #pragma unroll
            for(int j_local = 0; j_local < CHUNK_SIZE; ++j_local) {
                // int product = __any_sync(dim_mask, A_i_k && (B[j*warpSize + k] > 0.5f)) ? 1 : 0;
                const int product = __any_sync(dim_mask, A_i_k && (B_block[j_local][k] > 0.5f)) ? 1 : 0;

                const int C_ij = (C_block[j_local] >> vecLane) & 1;

                error_warp += error_measure(product, C_ij, inverse_density);
            }
        }
        __syncthreads();
    }
    
    if(warpLane == 0)
        reductionArray[warpIdIntern] = error_warp;
    __syncthreads();

    int error_block;
    if(warpIdIntern == 0) {
        error_block = warpReduceSum(reductionArray[warpLane], WARPSPERBLOCK);
        // Thread with threadIdx.x==0 now has total error of block

       if (threadIdx.x == 0) {
            atomicAdd(global_error, error_block);
       }
    }
}

__global__
void computeDistanceColsShared(const float * __restrict__ A,
                         const float * __restrict__ B,
                         const uint32_t * __restrict__ Cb, 
                         const int height, const int width,
                         const int padded_width,
                         const uint8_t factorDim,
                         const int inverse_density,
                         int *global_error)
{
    // const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpFirst = blockIdx.x * WARPSPERBLOCK;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ int reductionArray[WARPSPERBLOCK];

    __shared__ float A_block[CHUNK_SIZE][32];
    __shared__ uint32_t C_block[WARPSPERBLOCK];

    const uint32_t dim_mask = FULLMASK >> (32 - factorDim);

    const int j = warpId;
    const int k = warpLane;
    const bool B_j_k = B[j*warpSize + k] > 0.5f;

    int error_warp = 0;
    const int padded_heigth = SDIV(height, 32) * 32;
    for (int i_chunk = 0; i_chunk < padded_heigth; i_chunk += CHUNK_SIZE) {
        #pragma unroll
        for(int i_local = warpIdIntern; i_local < CHUNK_SIZE; i_local += WARPSPERBLOCK) {
            const int i = i_chunk + i_local;
            A_block[i_local][k] = i < height ? A[i * warpSize + k] : 0;
        }
        if(threadIdx.x < WARPSPERBLOCK) {
            const int j = warpFirst;
            const int vecRow = i_chunk / 32;
            const int vecId = vecRow * padded_width + j;
            C_block[threadIdx.x] = Cb[vecId + threadIdx.x];
        }
        __syncthreads();

        if (j < width) {
            #pragma unroll
            for(int i_local = 0; i_local < CHUNK_SIZE; ++i_local) {
                // int product = __any_sync(dim_mask, A_k && (B[j*warpSize + k] > 0.5f)) ? 1 : 0;
                const int product = __any_sync(dim_mask, B_j_k && (A_block[i_local][k] > 0.5f)) ? 1 : 0;

                const int i = i_chunk + i_local;
                const int vecLane = i % 32;
                const int C_ij = (C_block[warpIdIntern] >> vecLane) & 1;

                error_warp += error_measure(product, C_ij, inverse_density);
            }
        }
        __syncthreads();
    }

    if(warpLane == 0)
        reductionArray[warpIdIntern] = error_warp;
    __syncthreads();

    int error_block;
    if(warpIdIntern == 0) {
        error_block = warpReduceSum(reductionArray[warpLane], WARPSPERBLOCK);
        // Thread with threadIdx.x==0 now has total error of block

       if (threadIdx.x == 0) {
            atomicAdd(global_error, error_block);
       }
    }
}

// update kernerls ---------------------------------------------------------------
// [A] row Change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareRowWarpShared(bit_vector_t *A,
                                     const bit_vector_t * __restrict__ B,
                                     const bit_vector_t * __restrict__ C,
                                     const int height,
                                     const int width,
                                     const int padded_width,
                                     const uint8_t factorDim,
                                     const int startrow,
                                     int *global_error,
                                     const uint32_t seed, 
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipManyDepth,
                                     const int inverse_density)
{
    __shared__ bit_vector_t B_block[ 32 * WARPSPERBLOCK ];
    __shared__ bit_vector_t C_block[ 32 * WARPSPERBLOCK ];

    // int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpId = blockIdx.x * WARPSPERBLOCK + threadIdx.x / warpSize;
    const int warpLane = threadIdx.x % warpSize;

    // int i = (warpId + startrow) % height;
    const int padded_height_blocks = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int i = (startrow + warpId) % padded_height_blocks;

    fast_kiss_state32_t state;
    
    const bit_vector_t A_i = i < height ? A[i] : 0;
    bit_vector_t A_i_changed = 0;
    // if (warpLane == 0 && i < height) {
    if (i < height) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        A_i_changed = A_i ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
    }
    // A_i_changed = __shfl_sync(FULLMASK, A_i_changed, 0);
    
    const int vecRow = i / 32;
    const int vecFirst = vecRow * padded_width;
    const int vecLane = i % 32;
    const int col_in_tile = warpLane;
    const int padded_width_blocks = SDIV(width, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
    float error_thread = 0;
    for (int j = threadIdx.x; j < padded_width_blocks; j += WARPSPERBLOCK*32) {
        B_block[threadIdx.x] = (j < width) ? B[j] : 0;
        C_block[threadIdx.x] = (j < width) ? C[vecFirst + j] : 0;
        __syncthreads();

        if(i < height) {
            #pragma unroll
            for(int w = 0; w < WARPSPERBLOCK; ++w) {
                const bit_vector_t B_j = B_block[w*warpSize + warpLane];
                const int C_ij = (C_block[w*warpSize + col_in_tile] >> vecLane) & 1;
                // int col = j / blockDim.x * blockDim.x + w*warpSize + warpLane;
                // bit_vector_t B_j = B[col];
                // int C_ij = (C[vecFirst + col] >> vecLane) & 1;

                const int product_new = (B_j & A_i_changed) ? 1 : 0;
                const int product_old = (B_j & A_i        ) ? 1 : 0;

                // const float count_new = __popc(B_j & A_i_changed);
                // const float count_old = __popc(B_j & A_i);

                // if (col < width)
                error_thread += error_measure(product_new, C_ij, 0)
                              - error_measure(product_old, C_ij, 0);
            }
        }
        __syncthreads();
    }
    if(i < height) {
        const float error_warp = warpReduceSum(error_thread);
        // Thread with warpLane==0 now has total error of warp

        // Thread 0 checks if new low has been found and applies if necessary
        if (warpLane == 0) {
        // if (i < height && warpLane == 0) {
            // Metropolis–Hastings algorithm
            if (metro(state, error_warp, temperature, width)) {
                A[i] = A_i_changed;
                atomicAdd(global_error, error_warp);
            }
        }
    }
}

// [B] col change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareColWarpShared(const bit_vector_t * __restrict__ A,
                                     bit_vector_t *B,
                                     const bit_vector_t * __restrict__ C,
                                     const int height,
                                     const int width,
                                     const int padded_width,
                                     const uint8_t factorDim,
                                     const int startcol,
                                     int *global_error,
                                     const uint32_t seed,
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipManyDepth,
                                     const int inverse_density)
{
    __shared__ bit_vector_t A_block[32*WARPSPERBLOCK];
    // __shared__ bit_vector_t B_block[32];
    __shared__ bit_vector_t C_block[32*WARPSPERBLOCK];
    // __shared__ int error_warps[32];

    // int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;


    // int j = (startcol + warpId) % width;
    const int padded_width_blocks = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int j = (startcol + warpId) % padded_width_blocks;

    fast_kiss_state32_t state;

    const bit_vector_t B_j = j < width ? B[j] : 0;
    bit_vector_t B_j_changed = 0;
    // if (warpLane == 0 && j < width) {
    if (j < width) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        B_j_changed = B_j ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
    }
    // B_j_changed = __shfl_sync(FULLMASK, B_j_changed, 0);
    
    float error_thread = 0;
    const int vecLane = warpLane;
    const int col_in_tile = j % 32;
    const int colFirst = j / 32 * 32;
    const int padded_height_blocks = SDIV(height, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
    for (int i = threadIdx.x; i < padded_height_blocks; i += WARPSPERBLOCK*32) {
        A_block[threadIdx.x] = (i < height) ? A[i] : 0;
        const int vecRow = i / 32;
        const int vecFirst = vecRow * padded_width + colFirst;
        C_block[threadIdx.x] = C[vecFirst + warpLane];
        __syncthreads();

        if (j < width) {
            #pragma unroll
            for(int w = 0; w < WARPSPERBLOCK; ++w) {
                const bit_vector_t A_i = A_block[w*warpSize + warpLane];
                const int C_ij = (C_block[w*warpSize + col_in_tile] >> vecLane) & 1;

                // int row = i / blockDim.x * blockDim.x + w*warpSize + warpLane;
                // bit_vector_t A_i = (row < height) ? A[row] : 0;
                // int C_ij = (C[vecFirst + col_in_tile] >> vecLane) & 1;

                const int product_new = (A_i & B_j_changed) ? 1 : 0;
                const int product_old = (A_i & B_j        ) ? 1 : 0;

                // const float count_new = __popc(A_i & B_j_changed);
                // const float count_old = __popc(A_i & B_j);

                // if (row < height)
                error_thread += error_measure(product_new, C_ij, 0)
                              - error_measure(product_old, C_ij, 0);
            }
        }
        __syncthreads();
    }
    if (j < width) {
        const float error_warp = warpReduceSum(error_thread);
        // Thread with warpLane==0 now has total error of warp
        // if (warpLane == 0) {
        //     const bool pred = (j < width) && metro(state, error_warp, temperature);
        //     error_warps[warpIdIntern] = pred ? error_warp : 0;
        //     B_block[warpIdIntern] = pred ? B_j_changed : B_j;
        // }
        // __syncthreads();

        // if (warpIdIntern == 0 && warpLane < WARPSPERBLOCK) {
        //     const uint32_t mask = FULLMASK >> (32 - WARPSPERBLOCK - 1);

        //     B[j + warpLane] = B_block[warpLane];
        //     const int error_block = warpReduceSum(error_warps[warpLane], mask);
        //     if (warpLane == 0)
        //         atomicAdd(global_error, error_block);
        // }

        // Thread 0 checks if new low has been found and applies if necessary
        if (warpLane == 0) {
        // if (j < width && warpLane == 0) {
            // Metropolis–Hastings algorithm
            if (metro(state, error_warp, temperature, height)) {
                B[j] = B_j_changed;
                atomicAdd(global_error, error_warp);
            }
        }
    }
}

// [A] row Change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareRowWarpShared(float *A,
                                     const float * __restrict__ B,
                                     const bit_vector_t * __restrict__ Cb,
                                     const int height,
                                     const int width,
                                     const int padded_width,
                                     const uint8_t factorDim,
                                     const int startrow,
                                     int *global_error,
                                     const uint32_t seed, 
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipManyDepth,
                                     const int inverse_density)
{
    // const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ float B_block[CHUNK_SIZE][32];
    __shared__ bit_vector_t C_block[CHUNK_SIZE];

    const uint32_t dim_mask = FULLMASK >> (32 - factorDim);

    fast_kiss_state32_t state;

    const int padded_height_blocks = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int i = (startrow + warpId) % padded_height_blocks;

    const int k = warpLane;
    const float A_i_k_float = i < height ? A[i * warpSize + k] : 0;
    float A_i_k_float_changed = A_i_k_float;
    if (i < height) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        A_i_k_float_changed += get_float_update(factorDim, state, flipManyChance);
        A_i_k_float_changed = A_i_k_float_changed > 1.0f ? 1.0f : A_i_k_float_changed;
        A_i_k_float_changed = A_i_k_float_changed < 0.0f ? 0.0f : A_i_k_float_changed;
    }
    const bool A_i_k = A_i_k_float > 0.5f;
    const bool A_i_k_changed = A_i_k_float_changed > 0.5f;

    const int vecRow = i / 32;
    const int vecFirst = vecRow * padded_width;
    const int vecLane = i % 32;
    int error_warp = 0;
    for (int j_chunk = 0; j_chunk < padded_width; j_chunk += CHUNK_SIZE) {
        #pragma unroll
        for(int j_local = warpIdIntern; j_local < CHUNK_SIZE; j_local += WARPSPERBLOCK) {
            const int j = j_chunk + j_local;
            B_block[j_local][k] = j < width ? B[j * warpSize + k] : 0;
        }
        if(threadIdx.x < CHUNK_SIZE) {
            const int vecId = vecFirst + j_chunk;
            C_block[threadIdx.x] = Cb[vecId + threadIdx.x];
        }
        __syncthreads();

        if (i < height) {
            #pragma unroll
            for(int j_local = 0; j_local < CHUNK_SIZE; ++j_local) {
                const bool B_j_k = B_block[j_local][k] > 0.5f;
                const int product_old = __any_sync(dim_mask, A_i_k && B_j_k) ? 1 : 0;
                const int product_new = __any_sync(dim_mask, A_i_k_changed && B_j_k) ? 1 : 0;

                const int C_ij = (C_block[j_local] >> vecLane) & 1;

                error_warp += error_measure(product_new, C_ij, inverse_density)
                            - error_measure(product_old, C_ij, inverse_density);
            }
        }
        __syncthreads();
    }
    // each thread now has total error of warp

    if (i < height) {
        // Metropolis–Hastings algorithm
        if (metro(state, error_warp, temperature, width)) {
            A[i * warpSize + k] = A_i_k_float_changed;
            // only one thread updates the global error
            if (warpLane == 0)
                atomicAdd(global_error, error_warp);
        }
    }
}

// [B] col change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareColWarpShared(const float * __restrict__ A,
                                     float *B,
                                     const bit_vector_t * __restrict__ Cb,
                                     const int height,
                                     const int width,
                                     const int padded_width,
                                     const uint8_t factorDim,
                                     const int startcol,
                                     int *global_error,
                                     const uint32_t seed,
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipManyDepth,
                                     const int inverse_density)
{
    // const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpFirst = blockIdx.x * WARPSPERBLOCK;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ float A_block[CHUNK_SIZE][32];
    __shared__ uint32_t C_block[WARPSPERBLOCK];

    const uint32_t dim_mask = FULLMASK >> (32 - factorDim);

    fast_kiss_state32_t state;

    const int padded_width_blocks = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int j = (startcol + warpId) % padded_width_blocks;
    const int jFirst = (startcol + warpFirst) % padded_width_blocks;
    
    const int k = warpLane;
    const float B_j_k_float = j < width ? B[j * warpSize + k] : 0;
    float B_j_k_float_changed = B_j_k_float;
    if (j < width) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        B_j_k_float_changed += get_float_update(factorDim, state, flipManyChance);
        B_j_k_float_changed = B_j_k_float_changed > 1.0f ? 1.0f : B_j_k_float_changed;
        B_j_k_float_changed = B_j_k_float_changed < 0.0f ? 0.0f : B_j_k_float_changed;
    }
    const bool B_j_k = B_j_k_float > 0.5f;
    const bool B_j_k_changed = B_j_k_float_changed > 0.5f;

    int error_warp = 0;
    const int padded_heigth = SDIV(height, 32) * 32;
    for (int i_chunk = 0; i_chunk < padded_heigth; i_chunk += CHUNK_SIZE) {
        #pragma unroll
        for(int i_local = warpIdIntern; i_local < CHUNK_SIZE; i_local += WARPSPERBLOCK) {
            const int i = i_chunk + i_local;
            A_block[i_local][k] = i < height ? A[i * warpSize + k] : 0;
        }
        if(threadIdx.x < WARPSPERBLOCK) {
            const int vecRow = i_chunk / 32; // + i_local / 32 (=0)
            const int vecFirst = vecRow * padded_width + jFirst;
            C_block[threadIdx.x] = Cb[vecFirst + threadIdx.x];
        }
        __syncthreads();

        if (j < width) {
            // const int C_warp = C_block[warpIdIntern];
            #pragma unroll
            for(int i_local = 0; i_local < CHUNK_SIZE; ++i_local) {
                const bool A_i_k = A_block[i_local][k] > 0.5f;
                const int product_old = __any_sync(dim_mask, A_i_k && B_j_k) ? 1 : 0;
                const int product_new = __any_sync(dim_mask, A_i_k && B_j_k_changed) ? 1 : 0;

                const int i = i_chunk + i_local;
                const int vecLane = i % 32;
                const int C_ij = (C_block[warpIdIntern] >> vecLane) & 1;
                // const int C_ij = (C_warp >> vecLane) & 1;
                // const int C_ij = (Cb[i_chunk/32*padded_width+j] >> vecLane) & 1;

                error_warp += error_measure(product_new, C_ij, inverse_density)
                            - error_measure(product_old, C_ij, inverse_density);
            }
        }
        __syncthreads();
    }
    // each thread now has total error of warp

    if (j < width) {
        // Metropolis–Hastings algorithm
        if (metro(state, error_warp, temperature, height)) {
            B[j * warpSize + k] = B_j_k_float_changed;
            // only one thread updates the global error
            if (warpLane == 0)
                atomicAdd(global_error, error_warp);
        }
    }
}



template<typename factor_t = uint32_t>
class CuBin
{
    using factor_matrix_t = std::vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = std::vector<bit_vector_t>;

public:
    CuBin(const factor_matrix_t& A,
          const factor_matrix_t& B,
          const bit_matrix_t& C,
          const uint8_t factorDim,
          const float density)
    {
        cout << "~~~ GPU CuBin ~~~" << endl; 

        int device_id = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        cout << "Using device " << device_id << ": " << prop.name << endl;

        max_parallel_lines_ = prop.multiProcessorCount * WARPSPERBLOCK;

        if(factorDim > 32) {
            std::cerr << "Factor dimension too big! Maximum is 32." << endl;
            factorDim_ = 32;
        }
        else factorDim_ = factorDim;

        density_ = density;
        inverse_density_ = 1 / density;

        if(std::is_same<factor_t, uint32_t>::value) {
            lineSize_ = 1;
            lineSize_padded_ = 1;
        }
        if(std::is_same<factor_t, float>::value) {
            lineSize_ = factorDim_;
            lineSize_padded_ = 32;
        }

        initialize(A, B, C);
    }

    ~CuBin() {
        clear();
    }

    bool initialize(const bit_matrix_t& C, const size_t height, const size_t width) {

        if( SDIV(height,32) * width != C.size()) {
            std::cerr << "CuBin construction: Matrix dimension mismatch." << endl;
            return false;
        }

        if(initialized_) {
            std::cerr << "CuBin already initialized. Please clear CuBin before reinitialization." << endl;
            return false;
        }

        size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        height_ = height;
        // size_t height_padded = SDIV(height_, WARPSPERBLOCK) * WARPSPERBLOCK;
        cudaMalloc(&d_A, lineBytes_padded * height_); CUERR

        width_ = width;
        // size_t width_padded = SDIV(width_, WARPSPERBLOCK) * WARPSPERBLOCK;
        cudaMalloc(&d_B, lineBytes_padded * width_); CUERR
        
        size_t height_C = SDIV(height_, 32);
        width_C_padded_ = SDIV(width_, 32) * 32;
        cudaMalloc(&d_C, sizeof(bit_vector_t) * height_C * width_C_padded_); CUERR

        cudaMemcpy2D(d_C, sizeof(bit_vector_t) * width_C_padded_,
                     C.data(), sizeof(bit_vector_t) * width_,
                     sizeof(bit_vector_t) * width_,
                     height_C,
                     cudaMemcpyHostToDevice); CUERR

        cudaMallocHost(&distance_, sizeof(int)); CUERR
        cudaMalloc(&d_distance_, sizeof(int)); CUERR
        cudaMemset(d_distance_, 0, sizeof(int)); CUERR

        cout << "CuBin initialization complete." << endl;

        cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;
        cout << "Factor dimension:\t" << (int) factorDim_ << endl;

        return initialized_ = true;
    }

    // initialize factors as copy of host vectors
    bool initializeFactors(const factor_matrix_t& A, const factor_matrix_t& B, cudaStream_t stream = 0) {
        if( A.size() != height_ * lineSize_ || B.size() != width_ * lineSize_) {
            std::cerr << "CuBin initialization: Factor dimension mismatch." << endl;
            return false;
        }

        size_t lineBytes = sizeof(factor_t) * lineSize_;
        size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        cudaMemcpy2DAsync(d_A, lineBytes_padded,
                     A.data(), lineBytes,
                     lineBytes,
                     height_,
                     cudaMemcpyHostToDevice,
                     stream);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpy2DAsync(d_B, lineBytes_padded,
                     B.data(), lineBytes,
                     lineBytes,
                     width_,
                     cudaMemcpyHostToDevice,
                     stream);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemsetAsync(d_distance_, 0, sizeof(int), stream);
        // cudaStreamSynchronize(stream); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                        (d_A, d_B, d_C, height_, width_, width_C_padded_,
                         factorDim_, inverse_density_, d_distance_);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpyAsync(distance_, d_distance_, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); CUERR

        cout << "Factor initialization complete." << endl;

        cout << "Start distance: "
             << "\tabs_err: " << *distance_
             << "\trel_err: " << (float) *distance_ / (height_ * width_)
             << endl;

        return true;
    }

    // initialize factors on device according to INITIALIZATIONMODE
    bool initializeFactors(cudaStream_t stream = 0) {
        float threshold;

        switch(INITIALIZATIONMODE) {
            case 1:
                threshold = (sqrt(1 - pow(1 - density_, float(1) / factorDim_)));
                break;
            case 2:
                threshold = (density_ / 100);
                break;
            case 3:
                threshold = (density_);
                break;
        }
        
        uint32_t seed = 0;
        initFactor <<< SDIV(height_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32, 0, stream >>>
                    (d_A, height_, factorDim_, seed, threshold);
        // cudaStreamSynchronize(stream); CUERR

        seed += height_;
        initFactor <<< SDIV(width_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32, 0, stream >>>
                    (d_B, width_, factorDim_, seed, threshold);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemsetAsync(d_distance_, 0, sizeof(int), stream);
        // cudaStreamSynchronize(stream); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                        (d_A, d_B, d_C, height_, width_, width_C_padded_,
                         factorDim_, inverse_density_, d_distance_);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpyAsync(distance_, d_distance_, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); CUERR

        cout << "Factor initialization complete." << endl;

        cout << "Start distance: "
             << "\tabs_err: " << *distance_
             << "\trel_err: " << (float) *distance_ / (height_ * width_)
             << endl;

        return true;
    }


    bool initialize(const factor_matrix_t& A, const factor_matrix_t& B, const bit_matrix_t& C) {
        initialize(C, A.size()/lineSize_, B.size()/lineSize_);
        initializeFactors(A, B);
        // initializeFactors();
        return true;

        // if(std::is_same<factor_t, uint32_t>::value) {
        //     lineSize_ = 1;
        //     lineSize_padded_ = 1;
        // }
        // if(std::is_same<factor_t, float>::value) {
        //     lineSize_ = factorDim_;
        //     lineSize_padded_ = 32;
        // }

        // if( SDIV(A.size()/lineSize_,32) * B.size()/lineSize_ != C.size()) {
        //     std::cerr << "CuBin construction: Matrix dimension mismatch." << endl;
        //     return false;
        // }

        // if(initialized_) {
        //     std::cerr << "CuBin already initialized. Please clear CuBin before reinitialization." << endl;
        //     return false;
        // }

        // size_t lineBytes = sizeof(factor_t) * lineSize_;
        // size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        // height_ = A.size() / lineSize_;
        // // size_t height__padded = SDIV(height_, WARPSPERBLOCK) * WARPSPERBLOCK;
        // cudaMalloc(&d_A, lineBytes_padded * height_); CUERR

        // width_ = B.size() / lineSize_;
        // // size_t width_padded = SDIV(width_, WARPSPERBLOCK) * WARPSPERBLOCK;
        // cudaMalloc(&d_B, lineBytes_padded * width_); CUERR
        
        // size_t height_C = SDIV(height_, 32);
        // width_C_padded_ = SDIV(width_, 32) * 32;
        // cudaMalloc(&d_C, sizeof(bit_vector_t) * height_C * width_C_padded_); CUERR

        // // cudaMemcpy2D(d_A, lineBytes_padded,
        // //              A.data(), lineBytes,
        // //              lineBytes,
        // //              height_,
        // //              cudaMemcpyHostToDevice); CUERR
        // // cudaMemcpy2D(d_B, lineBytes_padded,
        // //              B.data(), lineBytes,
        // //              lineBytes,
        // //              width_,
        // //              cudaMemcpyHostToDevice); CUERR

        // float threshold;

        // switch(INITIALIZATIONMODE) {
        //     case 1:
        //         threshold = (sqrt(1 - pow(1 - density_, float(1) / factorDim_)));
        //         break;
        //     case 2:
        //         threshold = (density_ / 100);
        //         break;
        //     case 3:
        //         threshold = (density_);
        //         break;
        // }
        
        // uint32_t seed = 0;
        // initFactor <<< SDIV(height_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32 >>>
        //             (d_A, height_, factorDim_, seed, threshold);

        // seed += height_;
        // initFactor <<< SDIV(width_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32 >>>
        //             (d_B, width_, factorDim_, seed, threshold);

        // cudaMemcpy2D(d_C, sizeof(bit_vector_t) * width_C_padded_,
        //              C.data(), sizeof(bit_vector_t) * width_,
        //              sizeof(bit_vector_t) * width_,
        //              height_C,
        //              cudaMemcpyHostToDevice); CUERR


        // cudaMallocHost(&distance_, sizeof(int)); CUERR
        // cudaMalloc(&d_distance_, sizeof(int)); CUERR

        // cudaMemset(d_distance_, 0, sizeof(int)); CUERR

        // computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
        //                 (d_A, d_B, d_C, height_, width_, width_C_padded_,
        //                  factorDim_, inverse_density_, d_distance_);

        // cudaDeviceSynchronize(); CUERR

        // cudaMemcpy(distance_, d_distance_, sizeof(int), cudaMemcpyDeviceToHost); CUERR

        // cout << "CuBin initialization complete." << endl;

        // cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;
        // cout << "Factor dimension:\t" << (int) factorDim_ << endl;

        // cout << "Start distance: "
        //      << "\tabs_err: " << *distance_
        //      << "\trel_err: " << (float) *distance_ / (height_ * width_)
        //      << endl;

        // return initialized_ = true;
    }

    bool verifyDistance() {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return false;
        }

        int* distance_proof;
        int* d_distance_proof;

        cudaMallocHost(&distance_proof, sizeof(int)); CUERR
        cudaMalloc(&d_distance_proof, sizeof(int)); CUERR
        cudaMemset(d_distance_proof, 0, sizeof(int)); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                        (d_A, d_B, d_C, height_, width_, width_C_padded_,
                         factorDim_, inverse_density_, d_distance_proof);

        cudaDeviceSynchronize(); CUERR

        cudaMemcpy(distance_proof, d_distance_proof, sizeof(int), cudaMemcpyDeviceToHost); CUERR

        bool equal = *distance_ == *distance_proof;
        if(!equal) {
            cout << "----- !Distances differ! -----\n";
            cout << "Running distance:  " << *distance_ << "\n";
            cout << "Real distance:     " << *distance_proof << endl;
        } else {
            cout << "Distance verified" << endl;
        }

        cudaFreeHost(distance_proof);
        cudaFree(d_distance_proof);
        return equal;
    } 

    void clear() {
        if(initialized_) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            cudaFreeHost(distance_);
            cudaFree(d_distance_);
            initialized_ = false;
        }
    }

    void getFactors(factor_matrix_t& A, factor_matrix_t& B) {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t lineBytes = sizeof(factor_t) * lineSize_;
        size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        A.resize(height_);
        cudaMemcpy2D(A.data(), lineBytes,
                     d_A, lineBytes_padded,
                     lineBytes,
                     height_,
                     cudaMemcpyDeviceToHost); CUERR
        
        B.resize(width_);
        cudaMemcpy2D(B.data(), lineBytes,
                     d_B, lineBytes_padded,
                     lineBytes,
                     width_,
                     cudaMemcpyDeviceToHost); CUERR
    }

    int getDistance(cudaStream_t stream = 0) {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return -1;
        }
        cudaMemcpyAsync(distance_, d_distance_, sizeof(int), cudaMemcpyDeviceToHost, stream); CUERR
        return *distance_;
    }

    struct CuBin_config {
        size_t verbosity = 1;
        size_t linesAtOnce = 0;
        size_t maxIterations = 0;
        int distanceThreshold = 0;
        size_t distanceShowEvery = std::numeric_limits<size_t>::max();
        float tempStart = 0.0f;
        float tempEnd = -1.0f;
        float tempFactor = 0.98f;
        size_t tempStep = std::numeric_limits<size_t>::max();
        uint32_t seed = 0;
        bool loadBalance = false;
        float flipManyChance = 0.1f;
        uint32_t flipManyDepth = 2;
        size_t stuckIterationsBeforeBreak = std::numeric_limits<size_t>::max();
    };

    void run(const CuBin_config& config, cudaStream_t stream = 0) {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t linesAtOnce = SDIV(config.linesAtOnce, WARPSPERBLOCK) * WARPSPERBLOCK;
        if(config.loadBalance) {
            linesAtOnce = linesAtOnce / max_parallel_lines_ * max_parallel_lines_;
            if (!linesAtOnce) linesAtOnce = max_parallel_lines_;
        }

        if(config.verbosity > 0) {
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
            cout << "- - - - Starting " << config.maxIterations
                 << " GPU iterations, changing " << linesAtOnce
                 << " lines each time\n";
            cout << "- - - - Showing error every " << config.distanceShowEvery
                 << " steps\n";
            if(config.tempStart > 0) {
                cout << "- - - - Start temperature " << config.tempStart
                     << " multiplied by " << config.tempFactor
                     << " every " << config.tempStep
                     << " steps\n";

            }
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -";
            cout << endl;
        }

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
        float temperature = config.tempStart;
        size_t iteration = 0;
        size_t stuckIterations = 0;
        auto distancePrev = *distance_;
        while(
                *distance_ > config.distanceThreshold &&
                iteration++ < config.maxIterations
                && temperature > config.tempEnd
                && stuckIterations < config.stuckIterationsBeforeBreak) {

            // Change rows
            int lineToBeChanged = (fast_kiss32(state) % height_) / WARPSPERBLOCK * WARPSPERBLOCK;
            uint32_t gpuSeed = fast_kiss32(state) + iteration;

            vectorMatrixMultCompareRowWarpShared 
                <<< SDIV(min(linesAtOnce, height_), WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                (d_A, d_B, d_C, height_, width_, width_C_padded_, factorDim_,
                 lineToBeChanged, d_distance_, gpuSeed, temperature/10,
                 config.flipManyChance, config.flipManyDepth, inverse_density_);

            // cudaDeviceSynchronize(); CUERR

            // Change cols
            lineToBeChanged = (fast_kiss32(state) % width_) / WARPSPERBLOCK * WARPSPERBLOCK;
            gpuSeed = fast_kiss32(state) + iteration;

            vectorMatrixMultCompareColWarpShared 
                <<< SDIV(min(linesAtOnce, width_), WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                (d_A, d_B, d_C, height_, width_, width_C_padded_, factorDim_,
                 lineToBeChanged, d_distance_, gpuSeed, temperature/10,
                 config.flipManyChance, config.flipManyDepth, inverse_density_);

            cudaStreamSynchronize(stream); CUERR

            getDistance(stream);

            if(config.verbosity > 0 && iteration % config.distanceShowEvery == 0) {
                cout << "Iteration: " << iteration
                     << "\tabs_err: " << *distance_
                     << "\trel_err: " << (float) *distance_ / (height_*width_)
                     << "\ttemp: " << temperature;
                cout << endl;
            }
            if(iteration % config.tempStep == 0) {
                temperature *= config.tempFactor;
            }
            if(*distance_ == distancePrev)
                stuckIterations++;
            else
                stuckIterations = 0;
            distancePrev = *distance_;
        }

        if(config.verbosity > 0) {
            if (!(iteration < config.maxIterations))
                cout << "Reached iteration limit: " << config.maxIterations << endl;
            if (!(*distance_ > config.distanceThreshold))
                cout << "Distance below threshold." << endl;
            if (!(temperature > config.tempEnd))
                cout << "Temperature below threshold." << endl;
            if (!(stuckIterations < config.stuckIterationsBeforeBreak))
                cout << "Stuck for " << stuckIterations << " iterations." << endl;
        }
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        cout << "Final result: "
             << "\tabs_err: " << *distance_
             << "\trel_err: " << (float) *distance_ / (height_ * width_)
             << endl;
    }  

private:
    bool initialized_ = false;
    // factor_matrix_t A;
    // factor_matrix_t B;
    // bit_matrix_st C;
    factor_t *d_A;
    factor_t *d_B;
    bit_vector_t *d_C;
    float density_;
    int inverse_density_;
    int *distance_;
    int *d_distance_;
    // size_t height_padded;
    uint8_t factorDim_ = 20;
    size_t height_ = 0;
    size_t width_ = 0;
    size_t width_C_padded_ = 0;
    size_t lineSize_ = 1;
    size_t lineSize_padded_ = 1;
    int max_parallel_lines_;
};

#endif
