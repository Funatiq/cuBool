#ifndef CUBIN_GPU_CUH
#define CUBIN_GPU_CUH

#include <vector>
#include <iostream>
#include <limits>
#include <type_traits>

#include "helper/config.h"
#include "helper/rngpu.hpp"
#include "helper/cuda_helpers.cuh"

// uint32_t vector masks --------------------------------------------------------
__inline__ __device__
uint32_t get_flip_mask_many(fast_kiss_state32_t * state, const uint32_t rand_depth) {
    uint32_t bit_flip_mask = FULLMASK;
    #pragma unroll
    for(int i = 0; i < rand_depth; ++i) {
        bit_flip_mask &= fast_kiss32(state);
    }
    bit_flip_mask &= FULLMASK << (32-DIM_PARAM);
    return bit_flip_mask;
}

__inline__ __device__
uint32_t get_flip_mask_11(fast_kiss_state32_t * state) {
    uint32_t bit_flip_mask = 0;
    const uint32_t randomNumber = fast_kiss32(state);
    #pragma unroll
    for (int i = 0; i < DIM_PARAM; i++) {
        bit_flip_mask |= (randomNumber >> i) & 11 ? (0 << 32 - 1 - i) : (1 << 32 - 1 - i);
    }
    return bit_flip_mask;
}

__inline__ __device__
uint32_t get_flip_mask_one(fast_kiss_state32_t * state) {
    const uint32_t lane = fast_kiss32(state) % DIM_PARAM;
    return 1 << (32 - 1 - lane);
}

__inline__ __device__
uint32_t get_flip_mask(fast_kiss_state32_t * state,
                       const float flipManyChance,
                       const uint32_t flipManyDepth) {
    const float random_many = fast_kiss32(state) / (float) UINT32_MAX;

    return random_many < flipManyChance ? get_flip_mask_many(state, flipManyDepth) : get_flip_mask_one(state);
}

// float updates ---------------------------------------------------------------
__inline__ __device__
float get_float_update_many(fast_kiss_state32_t * state) {
    return 0.1f;
}

__inline__ __device__
float get_float_update_one(fast_kiss_state32_t * state) {
    const uint32_t lane = fast_kiss32(state) % DIM_PARAM;
    return threadIdx.x % warpSize == lane ? 0.1f : 0.0f;
}

__inline__ __device__
float get_float_update(fast_kiss_state32_t * state, const float flipManyChance) {
    const float random_many = fast_kiss32(state) / (float) UINT32_MAX;
    float update = random_many < flipManyChance ? get_float_update_many(state) : get_float_update_one(state);

    const float random_factor = fast_kiss32(state) / (float) UINT32_MAX;
    // update = random_factor < 0.5f ? 5*update : update;
    update = random_factor < 1.0f ? 5*update : update;

    const float random_sign = fast_kiss32(state) / (float) UINT32_MAX;
    return random_sign < 0.5f ? update : -1.0*update;
}

// Metropolis–Hastings algorithm
__inline__ __device__
bool metro(fast_kiss_state32_t * state, const int error, const float temperature) {
    if(error < 0)
        return true;
    if(temperature <= 0)
        return false;
    const float randomNumber = fast_kiss32(state) / (float) UINT32_MAX;
    const float metro = fminf(1.0f, expf((float) - error / temperature));
    return randomNumber < metro;
}

//
__global__
void computeFullDistance(const uint32_t * __restrict__ Ab,
                         const uint32_t * __restrict__ Bb,
                         const uint32_t * __restrict__ Cb, 
                         const int height, const int width,
                         const int padded_width,
                         int *distance_test)
{
    const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ int reductionArray[32];
    
    const int i = warpId;
    int error_thread = 0;
    if (i < height) {
        for (int j = warpLane; j < width; j += warpSize) {
            const int lineSum = (Ab[i] & Bb[j]) ? 1 : 0;
            const int intId = i / 32 * padded_width + j;
            const int intLane = i % 32;

            const int truthEntry = (Cb[intId] >> (32 - intLane - 1)) & 1; 
            error_thread += lineSum ^ truthEntry;
        }
    }
    
    const int error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

    if (threadIdx.x == 0)
        atomicAdd(distance_test, error_block);
}

__global__
void computeFullDistance(const float * __restrict__ A,
                         const float * __restrict__ B,
                         const uint32_t * __restrict__ Cb, 
                         const int height, const int width,
                         const int padded_width,
                         int *distance_test)
{
    // const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ int reductionArray[WARPSPERBLOCK];

    __shared__ float B_block[CHUNK_SIZE][32];
    __shared__ uint32_t C_block[CHUNK_SIZE];

    const uint32_t dim_mask = FULLMASK >> (32 - DIM_PARAM);
    
    const int i = warpId;
    const int k = warpLane;
    const bool A_i_k = A[i*warpSize + k] > 0.5f;

    const int vecRow = i / 32;
    const int vecFirst = vecRow * padded_width;
    const int vecLane = i % 32;
    const int shift = (32 - vecLane - 1);    
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
                // int lineSum = __any_sync(dim_mask, A_i_k && (B[j*warpSize + k] > 0.5f)) ? 1 : 0;
                const int lineSum = __any_sync(dim_mask, A_i_k && (B_block[j_local][k] > 0.5f)) ? 1 : 0;

                const int truthEntry = (C_block[j_local] >> shift) & 1; 

                error_warp += lineSum ^ truthEntry;
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
            atomicAdd(distance_test, error_block);
       }
    }
}

__global__
void computeFullDistance2(const float * __restrict__ A,
                         const float * __restrict__ B,
                         const uint32_t * __restrict__ Cb, 
                         const int height, const int width,
                         const int padded_width,
                         int *distance_test)
{
    // const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpFirst = blockIdx.x * WARPSPERBLOCK;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ int reductionArray[WARPSPERBLOCK];

    __shared__ float A_block[CHUNK_SIZE][32];
    __shared__ uint32_t C_block[WARPSPERBLOCK];

    const uint32_t dim_mask = FULLMASK >> (32 - DIM_PARAM);

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
                // int lineSum = __any_sync(dim_mask, A_k && (B[j*warpSize + k] > 0.5f)) ? 1 : 0;
                const int lineSum = __any_sync(dim_mask, B_j_k && (A_block[i_local][k] > 0.5f)) ? 1 : 0;

                const int i = i_chunk + i_local;
                const int vecLane = i % 32;
                const int shift = (32 - vecLane - 1);
                const int truthEntry = (C_block[warpIdIntern] >> shift) & 1; 

                error_warp += lineSum ^ truthEntry;
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
            atomicAdd(distance_test, error_block);
       }
    }
}

// [A] row Change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareRowWarpShared(bit_vector_t *A,
                                     const bit_vector_t * __restrict__ B,
                                     const bit_vector_t * __restrict__ C,
                                     const int height,
                                     const int width,
                                     const int padded_width,
                                     const int startrow,
                                     int *global_error,
                                     const uint32_t seed, 
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipDepth)
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

        A_i_changed = A_i ^ get_flip_mask(&state, flipManyChance, flipDepth);
    }
    // A_i_changed = __shfl_sync(FULLMASK, A_i_changed, 0);
    
    const int vecRow = i / 32;
    const int vecFirst = vecRow * padded_width;
    const int vecLane = i % 32;
    const int col_in_tile = warpLane;
    const int shift = (32 - vecLane - 1);
    const int padded_width_blocks = SDIV(width, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
    int error_thread = 0;
    for (int j = threadIdx.x; j < padded_width_blocks; j += blockDim.x) {
        B_block[threadIdx.x] = (j < width) ? B[j] : 0;
        C_block[threadIdx.x] = C[vecFirst + j];
        __syncthreads();

        if(i < height) {
            #pragma unroll
            for(int w = 0; w < WARPSPERBLOCK; ++w) {
                const bit_vector_t B_j_thread = B_block[w*warpSize + warpLane];
                const int cTruthEntry = (C_block[w*warpSize + col_in_tile] >> shift) & 1;
                // int col = j / blockDim.x * blockDim.x + w*warpSize + warpLane;
                // bit_vector_t B_j_thread = B[col];
                // int cTruthEntry = (C[vecFirst + col] >> shift) & 1; 

                const int cEntryNew = (B_j_thread & A_i_changed) ? 1 : 0;
                const int cEntryOld = (B_j_thread & A_i        ) ? 1 : 0;

                // if (col < width)
                error_thread += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
            }
        }
        __syncthreads();
    }
    if(i < height) {
        const int error_warp = warpReduceSum(error_thread);
        // Thread with warpLane==0 now has total error of warp

        // Thread 0 checks if new low has been found and applies if necessary
        if (warpLane == 0) {
        // if (i < height && warpLane == 0) {
            // Metropolis–Hastings algorithm
            if (metro(&state, error_warp, temperature)) {
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
                                     const int startcol,
                                     int *global_error,
                                     const uint32_t seed,
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipManyDepth)
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

        B_j_changed = B_j ^ get_flip_mask(&state, flipManyChance, flipManyDepth);
    }
    // B_j_changed = __shfl_sync(FULLMASK, B_j_changed, 0);
    
    int error_thread = 0;
    const int vecLane = warpLane;
    const int col_in_tile = j % 32;
    const int colFirst = j / 32 * 32;
    const int shift = (32 - vecLane - 1);
    const int padded_height_blocks = SDIV(height, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
    for (int i = threadIdx.x; i < padded_height_blocks; i += blockDim.x) {
        A_block[threadIdx.x] = (i < height) ? A[i] : 0;
        const int vecRow = i / 32;
        const int vecFirst = vecRow * padded_width + colFirst;
        C_block[threadIdx.x] = C[vecFirst + warpLane];
        __syncthreads();

        if (j < width) {
            #pragma unroll
            for(int w = 0; w < WARPSPERBLOCK; ++w) {
                const bit_vector_t A_i_thread = A_block[w*warpSize + warpLane];
                const int cTruthEntry = (C_block[w*warpSize + col_in_tile] >> shift) & 1;

                // int row = i / blockDim.x * blockDim.x + w*warpSize + warpLane;
                // bit_vector_t A_i_thread = (row < height) ? A[row] : 0;
                // int cTruthEntry = (C[vecFirst + col_in_tile] >> shift) & 1; 

                const int cEntryNew = (A_i_thread & B_j_changed) > 0 ? 1 : 0;
                const int cEntryOld = (A_i_thread & B_j        ) > 0 ? 1 : 0;

                // if (row < height)
                error_thread += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
            }
        }
        __syncthreads();
    }
    if (j < width) {
        const int error_warp = warpReduceSum(error_thread);
        // Thread with warpLane==0 now has total error of warp
        // if (warpLane == 0) {
        //     const bool pred = (j < width) && metro(&state, error_warp, temperature);
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
            if (metro(&state, error_warp, temperature)) {
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
                                     const int startrow,
                                     int *global_error,
                                     const uint32_t seed, 
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipDepth)
{
    // const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ float B_block[CHUNK_SIZE][32];
    __shared__ uint32_t C_block[CHUNK_SIZE];

    const uint32_t dim_mask = FULLMASK >> (32 - DIM_PARAM);

    fast_kiss_state32_t state;

    const int padded_height_blocks = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int i = (startrow + warpId) % padded_height_blocks;

    const int k = warpLane;
    const float A_i_k_float = i < height ? A[i * warpSize + k] : 0;
    float A_i_k_float_changed = A_i_k_float;
    if (i < height) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        A_i_k_float_changed += get_float_update(&state, flipManyChance);
        A_i_k_float_changed = A_i_k_float_changed > 1.0f ? 1.0f : A_i_k_float_changed;
        A_i_k_float_changed = A_i_k_float_changed < 0.0f ? 0.0f : A_i_k_float_changed;
    }
    const bool A_i_k = A_i_k_float > 0.5f;
    const bool A_i_k_changed = A_i_k_float_changed > 0.5f;

    const int vecRow = i / 32;
    const int vecFirst = vecRow * padded_width;
    const int vecLane = i % 32;
    const int shift = (32 - vecLane - 1);
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
                const int cEntryOld = __any_sync(dim_mask, A_i_k && B_j_k) ? 1 : 0;
                const int cEntryNew = __any_sync(dim_mask, A_i_k_changed && B_j_k) ? 1 : 0;

                const int cTruthEntry = (C_block[j_local] >> shift) & 1; 

                error_warp += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
            }
        }
        __syncthreads();
    }
    // each thread now has total error of warp

    if (i < height) {
        // Metropolis–Hastings algorithm
        if (metro(&state, error_warp, temperature)) {
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
                                     const int startcol,
                                     int *global_error,
                                     const uint32_t seed,
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipManyDepth)
{
    // const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpFirst = blockIdx.x * WARPSPERBLOCK;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ float A_block[CHUNK_SIZE][32];
    __shared__ uint32_t C_block[WARPSPERBLOCK];

    const uint32_t dim_mask = FULLMASK >> (32 - DIM_PARAM);

    fast_kiss_state32_t state;

    const int padded_width_blocks = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int j = (startcol + warpId) % padded_width_blocks;
    
    const int k = warpLane;
    const float B_j_k_float = j < height ? A[j * warpSize + k] : 0;
    float B_j_k_float_changed = B_j_k_float;
    if (j < height) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        B_j_k_float_changed += get_float_update(&state, flipManyChance);
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
            A_block[i_local][k] = i < width ? A[i * warpSize + k] : 0;
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
                const bool A_i_k = A_block[i_local][k] > 0.5f;
                const int cEntryOld = __any_sync(dim_mask, A_i_k && B_j_k) ? 1 : 0;
                const int cEntryNew = __any_sync(dim_mask, A_i_k && B_j_k_changed) ? 1 : 0;

                const int i = i_chunk + i_local;
                const int vecLane = i % 32;
                const int shift = (32 - vecLane - 1);
                const int cTruthEntry = (C_block[warpIdIntern] >> shift) & 1; 

                error_warp += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
            }
        }
        __syncthreads();
    }
    // each thread now has total error of warp

    if (j < height) {
        // Metropolis–Hastings algorithm
        if (metro(&state, error_warp, temperature)) {
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
    CuBin(const factor_matrix_t& A, const factor_matrix_t& B, const bit_matrix_t& C) {
        int device_id = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        max_parallel_lines = prop.multiProcessorCount * WARPSPERBLOCK;

        initialize(A, B, C);
    }

    ~CuBin() {
        clear();
    }

    bool initialize(const factor_matrix_t& A, const factor_matrix_t& B, const bit_matrix_t& C) {
        if(std::is_same<factor_t, uint32_t>::value) {
            lineSize = 1;
            lineSizePadded = 1;
        }
        if(std::is_same<factor_t, float>::value) {
            lineSize = DIM_PARAM;
            lineSizePadded = 32;
        }

        if( SDIV(A.size()/lineSize,32) * B.size()/lineSize != C.size()) {
            std::cerr << "CuBin construction: Matrix dimension mismatch." << std::endl;
            return false;
        }

        if(initialized) {
            std::cout << "CuBin already initialized. Please clear CuBin before reinitialization." << std::endl;
            return false;
        }

        size_t lineBytes = sizeof(factor_t) * lineSize;
        size_t lineBytes_padded = sizeof(factor_t) * lineSizePadded;

        height = A.size() / lineSize;
        // size_t height_padded = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
        cudaMalloc(&d_A, lineBytes_padded * height); CUERR

        width = B.size() / lineSize;
        // size_t width_padded = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
        cudaMalloc(&d_B, lineBytes_padded * width); CUERR
        
        size_t height_C = SDIV(height, 32);
        width_C_padded = SDIV(width, 32) * 32;
        cudaMalloc(&d_C, sizeof(bit_vector_t) * height_C * width_C_padded); CUERR

        cudaMemcpy2D(d_A, lineBytes_padded,
                     A.data(), lineBytes,
                     lineBytes,
                     height,
                     cudaMemcpyHostToDevice); CUERR
        cudaMemcpy2D(d_B, lineBytes_padded,
                     B.data(), lineBytes,
                     lineBytes,
                     width,
                     cudaMemcpyHostToDevice); CUERR
        cudaMemcpy2D(d_C, sizeof(bit_vector_t) * width_C_padded,
                     C.data(), sizeof(bit_vector_t) * width,
                     sizeof(bit_vector_t) * width,
                     height_C,
                     cudaMemcpyHostToDevice); CUERR


        cudaMallocHost(&distance, sizeof(int)); CUERR
        cudaMalloc(&d_distance, sizeof(int)); CUERR
        cudaMemset(d_distance, 0, sizeof(int)); CUERR

        computeFullDistance <<< SDIV(width, WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                        (d_A, d_B, d_C, height, width, width_C_padded, d_distance);

        cudaDeviceSynchronize(); CUERR

        cudaMemcpy(distance, d_distance, sizeof(int), cudaMemcpyDeviceToHost); CUERR

        return initialized = true;
    }

    bool verifyDistance() {
        if(!initialized) {
            std::cerr << "CuBin not initialized." << endl;
            return false;
        }

        int* distance_proof;
        int* d_distance_proof;

        cudaMallocHost(&distance_proof, sizeof(int)); CUERR
        cudaMalloc(&d_distance_proof, sizeof(int)); CUERR
        cudaMemset(d_distance_proof, 0, sizeof(int)); CUERR

        computeFullDistance <<< SDIV(width, WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                        (d_A, d_B, d_C, height, width, width_C_padded, d_distance_proof);

        cudaDeviceSynchronize(); CUERR

        cudaMemcpy(distance_proof, d_distance_proof, sizeof(int), cudaMemcpyDeviceToHost); CUERR

        bool equal = *distance == *distance_proof;
        if(!equal) {
            std::cout << "----- !Distances differ! -----\n";
            std::cout << "Running distance:  " << *distance << "\n";
            std::cout << "Real distance:     " << *distance_proof << std::endl;
        }

        cudaFreeHost(distance_proof);
        cudaFree(d_distance_proof);
        return equal;
    } 

    void clear() {
        if(initialized) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            cudaFreeHost(distance);
            cudaFree(d_distance);
            initialized = false;
        }
    }

    void getFactors(factor_matrix_t& A, factor_matrix_t& B) {
        if(!initialized) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        if(std::is_same<factor_t, uint32_t>::value) {
            lineSize = 1;
            lineSizePadded = 1;
        }
        if(std::is_same<factor_t, float>::value) {
            lineSize = DIM_PARAM;
            lineSizePadded = 32;
        }

        size_t lineBytes = sizeof(factor_t) * lineSize;
        size_t lineBytes_padded = sizeof(factor_t) * lineSizePadded;

        A.resize(height);
        cudaMemcpy2D(A.data(), lineBytes,
                     d_A, lineBytes_padded,
                     lineBytes,
                     height,
                     cudaMemcpyHostToDevice); CUERR
        
        B.resize(width);
        cudaMemcpy2D(B.data(), lineBytes,
                     d_B, lineBytes_padded,
                     lineBytes,
                     width,
                     cudaMemcpyHostToDevice); CUERR
    }

    int getDistance() {
        if(!initialized) {
            std::cerr << "CuBin not initialized." << endl;
            return -1;
        }
        cudaMemcpy(distance, d_distance, sizeof(int), cudaMemcpyDeviceToHost); CUERR
        return *distance;
    }

    struct CuBin_config {
        size_t verbosity = 2;
        size_t linesAtOnce = 0;
        size_t maxIterations = 0;
        int distanceThreshold = 0;
        size_t distanceShowEvery = std::numeric_limits<size_t>::max();
        float tempStart = 0.0f;
        float tempEnd = -1.0f;
        float tempFactor = 0.98f;
        size_t tempReduceEvery = std::numeric_limits<size_t>::max();
        uint32_t seed = 0;
        bool loadBalance = false;
        float flipManyChance = 0.1f;
        uint32_t flipManyDepth = 2;
    };

    void run(const CuBin_config& config) {
        if(!initialized) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t linesAtOnce = config.linesAtOnce;
        if(config.loadBalance) {
            linesAtOnce = linesAtOnce / max_parallel_lines * max_parallel_lines;
            if (!linesAtOnce) linesAtOnce = max_parallel_lines;
        }

        if(config.verbosity > 0) {
            std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
            std::cout << "- - - - Starting " << config.maxIterations
                      << " GPU iterations, changing " << linesAtOnce
                      << " lines each time\n";
            std::cout << "- - - - Showing error every " << config.distanceShowEvery
                      << " steps\n";
            if(config.tempStart > 0) {
                std::cout << "- - - - Start temperature " << config.tempStart
                          << " multiplied by " << config.tempFactor
                          << " every " << config.tempReduceEvery
                          << " steps\n";

            }
            std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -";
            std::cout << std::endl;
        }

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
        float temperature = config.tempStart;
        size_t iteration = 0;
        while( *distance > config.distanceThreshold
                && iteration++ < config.maxIterations
                && temperature > config.tempEnd) {

            // Change rows
            int lineToBeChanged = (fast_kiss32(&state) % height) / WARPSPERBLOCK * WARPSPERBLOCK;
            uint32_t gpuSeed = fast_kiss32(&state) + iteration;

            vectorMatrixMultCompareRowWarpShared 
                <<< SDIV(min(linesAtOnce, height), WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                (d_A, d_B, d_C, height, width, width_C_padded,
                 lineToBeChanged, d_distance, gpuSeed, temperature, config.flipManyChance, config.flipManyDepth);

            cudaDeviceSynchronize(); CUERR

            // Change cols
            lineToBeChanged = (fast_kiss32(&state) % width) / WARPSPERBLOCK * WARPSPERBLOCK;
            gpuSeed = fast_kiss32(&state) + iteration;

            vectorMatrixMultCompareColWarpShared 
                <<< SDIV(min(linesAtOnce, width), WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                (d_A, d_B, d_C, height, width, width_C_padded,
                 lineToBeChanged, d_distance, gpuSeed, temperature, config.flipManyChance, config.flipManyDepth);

            cudaDeviceSynchronize(); CUERR

            getDistance();

            if(config.verbosity > 0 && iteration % config.distanceShowEvery == 0) {
                std::cout << "Iteration: " << iteration 
                          << " \tCurrent distance: " << (float) *distance / (height*width)
                          << " = " << *distance << " elements" << std::endl;
            }
            if(iteration % config.tempReduceEvery == 0) {
                temperature *= config.tempFactor;
                if(config.verbosity > 1)
                    std::cout << "Iteration: " << iteration << " \tTemperature: " << temperature << std::endl;
            }
        }

        if(config.verbosity > 0) {
            if (!(iteration < config.maxIterations))
                std::cout << "Reached iteration limit: " << config.maxIterations << std::endl;
            if (!(*distance > config.distanceThreshold))
                std::cout << "Distance below threshold." << std::endl;
            if (!(temperature > config.tempEnd))
                std::cout << "Temperature below threshold." << std::endl;
        }
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        std::cout << "Final distance: " << (float) *distance / (height * width)
                  << " = " << *distance << " elements" << std::endl;
    }  

private:
    bool initialized = false;
    factor_matrix_t A;
    factor_matrix_t B;
    bit_matrix_t C;
    factor_t *d_A;
    factor_t *d_B;
    bit_vector_t *d_C;
    int *distance;
    int *d_distance;
    // size_t height_padded;
    size_t height = 0;
    size_t width = 0;
    size_t width_C_padded = 0;
    size_t lineSize = 1;
    size_t lineSizePadded = 1;
    int max_parallel_lines;
};

#endif
