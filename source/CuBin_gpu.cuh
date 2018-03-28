#ifndef CUBIN_GPU_CUH
#define CUBIN_GPU_CUH

#include "helper/cuda_helpers.cuh"

template <int rand_depth = 3>
__inline__ __device__
uint32_t get_flip_mask(fast_kiss_state32_t * state) {
    uint32_t bit_flip_mask = fast_kiss32(state);
    #pragma unroll
    for(int i = 1; i < rand_depth; ++i) {
        bit_flip_mask &= fast_kiss32(state);
    }
    bit_flip_mask <<= (32-DIM_PARAM);
    return bit_flip_mask;
}

__inline__ __device__
uint32_t get_flip_mask_11(fast_kiss_state32_t * state) {
    uint32_t bit_flip_mask = 0;
    uint32_t randomNumber = fast_kiss32(state);
    #pragma unroll
    for (int i = 0; i < DIM_PARAM; i++) {
        bit_flip_mask |= (randomNumber >> i) & 11 ? (0 << 32 - 1 - i) : (1 << 32 - 1 - i);
    }
    return bit_flip_mask;
}

__inline__ __device__
bool metro(fast_kiss_state32_t * state, int error, float temperature) {
    if(error < 0)
        return true;
    if(temperature <= 0)
        return false;
    // Metropolis–Hastings algorithm
    float randomNumber = fast_kiss32(state) / (float) UINT32_MAX;
    float metro = fminf(1.0f, expf((float) - error / temperature));
    return randomNumber < metro;
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
                                     const float temperature)
{
    __shared__ bit_vector_t B_block[ 32 * WARPSPERBLOCK ];
    __shared__ bit_vector_t C_block[ 32 * WARPSPERBLOCK ];

    // int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpId = blockIdx.x * WARPSPERBLOCK + threadIdx.x / warpSize;
    const int warpLane = threadIdx.x % warpSize;

    // int rowToBeChanged = (warpId + startrow) % height;
    const int padded_height_blocks = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int rowToBeChanged = (startrow + warpId) % padded_height_blocks;

    fast_kiss_state32_t state;
    
    const bit_vector_t currentRow = A[rowToBeChanged];
    bit_vector_t currentRow_changed = 0;
    if (warpLane == 0 && rowToBeChanged < height) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        currentRow_changed = currentRow ^ get_flip_mask(&state);
    }
    currentRow_changed = __shfl_sync(FULLMASK, currentRow_changed, 0);
    
    int error_thread = 0;
    const int vecRow = rowToBeChanged / 32;
    const int vecFirst = vecRow * padded_width;
    const int vecLane = rowToBeChanged % 32;
    const int col_in_tile = warpLane;
    const int shift = (32 - vecLane - 1);
    const int padded_width_blocks = SDIV(width, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
    for (int j = threadIdx.x; j < padded_width_blocks; j += blockDim.x) {
        B_block[threadIdx.x] = (j < width) ? B[j] : 0;
        C_block[threadIdx.x] = C[vecFirst + j];
        __syncthreads();

        #pragma unroll
        for(int w = 0; w < WARPSPERBLOCK; ++w) {
            const bit_vector_t currentColThread = B_block[w*warpSize + warpLane];
            const int cTruthEntry = (C_block[w*warpSize + col_in_tile] >> shift) & 1;
            // int col = j / blockDim.x * blockDim.x + w*warpSize + warpLane;
            // bit_vector_t currentColThread = B[col];
            // int cTruthEntry = (C[vecFirst + col] >> shift) & 1; 

            const int cEntryNew = (currentColThread & currentRow_changed) ? 1 : 0;
            const int cEntryOld = (currentColThread & currentRow        ) ? 1 : 0;

            // if (col < width)
            error_thread += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
        }
        __syncthreads();
    }
    // if(rowToBeChanged < height) {
        const int error_warp = warpReduceSum(error_thread);
        // Thread with warpLane==0 now has total error of warp

        // Thread 0 checks if new low has been found and applies if necessary
        if (rowToBeChanged < height && warpLane == 0) {
        // if (warpLane == 0) {
            // Metropolis–Hastings algorithm
            if (metro(&state, error_warp, temperature)) {
                A[rowToBeChanged] = currentRow_changed;
                atomicAdd(global_error, error_warp);
            }
        }
    // }
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
                                     const float temperature)
{
    __shared__ bit_vector_t A_block[32*WARPSPERBLOCK];
    // __shared__ bit_vector_t B_block[32];
    __shared__ bit_vector_t C_block[32*WARPSPERBLOCK];
    // __shared__ int error_warps[32];

    // int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const int warpLane = threadIdx.x % warpSize;


    // int colToBeChanged = (startcol + warpId) % width;
    const int padded_width_blocks = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int colToBeChanged = (startcol + warpId) % padded_width_blocks;

    fast_kiss_state32_t state;

    const bit_vector_t currentCol = B[colToBeChanged];
    bit_vector_t currentCol_changed = 0;
    if (warpLane == 0 && colToBeChanged < width) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        currentCol_changed = currentCol ^ get_flip_mask(&state);
    }
    currentCol_changed = __shfl_sync(FULLMASK, currentCol_changed, 0);
    
    int error_thread = 0;
    const int vecLane = warpLane;
    const int col_in_tile = colToBeChanged % 32;
    const int colFirst = colToBeChanged / 32 * 32;
    const int shift = (32 - vecLane - 1);
    const int padded_height_blocks = SDIV(height, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
    for (int i = threadIdx.x; i < padded_height_blocks; i += blockDim.x) {
        A_block[threadIdx.x] = (i < height) ? A[i] : 0;
        int vecRow = i / 32;
        int vecFirst = vecRow * padded_width + colFirst;
        C_block[threadIdx.x] = C[vecFirst + warpLane];
        __syncthreads();

        #pragma unroll
        for(int w = 0; w < WARPSPERBLOCK; ++w) {
            const bit_vector_t currentRowThread = A_block[w*warpSize + warpLane];
            const int cTruthEntry = (C_block[w*warpSize + col_in_tile] >> shift) & 1;

            // int row = i / blockDim.x * blockDim.x + w*warpSize + warpLane;
            // bit_vector_t currentRowThread = (row < height) ? A[row] : 0;
            // int cTruthEntry = (C[vecFirst + col_in_tile] >> shift) & 1; 

            const int cEntryNew = (currentRowThread & currentCol_changed) > 0 ? 1 : 0;
            const int cEntryOld = (currentRowThread & currentCol        ) > 0 ? 1 : 0;

            // if (row < height)
            error_thread += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
        }
        __syncthreads();
    }
    // if (colToBeChanged < width) {
        const int error_warp = warpReduceSum(error_thread);
        // Thread with warpLane==0 now has total error of warp
        // if (warpLane == 0) {
        //     const bool pred = (colToBeChanged < width) && metro(&state, error_warp, temperature);
        //     error_warps[warpIdIntern] = pred ? error_warp : 0;
        //     B_block[warpIdIntern] = pred ? currentCol_changed : currentCol;
        // }
        // __syncthreads();

        // if (warpIdIntern == 0 && warpLane < WARPSPERBLOCK) {
        //     const uint32_t mask = FULLMASK >> (32 - WARPSPERBLOCK - 1);

        //     B[colToBeChanged + warpLane] = B_block[warpLane];
        //     const int error_block = warpReduceSum(error_warps[warpLane], mask);
        //     if (warpLane == 0)
        //         atomicAdd(global_error, error_block);
        // }

        // Thread 0 checks if new low has been found and applies if necessary
        if (colToBeChanged < width && warpLane == 0) {
            // Metropolis–Hastings algorithm
            if (metro(&state, error_warp, temperature)) {
                B[colToBeChanged] = currentCol_changed;
                atomicAdd(global_error, error_warp);
            }
        }
    // }
}


#endif
