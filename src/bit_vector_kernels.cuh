#ifndef BIT_VECTOR_KERNELS_CUH
#define BIT_VECTOR_KERNELS_CUH

// using index_t = uint32_t;

// __device__ __constant__ index_t height_c;
// __device__ __constant__ index_t width_c;
// __device__ __constant__ index_t width_padded_c;
// __device__ __constant__ int weight_c;
// __device__ __constant__ float flipManyChance_c;
// __device__ __constant__ uint32_t flipManyDepth_c;
// __device__ __constant__ uint8_t factorDim_c;

// init kernel ---------------------------------------------------------------
template<typename bit_vector_t, typename index_t>
__global__
void initFactor(bit_vector_t * Ab,
                const index_t height,
                const uint8_t factorDim,
                const uint32_t seed, 
                const float threshold)
{
    const index_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < height) {
        fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + tid);

        const int randDepth = -log2f(threshold)+1;
        bit_vector_t Ai = 0;
        // if threshold very small simply initilize as 0s (also catch threshold=0)
        if(randDepth < 16) {
            Ai = ~bit_vector_t(0) >> (32-factorDim);
            for(int d=0; d<randDepth; ++d)
                Ai &= fast_kiss32(state);
        }
        Ab[tid] = Ai;
    }
}


// distance kernels ---------------------------------------------------------------
template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
__global__
void computeDistanceRows(const bit_factor_t * __restrict__ Ab,
                         const bit_factor_t * __restrict__ Bb,
                         const bit_matrix_t * __restrict__ Cb, 
                         const index_t height, const index_t width,
                         const index_t padded_width,
                         const uint8_t factorDim,
                         const int weight,
                         error_t *global_error)
{
    const index_t warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const index_t warpLane = threadIdx.x % warpSize;

    __shared__ error_t reductionArray[WARPSPERBLOCK];
    
    const index_t i = warpId;
    error_t error_thread = 0;
    if (i < height) {
        const bit_factor_t A_i = Ab[i];

        for (index_t j = warpLane; j < width; j += warpSize) {
            const int product = (A_i & Bb[j]) ? 1 : 0;

            const index_t vecId = i / 32 * padded_width + j;
            const index_t vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            error_thread += error_measure(product, C_ij, weight);
        }
    }
    
    const error_t error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

    if (threadIdx.x == 0)
        atomicAdd(global_error, error_block);
}

template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
__global__
void computeDistanceRowsShared(const bit_factor_t * __restrict__ Ab,
                     const bit_factor_t * __restrict__ Bb,
                     const bit_matrix_t * __restrict__ Cb, 
                     const index_t height,
                     const index_t width,
                     const index_t padded_width,
                     const uint8_t factorDim,
                     const int weight,
                     error_t *global_error)
{
    __shared__ bit_factor_t B_block[ 32 * WARPSPERBLOCK ];
    __shared__ bit_matrix_t C_block[ 32 * WARPSPERBLOCK ];

    // const index_t warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const index_t warpIdIntern = threadIdx.x / warpSize;
    const index_t warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const index_t warpLane = threadIdx.x % warpSize;

    const index_t blockSize = WARPSPERBLOCK*32;

    __shared__ error_t reductionArray[WARPSPERBLOCK];
    
    const index_t i = warpId;
    const bit_factor_t A_i = i < height ? Ab[i] : 0;

    const index_t vecRow = i / 32;
    const index_t vecFirst = vecRow * padded_width;
    const index_t vecLane = i % 32;
    const index_t col_in_tile = warpLane;
    const index_t padded_width_blocks = SDIV(width, blockSize) * blockSize;
    error_t error_thread = 0;
    for (index_t j = threadIdx.x; j < padded_width_blocks; j += blockSize) {
        B_block[threadIdx.x] = (j < width) ? Bb[j] : 0;
        C_block[threadIdx.x] = (j < width) ? Cb[vecFirst + j] : 0;
        __syncthreads();

        if(i < height) {
            #pragma unroll
            for(index_t w = 0; w < WARPSPERBLOCK; ++w) {
                const bit_factor_t B_j = B_block[w*warpSize + warpLane];

                // const index_t vecId = vecFirst + j/(blockSize)*(blockSize) + w*warpSize + warpLane;
                // const int C_ij = (Cb[vecId] >> vecLane) & 1;
                const int C_ij = (C_block[w*warpSize + col_in_tile] >> vecLane) & 1;

                const int product = (B_j & A_i) ? 1 : 0;

                error_thread += error_measure(product, C_ij, weight);
            }
        }
        __syncthreads();
    }

    const error_t error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

    if (threadIdx.x == 0)
        atomicAdd(global_error, error_block);
}


// update kernerls ---------------------------------------------------------------
// [A] row Change kernel
template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
__global__ void
vectorMatrixMultCompareRowWarpShared(bit_factor_t *A,
                                     const bit_factor_t * __restrict__ B,
                                     const bit_matrix_t * __restrict__ C,
                                     const index_t height,
                                     const index_t width,
                                     const index_t padded_width,
                                     const uint8_t factorDim,
                                     const index_t startrow,
                                     error_t *global_error,
                                     const uint32_t seed, 
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipManyDepth,
                                     const error_t weight)
{
    __shared__ bit_factor_t B_block[ 32 * WARPSPERBLOCK ];
    __shared__ bit_matrix_t C_block[ 32 * WARPSPERBLOCK ];

    // index_t warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const index_t warpId = blockIdx.x * WARPSPERBLOCK + threadIdx.x / warpSize;
    const index_t warpLane = threadIdx.x % warpSize;

    // index_t i = (warpId + startrow) % height;
    const index_t padded_height_blocks = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
    const index_t i = (startrow + warpId) % padded_height_blocks;

    fast_kiss_state32_t state;
    
    const bit_factor_t A_i = i < height ? A[i] : 0;
    bit_factor_t A_i_changed = 0;
    // bit_factor_t A_i_changed2 = 0;
    // if (warpLane == 0 && i < height) {
    if (i < height) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        A_i_changed = A_i ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
        // A_i_changed2 = A_i ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
        // A_i_changed = get_flip_mask_many(factorDim, state, flipManyDepth);
    }
    // A_i_changed = __shfl_sync(FULLMASK, A_i_changed, 0);
    
    const index_t vecRow = i / 32;
    const index_t vecFirst = vecRow * padded_width;
    const index_t vecLane = i % 32;
    const index_t col_in_tile = warpLane;
    const index_t padded_width_blocks = SDIV(width, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
    error_t error_thread = 0;
    // error_t error_thread2 = 0;
    for (index_t j = threadIdx.x; j < padded_width_blocks; j += WARPSPERBLOCK*32) {
        B_block[threadIdx.x] = (j < width) ? B[j] : 0;
        C_block[threadIdx.x] = (j < width) ? C[vecFirst + j] : 0;
        __syncthreads();

        if(i < height) {
            #pragma unroll
            for(index_t w = 0; w < WARPSPERBLOCK; ++w) {
                // const uint32_t dropout = fast_kiss32(state) | fast_kiss32(state) | fast_kiss32(state) & (1 << warpLane);
                // const bit_factor_t B_j = dropout ? B_block[w*warpSize + warpLane] : 0;
                const bit_factor_t B_j = B_block[w*warpSize + warpLane];
                const int C_ij = (C_block[w*warpSize + col_in_tile] >> vecLane) & 1;
                // index_t col = j / blockDim.x * blockDim.x + w*warpSize + warpLane;
                // bit_factor_t B_j = B[col];
                // int C_ij = (C[vecFirst + col] >> vecLane) & 1;

                const int product_new = (B_j & A_i_changed) ? 1 : 0;
                // const int product_new2 = (B_j & A_i_changed2) ? 1 : 0;
                const int product_old = (B_j & A_i        ) ? 1 : 0;

                // const float count_new = __popc(B_j & A_i_changed);
                // const float count_old = __popc(B_j & A_i);

                // if (col < width)
                error_thread += error_measure(product_new, C_ij, weight)
                              - error_measure(product_old, C_ij, weight);
                // error_thread += 0.001f * (__popc(A_i_changed) - __popc(A_i));
                // error_thread2 += error_measure(product_new2, C_ij, weight)
                //               - error_measure(product_old, C_ij, weight);
            }
        }
        __syncthreads();
    }
    if(i < height) {
        const error_t error_warp = warpReduceSum(error_thread);
        // const error_t error_warp2 = warpReduceSum(error_thread2);
        // Thread with warpLane==0 now has total error of warp

        // Thread 0 checks if new low has been found and applies if necessary
        if (warpLane == 0) {
        // if (i < height && warpLane == 0) {
            // if(error_warp < error_warp2) {
                // Metropolis–Hastings algorithm
                if (metro(state, error_warp, temperature, width)) {
                    A[i] = A_i_changed;
                    atomicAdd(global_error, error_warp);
                }
            // } else {
            //     // Metropolis–Hastings algorithm
            //     if (metro(state, error_warp2, temperature, width)) {
            //         A[i] = A_i_changed2;
            //         atomicAdd(global_error, error_warp2);
            //     }
            // }
        }
    }
}

// [B] col change kernel
template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
__global__ void
vectorMatrixMultCompareColWarpShared(const bit_factor_t * __restrict__ A,
                                     bit_factor_t *B,
                                     const bit_matrix_t * __restrict__ C,
                                     const index_t height,
                                     const index_t width,
                                     const index_t padded_width,
                                     const uint8_t factorDim,
                                     const index_t startcol,
                                     error_t *global_error,
                                     const uint32_t seed,
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipManyDepth,
                                     const error_t weight)
{
    __shared__ bit_factor_t A_block[32*WARPSPERBLOCK];
    // __shared__ bit_factor_t B_block[32];
    __shared__ bit_matrix_t C_block[32*WARPSPERBLOCK];
    // __shared__ int error_warps[32];

    // index_t warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const index_t warpIdIntern = threadIdx.x / warpSize;
    const index_t warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
    const index_t warpLane = threadIdx.x % warpSize;


    // index_t j = (startcol + warpId) % width;
    const index_t padded_width_blocks = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
    const index_t j = (startcol + warpId) % padded_width_blocks;

    fast_kiss_state32_t state;

    const bit_factor_t B_j = j < width ? B[j] : 0;
    bit_factor_t B_j_changed = 0;
    // bit_factor_t B_j_changed2 = 0;
    // if (warpLane == 0 && j < width) {
    if (j < width) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        B_j_changed = B_j ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
        // B_j_changed2 = B_j ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
        // B_j_changed = get_flip_mask_many(factorDim, state, flipManyDepth);
    }
    // B_j_changed = __shfl_sync(FULLMASK, B_j_changed, 0);
    
    error_t error_thread = 0;
    // error_t error_thread2 = 0;
    const index_t vecLane = warpLane;
    const index_t col_in_tile = j % 32;
    const index_t colFirst = j / 32 * 32;
    const index_t padded_height_blocks = SDIV(height, WARPSPERBLOCK*32) * WARPSPERBLOCK*32;
    for (index_t i = threadIdx.x; i < padded_height_blocks; i += WARPSPERBLOCK*32) {
        A_block[threadIdx.x] = (i < height) ? A[i] : 0;
        const index_t vecRow = i / 32;
        const index_t vecFirst = vecRow * padded_width + colFirst;
        C_block[threadIdx.x] = (vecRow < SDIV(height,32)) ? C[vecFirst + warpLane] : 0;
        __syncthreads();

        if (j < width) {
            #pragma unroll
            for(index_t w = 0; w < WARPSPERBLOCK; ++w) {
                // const uint32_t dropout = fast_kiss32(state) | fast_kiss32(state) | fast_kiss32(state) & (1 << warpLane);
                // const bit_factor_t A_i = dropout ? A_block[w*warpSize + warpLane] : 0;
                const bit_factor_t A_i = A_block[w*warpSize + warpLane];
                const int C_ij = (C_block[w*warpSize + col_in_tile] >> vecLane) & 1;

                // index_t row = i / blockDim.x * blockDim.x + w*warpSize + warpLane;
                // bit_factor_t A_i = (row < height) ? A[row] : 0;
                // int C_ij = (C[vecFirst + col_in_tile] >> vecLane) & 1;

                const int product_new = (A_i & B_j_changed) ? 1 : 0;
                // const int product_new2 = (A_i & B_j_changed2) ? 1 : 0;
                const int product_old = (A_i & B_j        ) ? 1 : 0;

                // const float count_new = __popc(A_i & B_j_changed);
                // const float count_old = __popc(A_i & B_j);

                // if (row < height)
                error_thread += error_measure(product_new, C_ij, weight)
                              - error_measure(product_old, C_ij, weight);
                // error_thread += 0.001f * (__popc(B_j_changed) - __popc(B_j));
                // error_thread2 += error_measure(product_new2, C_ij, weight)
                //               - error_measure(product_old, C_ij, weight);
            }
        }
        __syncthreads();
    }
    if (j < width) {
        const error_t error_warp = warpReduceSum(error_thread);
        // const error_t error_warp2 = warpReduceSum(error_thread2);
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
            // if(error_warp < error_warp2) {
                // Metropolis–Hastings algorithm
                if (metro(state, error_warp, temperature, height)) {
                    B[j] = B_j_changed;
                    atomicAdd(global_error, error_warp);
                }
            // } else {
            //     // Metropolis–Hastings algorithm
            //     if (metro(state, error_warp2, temperature, height)) {
            //         B[j] = B_j_changed2;
            //         atomicAdd(global_error, error_warp2);
            //     }
            // }
        }
    }
}


#endif
