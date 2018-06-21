#ifndef FLOAT_KERNELS_CUH
#define FLOAT_KERNELS_CUH

// init kernel ---------------------------------------------------------------
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

#endif
