#ifndef CUBIN_GPU_CUH
#define CUBIN_GPU_CUH

#include <vector>
#include <iostream>
#include <limits>
#include <type_traits>

#include "helper/config.h"
#include "helper/rngpu.hpp"
#include "helper/cuda_helpers.cuh"

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
    uint32_t randomNumber = fast_kiss32(state);
    #pragma unroll
    for (int i = 0; i < DIM_PARAM; i++) {
        bit_flip_mask |= (randomNumber >> i) & 11 ? (0 << 32 - 1 - i) : (1 << 32 - 1 - i);
    }
    return bit_flip_mask;
}

__inline__ __device__
uint32_t get_flip_mask_one(fast_kiss_state32_t * state) {
    uint32_t lane = fast_kiss32(state) % DIM_PARAM;
    return 1 << (32 - 1 - lane);
}

__inline__ __device__
uint32_t get_flip_mask(fast_kiss_state32_t * state,
                       const float flipManyChance = 1.0f,
                       const uint32_t flipManyDepth = 2) {
    float randomNumber = fast_kiss32(state) / (float) UINT32_MAX;

    return randomNumber < flipManyChance ? get_flip_mask_many(state, flipManyDepth) : get_flip_mask_one(state);
}


__inline__ __device__
bool metro(fast_kiss_state32_t * state, const int error, const float temperature) {
    if(error < 0)
        return true;
    if(temperature <= 0)
        return false;
    // Metropolis–Hastings algorithm
    float randomNumber = fast_kiss32(state) / (float) UINT32_MAX;
    float metro = fminf(1.0f, expf((float) - error / temperature));
    return randomNumber < metro;
}

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
    const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int warpIdIntern = threadIdx.x / warpSize;
    const int warpLane = threadIdx.x % warpSize;

    __shared__ int reductionArray[WARPSPERBLOCK];

    const uint32_t dim_mask = FULLMASK >> (32 - DIM_PARAM);
    
    const int i = warpId;
    const int k = warpLane;
    int error_warp = 0;
    if (i < height) {
        for (int j = 0; j < width; ++j) {
            int lineSum = __any_sync(dim_mask, (A[i*warpSize + k] > 0.5f) && (B[j*warpSize + k] > 0.5f)) ? 1 : 0;
            if(warpLane == 0) {
                const int intId = i / 32 * padded_width + j;
                const int intLane = i % 32;

                const int truthEntry = (Cb[intId] >> (32 - intLane - 1)) & 1; 
                error_warp += lineSum ^ truthEntry;
            }
        }
    }
    if(warpLane == 0)
        reductionArray[warpIdIntern] = error_warp;
    __syncthreads();

    int error_block;
    if(warpIdIntern == 0) {
        error_block = warpReduceSum(reductionArray[warpLane], (32 - WARPSPERBLOCK - 1));
        
        // Thread with threadIdx.x==0 now has total error of block

       if (threadIdx.x == 0)
            atomicAdd(distance_test, error_block);
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

    // int rowToBeChanged = (warpId + startrow) % height;
    const int padded_height_blocks = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int rowToBeChanged = (startrow + warpId) % padded_height_blocks;

    fast_kiss_state32_t state;
    
    const bit_vector_t currentRow = rowToBeChanged < height ? A[rowToBeChanged] : 0;
    bit_vector_t currentRow_changed = 0;
    if (warpLane == 0 && rowToBeChanged < height) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        currentRow_changed = currentRow ^ get_flip_mask(&state, flipManyChance, flipDepth);
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


    // int colToBeChanged = (startcol + warpId) % width;
    const int padded_width_blocks = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
    const int colToBeChanged = (startcol + warpId) % padded_width_blocks;

    fast_kiss_state32_t state;

    const bit_vector_t currentCol = colToBeChanged < width ? B[colToBeChanged] : 0;
    bit_vector_t currentCol_changed = 0;
    if (warpLane == 0 && colToBeChanged < width) {
        state = get_initial_fast_kiss_state32(seed + warpId);

        currentCol_changed = currentCol ^ get_flip_mask(&state, flipManyChance, flipManyDepth);
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

// [A] row Change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareRowWarpShared(float *A,
                                     const float * __restrict__ B,
                                     const bit_vector_t * __restrict__ C,
                                     const int height,
                                     const int width,
                                     const int padded_width,
                                     const int startrow,
                                     int *global_error,
                                     const uint32_t seed, 
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipDepth);

// [B] col change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareColWarpShared(const float * __restrict__ A,
                                     float *B,
                                     const bit_vector_t * __restrict__ C,
                                     const int height,
                                     const int width,
                                     const int padded_width,
                                     const int startcol,
                                     int *global_error,
                                     const uint32_t seed,
                                     const float temperature,
                                     const float flipManyChance,
                                     const uint32_t flipManyDepth);


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
                        (d_A, d_B, d_C, height, width, width_C_padded, d_distance); CUERR

        cudaMemcpy(distance, d_distance, sizeof(int), cudaMemcpyDeviceToHost);

        return initialized = true;
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
        A.resize(height);
        cudaMemcpy(A.data(), d_A, sizeof(factor_t) * height, cudaMemcpyDeviceToHost); CUERR
        
        B.resize(width);
        cudaMemcpy(B.data(), d_B, sizeof(factor_t) * width, cudaMemcpyDeviceToHost); CUERR
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
                        (d_A, d_B, d_C, height, width, width_C_padded, d_distance_proof); CUERR

        cudaMemcpy(distance_proof, d_distance_proof, sizeof(int), cudaMemcpyDeviceToHost);

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
