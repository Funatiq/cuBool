#ifndef CUBIN_GPU_CUH
#define CUBIN_GPU_CUH

#include <vector>
#include <iostream>
#include <limits>

#include "helper/config.h"
#include "helper/rngpu.hpp"
#include "helper/cuda_helpers.cuh"

template <int rand_depth = 2>
__inline__ __device__
uint32_t get_flip_mask_many(fast_kiss_state32_t * state) {
    uint32_t bit_flip_mask = fast_kiss32(state);
    #pragma unroll
    for(int i = 1; i < rand_depth; ++i) {
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
uint32_t get_flip_mask(fast_kiss_state32_t * state, float flipManyChance = 1.0f) {
    float randomNumber = fast_kiss32(state) / (float) UINT32_MAX;

    return randomNumber < flipManyChance ? get_flip_mask_many(state) : get_flip_mask_one(state);
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

__global__
void computeFullDistance(const uint32_t *Ab, const uint32_t *Bb, const uint32_t *Cb, 
                      const int height, const int width,const int padded_width,
                      int *distance_test) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    //__shared__ volatile int shared_distance[THREADSPERBLOCK];
    //shared_distance[threadIdx.x] = 0;
    __shared__ int reductionArray[32];
    
    int error_thread = 0;
    if (j < width) {
        for (int i = 0; i < height; i++) {
            int lineSum = (Ab[i] & Bb[j]) ? 1 : 0;
            int intId = i / 32 * padded_width + j;
            int intLane = i % 32;
            // int intId = (j * height + i) / 32;
            // int intLane = (j * height + i) % 32;
            int truthEntry = (Cb[intId] >> (32 - intLane - 1)) & 1; 
            error_thread += lineSum ^ truthEntry;
        }
    }
    __syncthreads();
    
    int error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

   if (threadIdx.x == 0) {
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
                                     const float flipManyChance = 1.0f)
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

        currentRow_changed = currentRow ^ get_flip_mask(&state, flipManyChance);
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
                                     const float flipManyChance = 1.0f)
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

        currentCol_changed = currentCol ^ get_flip_mask(&state, flipManyChance);
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



class CuBin
{
    using bit_vector_t = uint32_t;
    using bit_matrix = std::vector<bit_vector_t>;

public:
    CuBin(const bit_matrix& A, const bit_matrix& B, const bit_matrix& C) {
        int device_id = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        max_parallel_lines = prop.multiProcessorCount * WARPSPERBLOCK;

        initialize(A, B, C);
    }

    ~CuBin() {
        clear();
    }

    bool initialize(const bit_matrix& A, const bit_matrix& B, const bit_matrix& C) {
        if( SDIV(A.size(),32) * B.size() != C.size()) {
            std::cerr << "CuBin construction: Matrix dimension mismatch." << std::endl;
            return false;
        }

        if(initialized) {
            std::cout << "CuBin already initialized. Please clear CuBin before reinitialization." << std::endl;
            return false;
        }

        height = A.size();
        // size_t height_padded = SDIV(height, WARPSPERBLOCK) * WARPSPERBLOCK;
        cudaMalloc(&d_A, sizeof(bit_vector_t) * height); CUERR

        width = B.size();
        // size_t width_padded = SDIV(width, WARPSPERBLOCK) * WARPSPERBLOCK;
        cudaMalloc(&d_B, sizeof(bit_vector_t) * width); CUERR
        
        size_t height_C = SDIV(height, 32);
        width_C_padded = SDIV(width, 32) * 32;
        cudaMalloc(&d_C, sizeof(bit_vector_t) * height_C * width_C_padded); CUERR

        cudaMemcpy(d_A, A.data(), sizeof(bit_vector_t) * height, cudaMemcpyHostToDevice); CUERR
        cudaMemcpy(d_B, B.data(), sizeof(bit_vector_t) * width, cudaMemcpyHostToDevice); CUERR
        cudaMemcpy2D(d_C, sizeof(bit_vector_t) * width_C_padded,
                     C.data(), sizeof(bit_vector_t) * width,
                     sizeof(bit_vector_t) * width,
                     height_C,
                     cudaMemcpyHostToDevice); CUERR


        cudaMallocHost(&distance, sizeof(int)); CUERR
        cudaMalloc(&d_distance, sizeof(int)); CUERR
        cudaMemset(d_distance, 0, sizeof(int)); CUERR

        computeFullDistance <<< SDIV(width, THREADSPERBLOCK), THREADSPERBLOCK >>>
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

    void getFactors(bit_matrix& A, bit_matrix& B) {
        if(!initialized) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }
        A.resize(height);
        cudaMemcpy(A.data(), d_A, sizeof(bit_vector_t) * height, cudaMemcpyDeviceToHost); CUERR
        
        B.resize(width);
        cudaMemcpy(B.data(), d_B, sizeof(bit_vector_t) * width, cudaMemcpyDeviceToHost); CUERR
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
        size_t verbosity = 1;
        size_t linesAtOnce = 0;
        size_t maxIterations = 0;
        int distanceThreshold = 0;
        size_t distanceShowEvery = std::numeric_limits<size_t>::max();
        float tempStart = 0.0f;
        float tempFactor = 0.98f;
        size_t tempReduceEvery = std::numeric_limits<size_t>::max();
        uint32_t seed = 0;
        bool loadBalance = false;
        float flipManyChance = 0.1f;
    };

    void run(const CuBin_config& config) {
        if(!initialized) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t linesAtOnce = config.linesAtOnce;
        if(config.loadBalance)
            linesAtOnce = linesAtOnce / max_parallel_lines * max_parallel_lines;

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
        while( *distance > config.distanceThreshold && iteration++ < config.maxIterations) {

            // Change rows
            int lineToBeChanged = (fast_kiss32(&state) % height) / WARPSPERBLOCK * WARPSPERBLOCK;
            uint32_t gpuSeed = fast_kiss32(&state) + iteration;

            vectorMatrixMultCompareRowWarpShared 
                <<< SDIV(min(linesAtOnce, height), WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                (d_A, d_B, d_C, height, width, width_C_padded,
                 lineToBeChanged, d_distance, gpuSeed, temperature, config.flipManyChance);

            cudaDeviceSynchronize(); CUERR

            // Change cols
            lineToBeChanged = (fast_kiss32(&state) % width) / WARPSPERBLOCK * WARPSPERBLOCK;
            gpuSeed = fast_kiss32(&state) + iteration;

            vectorMatrixMultCompareColWarpShared 
                <<< SDIV(min(linesAtOnce, width), WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                (d_A, d_B, d_C, height, width, width_C_padded,
                 lineToBeChanged, d_distance, gpuSeed, temperature, config.flipManyChance);

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

        computeFullDistance <<< SDIV(width, THREADSPERBLOCK), THREADSPERBLOCK >>>
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
    bit_matrix A;
    bit_matrix B;
    bit_matrix C;
    bit_vector_t *d_A;
    bit_vector_t *d_B;
    bit_vector_t *d_C;
    int *distance;
    int *d_distance;
    // size_t height_padded;
    size_t height = 0;
    size_t width = 0;
    size_t width_C_padded = 0;
    int max_parallel_lines;
};

#endif
