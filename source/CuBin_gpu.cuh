#ifndef CUBIN_GPU_CUH
#define CUBIN_GPU_CUH

#include "helper/matrix_mult.cuh"

#define FULLMASK 0xffffffff

__inline__ __device__
int warpReduceSum(int val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULLMASK, val, offset);
    return val;
}

__inline__ __device__
int blockReduceSum(int val, int* reductionArray) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum(val);
    if (lane == 0) reductionArray[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? reductionArray[lane] : 0;
        val = warpReduceSum(val);
    }
    return val;
}


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
vectorMatrixMultCompareRow( bit_vector_t *A, bit_vector_t *B, bit_vector_t *C, 
                            const int width, const int height,
                            const int padded_width, const int padded_height,
                            const int startrow, int *global_error,
                            const uint32_t seed, const float temperature) {

    int rowToBeChanged = (blockIdx.x + startrow) % padded_height;
    if (rowToBeChanged >= height) return;

    fast_kiss_state32_t state;
    __shared__ int reductionArray[32];
    __shared__ bit_vector_t shared_currentRow_changed;
    
    bit_vector_t currentRow = A[rowToBeChanged];
    if (threadIdx.x == 0) {
        state = get_initial_fast_kiss_state32(seed + blockIdx.x);

        shared_currentRow_changed = currentRow ^ get_flip_mask(&state);
        // shared_currentRow_changed = currentRow ^  get_flip_mask_11(&state);
    }
    __syncthreads();
    
    bit_vector_t currentRow_changed = shared_currentRow_changed;
    int error_thread = 0;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        // if (rowToBeChanged < height) {
            bit_vector_t currentColThread = B[j];
            int intId = rowToBeChanged / 32 * padded_width + j;
            int intLane = rowToBeChanged % 32;
            int cTruthEntry = (C[intId] >> (32 - intLane - 1)) & 1; 

            int cEntryOld = (currentRow         & currentColThread) ? 1 : 0;
            int cEntryNew = (currentRow_changed & currentColThread) ? 1 : 0;
            error_thread += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
        // }
    }
    __syncthreads();

    int error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

    // Thread 0 checks if new low has been found and applies if necessary
    if (threadIdx.x == 0) {
        // Metropolis–Hastings algorithm
        if (metro(&state, error_block, temperature)) {
            A[rowToBeChanged] = currentRow_changed;
            atomicAdd(global_error, error_block);
        }
    }
}

// [B] col change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareCol(bit_vector_t *A, bit_vector_t *B, bit_vector_t *C, 
                            const int width, const int height,
                            const int padded_width, const int padded_height,
                            const int startcol, int *global_error,
                            const uint32_t seed, const float temperature) {

    int colToBeChanged = (blockIdx.x + startcol) % padded_width;
    if (colToBeChanged >= width) return;

    fast_kiss_state32_t state;
    __shared__ int reductionArray[32];
    __shared__ bit_vector_t shared_currentCol_changed;

    bit_vector_t currentCol = B[colToBeChanged];
    if (threadIdx.x == 0) {
        state = get_initial_fast_kiss_state32((seed + blockIdx.x) % UINT32_MAX);

        shared_currentCol_changed = currentCol ^ get_flip_mask(&state);
        // shared_currentCol_changed = currentCol ^ get_flip_mask_11(&state);
    }
    __syncthreads();
    
    bit_vector_t currentCol_changed = shared_currentCol_changed;
    int error_thread = 0;
    for (int i = threadIdx.x; i < height; i += blockDim.x) {
        // if (colToBeChanged < width) {
            bit_vector_t currentRowThread = A[i];
            int intId = i / 32 * padded_width + colToBeChanged;
            int intLane = i % 32;
            int cTruthEntry = (C[intId] >> (32 - intLane - 1)) & 1; 
            
            int cEntryOld = (currentCol         & currentRowThread) > 0 ? 1 : 0;        
            int cEntryNew = (currentCol_changed & currentRowThread) > 0 ? 1 : 0;
            error_thread += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
        // }
    }
    __syncthreads();

    int error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

    // Thread 0 checks if new low has been found and applies if necessary
    if (threadIdx.x == 0) {
        // Metropolis–Hastings algorithm
        if (metro(&state, error_block, temperature)) {
            B[colToBeChanged] = currentCol_changed;
            atomicAdd(global_error, error_block);
        }
    }
}


// Used for debugging and checking for correctness, not optimized
template<typename bit_vector_t, typename element_t = uint32_t>
void aftertestGPU(  bit_vector_t *d_Ab, bit_vector_t *d_Bb, bit_vector_t *d_C0b, 
                    int width, int height, int padded_width) {
    TIMERSTART(aftertestGPU)

    element_t *A = (element_t*) malloc(sizeof(element_t) * DIM_PARAM * height);
    element_t *B = (element_t*) malloc(sizeof(element_t) * width * DIM_PARAM);
    element_t *C0 = (element_t*) malloc(sizeof(element_t) * width * height);

    bit_vector_t *Ab = (bit_vector_t*) malloc(sizeof(bit_vector_t) * height);
    bit_vector_t *Bb = (bit_vector_t*) malloc(sizeof(bit_vector_t) * width);

    int padded_height_32 = SDIV(height,32);
    int sizeCb = width * padded_height_32;
    // int sizeCb = ((long long) (height * width) / 32.0 + 1);
    bit_vector_t *C0b = (bit_vector_t*) malloc(sizeof(bit_vector_t) * sizeCb);
    
    element_t *d_A, *d_B;
    cudaMalloc((void**)&d_A, sizeof(element_t) * DIM_PARAM * height);                                           CUERR
    cudaMalloc((void**)&d_B, sizeof(element_t) * width * DIM_PARAM);                                            CUERR
    
    cudaMemcpy(Ab, d_Ab, sizeof(bit_vector_t) * height, cudaMemcpyDeviceToHost);                                CUERR
    cudaMemcpy(Bb, d_Bb, sizeof(bit_vector_t) * width, cudaMemcpyDeviceToHost);                                 CUERR
    cudaMemcpy2D(C0b, sizeof(bit_vector_t) * width,
                 d_C0b, sizeof(bit_vector_t) * padded_width,
                 sizeof(bit_vector_t) * width,
                 padded_height_32,
                 cudaMemcpyDeviceToHost);                              CUERR

    uint32_t counterDensity = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < DIM_PARAM; j++){
            A[i * DIM_PARAM + j] = (Ab[i] >> 32 - j - 1) & 1;
            if (A[i*DIM_PARAM + j]) counterDensity++;
        }
    }
    float densityA = counterDensity / (double) (height*DIM_PARAM);
    
    counterDensity = 0;
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < DIM_PARAM; j++) {
            B[j * width + i] = (Bb[i] >> 32 - j - 1) & 1;
            if (B[j * width + i]) counterDensity++;
        }
    }
    float densityB = counterDensity / (double) (DIM_PARAM * width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // int intId = (i * height + j) / 32;
            // int intLane = (i * height + j) % 32;
            int intId = i / 32 * width + j;
            int intLane = i % 32;           
            C0[i * width + j] = (C0b[intId] >> 32 - intLane - 1) & 1;
        }
    }
    
    cudaMemcpy(d_A, A, sizeof(element_t)*height*DIM_PARAM, cudaMemcpyHostToDevice);                           CUERR
    cudaMemcpy(d_B, B, sizeof(element_t)*width*DIM_PARAM, cudaMemcpyHostToDevice);                            CUERR   
    
    // Doing a check two times: once with A,B and once with Ab,Bb just to make sure
    // First check
    element_t *C_test_GPU = (element_t *) malloc(sizeof(element_t) * width * height);
    element_t *d_C_test_GPU;
    cudaMalloc((void **) &d_C_test_GPU, sizeof(element_t) * height * width);                             CUERR
    
    matrixMultiply <<< SDIV(width, THREADSPERBLOCK), THREADSPERBLOCK >>> 
                        (d_Ab, d_Bb, d_C_test_GPU, width, height);                                      CUERR
                        
    cudaMemcpy(C_test_GPU, d_C_test_GPU, sizeof(element_t) * height * width, 
                    cudaMemcpyDeviceToHost);                                                            CUERR
    
    int error_test_GPU = 0;
    for (int i = 0; i < height * width; i++)
        error_test_GPU += (((C0[i] - C_test_GPU[i]) * (C0[i] - C_test_GPU[i])));

    // Second check with regular matrix multiplication
    dim3 dimGrid((width - 1) / TILE_WIDTH + 1, (height - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiplyInt <<< dimGrid, dimBlock >>> 
                            (d_A, d_B, d_C_test_GPU, height, width, DIM_PARAM);                         CUERR

    cudaMemcpy(C_test_GPU, d_C_test_GPU, sizeof(element_t) * height * width, cudaMemcpyDeviceToHost);         CUERR
    int error_test_GPU_2 = 0;
    for (int i = 0; i < height * width; i++)
        error_test_GPU_2 += (((C0[i] - C_test_GPU[i]) * (C0[i] - C_test_GPU[i])));
    
    TIMERSTOP(aftertestGPU)
    printf("Aftertest error between C0 and C on GPU (bitwise): %i\n", error_test_GPU);
    printf("Aftertest error between C0 and C on GPU (float): %i\n", error_test_GPU_2);
    printf("Density A: %f, Density B: %f\n", densityA, densityB);
    
}


// [A] row Change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareRowWarp( bit_vector_t *A, bit_vector_t *B, bit_vector_t *C, 
                            int width, int height, int padded_width,
                            int startrow, int *global_error,
                            uint32_t seed, float temperature) {

    int rowToBeChanged = (blockIdx.x + startrow) % height;

    fast_kiss_state32_t state;
    
    bit_vector_t currentRow = A[rowToBeChanged];
    bit_vector_t currentRow_changed;
    if (threadIdx.x == 0) {
        state = get_initial_fast_kiss_state32(seed + blockIdx.x);

        currentRow_changed = currentRow ^ get_flip_mask(&state);
    }
    currentRow_changed = __shfl_sync(FULLMASK, currentRow_changed, 0);
    
    int error_thread = 0;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        if (j < width) {
            bit_vector_t currentColThread = B[j];
            int intId = rowToBeChanged / 32 * padded_width + j;
            int intLane = rowToBeChanged % 32;
            int cTruthEntry = (C[intId] >> (32 - intLane - 1)) & 1; 

            int cEntryOld = (currentRow         & currentColThread) ? 1 : 0;
            int cEntryNew = (currentRow_changed & currentColThread) ? 1 : 0;
            error_thread += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
        }
    }
    int error_warp = warpReduceSum(error_thread);
    // Thread with threadIdx.x==0 now has total error of warp

    // Thread 0 checks if new low has been found and applies if necessary
    if (threadIdx.x == 0) {
        // Metropolis–Hastings algorithm
        if (metro(&state, error_warp, temperature)) {
            A[rowToBeChanged] = currentRow_changed;
            atomicAdd(global_error, error_warp);
        }
    }
}

// [B] col change kernel
template<typename bit_vector_t>
__global__ void
vectorMatrixMultCompareColWarp(bit_vector_t *A, bit_vector_t *B, bit_vector_t *C, 
                            int width, int height, int padded_width,
                            int startcol, int *global_error,
                            uint32_t seed, float temperature) {

    int colToBeChanged = (blockIdx.x + startcol) % width;

    fast_kiss_state32_t state;

    bit_vector_t currentCol = B[colToBeChanged];
    bit_vector_t currentCol_changed;
    if (threadIdx.x == 0) {
        state = get_initial_fast_kiss_state32((seed + blockIdx.x) % UINT32_MAX);

        currentCol_changed = currentCol ^ get_flip_mask(&state);
    }
    currentCol_changed = __shfl_sync(FULLMASK, currentCol_changed, 0);
    
    int error_thread = 0;
    for (int i = threadIdx.x; i < height; i += blockDim.x) {
        if (i < height) {
            bit_vector_t currentRowThread = A[i];
            int intId = i / 32 * padded_width + colToBeChanged;
            int intLane = i % 32;
            int cTruthEntry = (C[intId] >> (32 - intLane - 1)) & 1; 
            
            int cEntryOld = (currentCol         & currentRowThread) > 0 ? 1 : 0;        
            int cEntryNew = (currentCol_changed & currentRowThread) > 0 ? 1 : 0;
            error_thread += (cEntryNew ^ cTruthEntry) - (cEntryOld ^ cTruthEntry);
        }
    }
    int error_warp = warpReduceSum(error_thread);
    // Thread with threadIdx.x==0 now has total error of warp

    // Thread 0 checks if new low has been found and applies if necessary
    if (threadIdx.x == 0) {
        // Metropolis–Hastings algorithm
        if (metro(&state, error_warp, temperature)) {
            B[colToBeChanged] = currentCol_changed;
            atomicAdd(global_error, error_warp);
        }
    }
}


#endif