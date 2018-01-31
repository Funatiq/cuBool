#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/types.h>
#include <math.h>
#include <time.h>
#include "rngpu.hpp"
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

using namespace std;

//texture<uint, cudaTextureType2D, cudaReadModeElementType> texRef;

void readInputFileData(uint32_t**, uint32_t**, int*, int*, double*, string);

void readInputFileTSV(uint32_t**, uint32_t**, int*, int*, double*, string);

//void readInputFileMovieLens(uint32_t**, uint32_t**, int*, int *, double*, string);

bool endsWith(const string&, const string&);

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

// Main
//
int main(int argc, char **argv) {
    cudaProfilerStart();
    /* ./a.out  [data]
                [updateStep]
                [lines at once]
                [threshold]
                [gpu iterations]
                [startingTemperature]
                [iterations till temperature is reduced]
    */
    if(argc == 1) return 0;
    std::string filename = argv[1];
    int updateStep = argc > 2 ? atoi(argv[2]) : 1000;
    int linesAtOnce = argc > 3 ? atoi(argv[3]) : 4000;
    float threshold = argc > 4 ? atof(argv[4]) : 0;
    int gpuiterations = argc > 5 ? atoi(argv[5]) : 10000;
    int maxIterationsNoImp = argc > 6 ? atoi(argv[6]) : 10000;
    float temperature = argc > 7 ?  atof(argv[7]) : 0;
    int iterationsTillReduced = argc > 8 ? (atoi(argv[8]) > 0? atoi(argv[8]) : INT_MAX) : INT_MAX;
    float tempFactor = argc > 9 ? atof(argv[9]) : 0.99;
    int seed = argc > 10 ? atoi(argv[10]) : 41;
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);
    
    // Discard first 100000 entries of PRNG
    for (int i = 0; i < 100000; i++)
        fast_kiss32(&state);
    
    // Read file and save matrix in C0 and d_C0
    uint32_t *C0, *d_C0;
    int width, height;
    double density;
    
    // Dense TSV
    std::string ending = "tsv";
    if (endsWith(filename, ending)) {
        readInputFileTSV(&C0, &d_C0, &width, &height, &density, filename);
    } 
    
    // COO coordinates
    ending = "dat";
    if (endsWith(filename, ending)) {
        readInputFileData(&C0, &d_C0, &width, &height, &density, filename);
    }
        
    // Initialize Texture Memory with C0
    // initializeTextureMemory(&C0, width, height);

    // Initialize Ab, Bb, d_Ab, d_Bb, all bitwise used matrices
    uint32_t *Ab, *Bb;
    uint32_t *d_Ab, *d_Bb;
    initializeFactors(&Ab, &Bb, &d_Ab, &d_Bb, width, height, density, &state);
    // A and B now initialized on device and host

    // Calculate original error, save it two times each one for GPU, one for CPU
    int error_C0_C_start = 0;
    int error_C0_C       = 0;
    int *d_error_C0_C_start, *d_error_C0_C;
    cudaMalloc((void **) &d_error_C0_C, sizeof(int));                                                           CUERR
    cudaMalloc((void **) &d_error_C0_C_start, sizeof(int));                                                     CUERR
    cudaMemcpy(d_error_C0_C_start, &error_C0_C_start, sizeof(int), cudaMemcpyHostToDevice);                     CUERR
    
    computeStartError(d_Ab, d_Bb, d_C0, width, height, &d_error_C0_C_start, &error_C0_C_start);
    
    cudaMemcpy(d_error_C0_C, d_error_C0_C_start, sizeof(int), cudaMemcpyDeviceToDevice);
    error_C0_C = error_C0_C_start;
    // Now the starting errors is in stored in 4 values
    // error_C0_C_start, error_C0_C on CPU and GPU


    // MAIN PART
    // on GPU
    int iterations = 0;
    int toBeChanged;
    int iterationsNoImp = 0;
    int error_C0_C_before = 0;
    threshold *= (width*height);
    #ifdef PERF
    vector<double> errorVector; // vector for error measurement
    vector<int> impVector;
    updateStep = 1;
    #endif
    printf("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("- Starting %i GPU iterations, pulling error every %i steps -\n", gpuiterations, updateStep);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    TIMERSTART(GPUKERNELLOOP)
    while (threshold < error_C0_C && iterations < gpuiterations && iterationsNoImp < maxIterationsNoImp) {
        iterations++;
        if (iterations % iterationsTillReduced == 0)
            temperature *= tempFactor;
        
        // Pull error from GPU to show it
        if (iterations % updateStep == 0) {
            #ifndef PERF
            printf("Current error: %f\n", error_C0_C / (double) (width * height));
            #endif
            if (error_C0_C_before - error_C0_C == 0) {
                iterationsNoImp += updateStep;
            } else {
                #ifdef PERF
                impVector.push_back(iterationsNoImp);
                #endif
                iterationsNoImp = 0;
            }
            error_C0_C_before = error_C0_C;
            #ifdef PERF
            errorVector.push_back(error_C0_C / (double) (width * height));
            #endif
            //checkDistance(d_Ab, d_Bb, d_C0, height, width);
        }

        // Change col
        toBeChanged = ((unsigned int) fast_kiss32(&state)) % width;
        vectorMatrixMultCompareCol <<< min(linesAtOnce, width), THREADSPERBLOCK >>>
                                        (d_Ab, d_Bb, d_C0, width, height, toBeChanged, d_error_C0_C, 
                                        ((fast_kiss32(&state) + iterations) % UINT32_MAX), temperature);        CUERR
        cudaDeviceSynchronize();                                                                                CUERR

        // Change row
        toBeChanged = ((unsigned int) fast_kiss32(&state)) % height;
        vectorMatrixMultCompareRow <<< min(linesAtOnce, height), THREADSPERBLOCK >>>
                                        (d_Ab, d_Bb, d_C0, width, height, toBeChanged, d_error_C0_C, 
                                        ((fast_kiss32(&state) + iterations) % UINT32_MAX), temperature);        CUERR
        cudaDeviceSynchronize();                                                                                CUERR
        
        cudaMemcpy(&error_C0_C, d_error_C0_C, sizeof(int), cudaMemcpyDeviceToHost);                             CUERR
    }
    // Pull final error from GPU
    cudaMemcpy(&error_C0_C, d_error_C0_C, sizeof(int), cudaMemcpyDeviceToHost);                                 CUERR
    printf("- - - - - - - - -\n");
    printf("Final Error on GPU: %f, %i wrong entries\n", error_C0_C / (double) (height * width), error_C0_C);
    TIMERSTOP(GPUKERNELLOOP)
    
    // Aftertest GPU
    #ifdef TEXT
    //aftertestGPU(d_Ab, d_Bb, d_C0, width, height);
    #endif
    
    writeToFiles(d_Ab, d_Bb, width, height);

    
    #ifdef PERF
    string writeFile = string("perf.csv");
    ofstream myfile(writeFile);
    if (myfile.is_open()) {
        myfile << "x,y\n";
        for (int i = 0; i < errorVector.size(); i++) {
            myfile << (i * timeGPUKERNELLOOP / (double) iterations) / (double) 1000 << "," << errorVector[i] << "\n";            
        }
    }
    writeFile = string("update.csv");
    ofstream myfile1(writeFile);
    if (myfile1.is_open()) {
        myfile1 << "x,y\n";
        for (int i = 0; i < impVector.size(); i++) {
            myfile1 << i << "," << impVector[i] << "\n";        
        }
    }
    #endif

    #ifdef CPU
    // CPU COMPUTATION
    //CPUcomputation(Ab, Bb, C0, width, height, error_C0_C_start, 42, updateStep, threshold, linesAtOnce);
    // Aftertest CPU
    //aftertestCPU(Ab, Bb, d_Ab, d_Bb, C0, width, height);
    #endif

    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");


    // Cleaning up
    cudaProfilerStop();
    cudaDeviceReset();
    free(Ab);
    free(Bb);
    free(C0);
    cudaFree(d_Ab);
    cudaFree(d_Bb);
    cudaFree(d_C0);
    cudaFree(d_error_C0_C);
    cudaFree(d_error_C0_C_start);
    return 0;
}

__inline__ __device__
int warpReduceSum(int val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

// [A] row Change kernel
__global__ void
vectorMatrixMultCompareRow( uint32_t *A, uint32_t *B, uint32_t *C, 
                            int width, int height, 
                            int startrow, int *global_error,
                            uint32_t seed, float temperature) {

    int rowToBeChanged = (blockIdx.x + startrow) % height;
    int cTruthEntry;
    int cEntryOld;
    int cEntryNew;
    int error_thread;
    //int intId;
    //int intLane;
    uint32_t randomNumber;
    //uint32_t currentColThread;
    //uint32_t currentRow;
    //uint32_t currentRow_changed;
    float metro;
    __shared__ fast_kiss_state32_t state;
    __shared__ int reductionArray[32];
    //__shared__ uint32_t shared_currentRow_changed;
    __shared__ uint32_t shared_currentRow[DIM_PARAM];
    __shared__ uint32_t shared_currentRow_changed[DIM_PARAM];
    
    //currentRow = A[rowToBeChanged];
    if (threadIdx.x == 0) {
        //shared_currentRow_changed = currentRow; // load row to be changed in shared memory
        
        state = get_initial_fast_kiss_state32((seed + blockIdx.x) % UINT32_MAX);
        randomNumber = fast_kiss32(&state);
        #pragma unroll
        for (int i = 0; i < DIM_PARAM; i++){
            shared_currentRow[i] = A[(rowToBeChanged*DIM_PARAM) + i];
            shared_currentRow_changed[i] = shared_currentRow[i];
            shared_currentRow_changed[i] ^= (randomNumber >> i) & 11 ? 0 : 1;
            // shared_currentRow_changed ^= (randomNumber >> i) & 11 ? (0 << i) : (1 << i);
        }
    }
    __syncthreads();
    
    // currentRow_changed = shared_currentRow_changed;
    error_thread = 0;
    for (int i = 0; i <= ((width - 1) / blockDim.x); i++) {
        if ((i * blockDim.x + threadIdx.x) < width) {
            //currentColThread = B[i * blockDim.x + threadIdx.x];
            //intId = (((i * blockDim.x + threadIdx.x) * height) + rowToBeChanged) / 32;
            //intLane = (((i * blockDim.x + threadIdx.x) * height) + rowToBeChanged) % 32;
            //cTruthEntry = (C[intId] >> 32 - intLane - 1) & 1; 
            
            cTruthEntry = C[rowToBeChanged*width + (i*blockDim.x) + threadIdx.x];

            cEntryOld = cEntryNew = 0;
            for (int j = 0; j < DIM_PARAM; j++){
                if (cEntryOld < 0.5)
                    cEntryOld += shared_currentRow[j] * B[(j*width) + (i*blockDim.x + threadIdx.x)];                
                if (cEntryNew < 0.5)
                    cEntryNew += shared_currentRow_changed[j] * B[(j*width) + (i*blockDim.x + threadIdx.x)];
            }
            //cEntryOld = (currentRow         & currentColThread) > 0 ? 1 : 0;
            //cEntryNew = (currentRow_changed & currentColThread) > 0 ? 1 : 0;
            error_thread += ((cEntryNew - cTruthEntry) * (cEntryNew - cTruthEntry)) -
                            ((cEntryOld - cTruthEntry) * (cEntryOld - cTruthEntry));
        }
    }
    __syncthreads();

    // Reduction across block
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    error_thread = warpReduceSum(error_thread);
    if (lane == 0) reductionArray[wid] = error_thread;
    __syncthreads();
    error_thread = (threadIdx.x < blockDim.x / warpSize) ? reductionArray[lane] : 0;
    if (wid == 0) error_thread = warpReduceSum(error_thread);
    // Thread with threadIdx.x==0 now has error in error_thread

    // Thread 0 checks if new low has been found and applies if necessary
    if (threadIdx.x == 0) {
        if (error_thread < 0) {
             //A[rowToBeChanged] = shared_currentRow_changed;
            for(int i=0; i<DIM_PARAM; i++)
                A[rowToBeChanged*DIM_PARAM+i] = shared_currentRow_changed[i];
            atomicAdd(global_error, error_thread);
        } else { // Metropolis–Hastings algorithm
            randomNumber = fast_kiss32(&state) / (double) UINT32_MAX;
            metro = temperature > 0.0f ? fminf(1, expf(-error_thread / temperature)) : 0 ;
            if (randomNumber < metro) {
                // A[rowToBeChanged] = shared_currentRow_changed;
                for(int i=0; i<DIM_PARAM; i++)
                    A[rowToBeChanged*DIM_PARAM+i] = shared_currentRow_changed[i];
                atomicAdd(global_error, error_thread);
            }
        }
    }
}

// [B] col change kernel
//
__global__ void
vectorMatrixMultCompareCol(uint32_t *A, uint32_t *B, uint32_t *C, 
                            int width, int height, 
                            int startcol, int *global_error,
                            uint32_t seed, float temperature) {

    int colToBeChanged = (blockIdx.x + startcol) % width;
    int cTruthEntry;
    int cEntryOld;
    int cEntryNew;
    int error_thread;
    //int intId;
    //int intLane;
    uint32_t randomNumber;
    //uint32_t currentRowThread;
    //uint32_t currentCol;
    //uint32_t currentCol_changed;
    float metro;
    __shared__ fast_kiss_state32_t state;
    __shared__ int shared[32];
    //__shared__ uint32_t shared_currentCol_changed;
    __shared__ uint32_t shared_currentCol[DIM_PARAM];
    __shared__ uint32_t shared_currentCol_changed[DIM_PARAM];

    //currentCol = B[colToBeChanged];
    if (threadIdx.x == 0) {
        //shared_currentCol_changed = currentCol; // load row to be changed in shared memory

        state = get_initial_fast_kiss_state32((seed + blockIdx.x) % UINT32_MAX);
        randomNumber = fast_kiss32(&state);
        #pragma unroll
        for (int i = 0; i < DIM_PARAM; i++){
            shared_currentCol[i] = B[i*width + colToBeChanged];
            shared_currentCol_changed[i] = shared_currentCol[i];
            shared_currentCol_changed[i] ^= (randomNumber >> i) & 11 ? 0 : 1;
            //shared_currentCol_changed ^= (randomNumber >> i) & 11 ? (0 << i) : (1 << i);
        }
    }
    __syncthreads();
    
    //currentCol_changed = shared_currentCol_changed;
    error_thread = 0;
    for (int i = 0; i <= ((height - 1) / blockDim.x); i++) {
        if ((i * blockDim.x + threadIdx.x) < height) {
            //currentRowThread = A[i * blockDim.x + threadIdx.x];
            //intId = ((colToBeChanged * height) + (i * blockDim.x + threadIdx.x)) / 32;
            //intLane = ((colToBeChanged * height) + (i * blockDim.x + threadIdx.x)) % 32;
            //cTruthEntry = (C[intId] >> 32 - intLane - 1) & 1; 
            
			cTruthEntry = C[(i*blockDim.x + threadIdx.x)*width + colToBeChanged];
            
            cEntryOld = cEntryNew = 0;
            for (int j = 0; j < DIM_PARAM; j++) {
                if (cEntryOld < 0.5)
                    cEntryOld += shared_currentCol[j] * A[(blockDim.x*i + threadIdx.x) * DIM_PARAM + j];
                if (cEntryNew < 0.5)
                    cEntryNew += shared_currentCol_changed[j] * A[(blockDim.x*i + threadIdx.x) * DIM_PARAM + j];
            }
            
            //cEntryOld = (currentCol         & currentRowThread) > 0 ? 1 : 0;        
            //cEntryNew = (currentCol_changed & currentRowThread) > 0 ? 1 : 0;
            error_thread += ((cEntryNew - cTruthEntry) * (cEntryNew - cTruthEntry)) -
                            ((cEntryOld - cTruthEntry) * (cEntryOld - cTruthEntry));
        }
    }
    __syncthreads();

    // Reduction across block
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    error_thread = warpReduceSum(error_thread);
    if (lane == 0) shared[wid] = error_thread;
    __syncthreads();
    error_thread = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) error_thread = warpReduceSum(error_thread);
    // Thread with threadIdx.x==0 now has error in error_thread

    // Thread 0 checks if new low has been found and applies if necessary
    if (threadIdx.x == 0) {
        if (error_thread < 0) {
            //B[colToBeChanged] = shared_currentCol_changed;
            for(int i=0; i<DIM_PARAM; i++)
                B[i*width + colToBeChanged] = shared_currentCol_changed[i];
            atomicAdd(global_error, error_thread);
        }  else { // Metropolis–Hastings algorithm
            randomNumber = fast_kiss32(&state) / (double) UINT32_MAX;
            metro = temperature > 0.0f ? fminf(1, expf(-error_thread / temperature)) : 0 ;
            if (randomNumber < metro) {
				for(int i=0; i<DIM_PARAM; i++)
					B[i*width + colToBeChanged] = shared_currentCol_changed[i];
                //B[colToBeChanged] = shared_currentCol_changed;
                atomicAdd(global_error, error_thread);
            }
        }
    }
    __syncthreads();
}

// Start error kernel
__global__ void computeFullError(   uint32_t *A, uint32_t *B, uint32_t *C, 
                                    int width, int height, int *distance_test) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lineSum;
    int truthEntry;
    int intId;
    int intLane;
    int error_thread;
    //__shared__ volatile int shared_distance[THREADSPERBLOCK];
    //shared_distance[threadIdx.x] = 0;
    __shared__ int reductionArray[32];
    
    
    if (tid < width) {
        error_thread = 0;
        for (int j = 0; j < height; j++) {
            lineSum = 0;
            for (int i = 0; i < DIM_PARAM; i++) {
				if(lineSum < 0.5)
					lineSum += (A[j*DIM_PARAM + i]) * (B[(i*width) + tid]);
            }
            //lineSum = (A[j] & B[tid]) > 0 ? 1 : 0;
            //intId = (tid * height + j) / 32;
            //intLane = (tid * height + j) % 32;
            
            //truthEntry = (C[intId] >> 32 - intLane - 1) & 1; 
            //if(A[j*DIM_PARAM + i] * B[(i*width) + tid] < 0)
            //    printf("BALBAL \t");
            truthEntry = C[j*width + tid];
            
            error_thread += ((lineSum - truthEntry) * (lineSum - truthEntry));
        }
        __syncthreads();
        //printf("Own error: %d\n", error_thread);
        
        // Reduction across block
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;
        error_thread = warpReduceSum(error_thread);
        if (lane == 0) reductionArray[wid] = error_thread;
        __syncthreads();
        error_thread = (threadIdx.x < blockDim.x / warpSize) ? reductionArray[lane] : 0;
        if (wid == 0) error_thread = warpReduceSum(error_thread);
        // Thread with threadIdx.x==0 now has error in error_thread

        if (threadIdx.x == 0)
            atomicAdd(distance_test, error_thread);
        __syncthreads();
        
    }
}

// Each thread one entry of a row
__global__ void matrixMultiply( uint32_t *A, uint32_t *B, uint32_t *C, int width, int height) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < width)
        for (int i = 0; i < height; i++)
            C[i * width + tid] = (A[i] & B[tid]) > 0 ? 1 : 0;
}

__global__ void matrixMultiplyInt(  int * A0, int * B0, uint32_t * C0, 
                                    int m, int k, int n) {
    __shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sum = 0;

    for(int t = 0; t < (n - 1) / TILE_WIDTH + 1; t++) {
        if(row < m && t * TILE_WIDTH + tx < n)
            ds_A[ty][tx] = roundf(A0[row * n + t * TILE_WIDTH + tx]);
        else
            ds_A[ty][tx] = 0.0;
        if(t * TILE_WIDTH + ty < n && col < k)
            ds_B[ty][tx] = roundf(B0[(t * TILE_WIDTH + ty) * k + col]);
        else
            ds_B[ty][tx] = 0.0;
        __syncthreads();
        for(int i = 0; i < TILE_WIDTH; i++){
            sum += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    }
    if(row < m && col < k)
        C0[col + row * k] = min(1, sum);

}

void readInputFileData( uint32_t **C0, uint32_t **d_C0, 
                    int *width, int *height, 
                    double *density, string filename) {
    int x, y;
    int nonzeroelements = 0;
    //int intID;
    //int intLane;
    ifstream infile;
    string linestring;
    string field;

    // First line: #height,#width,#non-zero-elements
    infile.open(filename);
    getline(infile, linestring);
    stringstream sep(linestring);
    getline(sep, field, ',');
    (*height) = stoi(field, nullptr);
    getline(sep, field, '\n');
    (*width) = stoi(field, nullptr);
    
    // Malloc for C0 and d_C0
    int sizeC = ((*width) * (*height));
    //int sizeC = (int) ceil((*width) * (*height) / (double) 32.0);
    (*C0) = (uint32_t *) malloc(sizeof(uint32_t) * sizeC);
    cudaMalloc((void **) d_C0, sizeof(uint32_t) * sizeC);                                       CUERR
    
    // Set all entries 0
    for (int i = 0; i < sizeC; i++)
        (*C0)[i] = 0;

    // Read rest of file
    while (getline(infile, linestring)) {
        stringstream sep1(linestring);
        string fieldtemp;
        getline(sep1, fieldtemp, ',');
        y = stoi(fieldtemp, nullptr);
        getline(sep1, fieldtemp, ',');
        x = stoi(fieldtemp, nullptr);
        //intID = (x * (*height) + y) / 32;
        //intLane = (x * (*height) + y) % 32;
        //(*C0)[intID] |= 1 << 32 - intLane - 1;
        (*C0)[y * (*width) + x] = 1;
        nonzeroelements++;
    }
    
    (*density) = (double) nonzeroelements / ((*width) * (*height));

    cudaMemcpy((*d_C0), (*C0), sizeof(uint32_t) * sizeC, cudaMemcpyHostToDevice);               CUERR
       
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("READING OF .DATA FILE COMPLETE\n");
    printf("Read height: %i\nRead width: %i\nNon-zero elements: %i\nDensity: %f\n",
           (*height), (*width), nonzeroelements, (*density));
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}


void readInputFileTSV( uint32_t **C0, uint32_t **d_C0, 
                    int *width, int *height, 
                    double *density, string filename) {


    int intID;
    int intLane;
    int nonzeroelements;
    ifstream input(filename);
    char const row_delim = '\n';
    char const field_delim = '\t';
    vector<uint32_t> x_values;
    vector<uint32_t> y_values;
    int x_counter = 0;
    int y_counter = 0;
    
    // Read file
    for (string row; getline(input, row, row_delim); ) {
        x_counter = 0;
        istringstream ss(row);
        for (string field; getline(ss, field, field_delim); ) {
            if (stoi(field) == 1) {
                x_values.push_back(x_counter);
                y_values.push_back(y_counter);
            }
            x_counter++;
        }
        y_counter++;
    }
    *width = x_counter;
    *height = y_counter;
    nonzeroelements = x_values.size();
    (*density) = (double) nonzeroelements / ((*width) * (*height));
    
    
    // Malloc for C0 and d_C0
    int sizeC = (int) ceil((*width) * (*height));
    //int sizeC = (int) ceil((*width) * (*height) / (double) 32.0);
    (*C0) = (uint32_t *) malloc(sizeof(uint32_t) * sizeC);
    cudaMalloc((void **) d_C0, sizeof(uint32_t) * sizeC);                                       CUERR
    
    // Set all entries 0
    for (int i = 0; i < sizeC; i++)
        (*C0)[i] = 0;

    // Read rest of file
    for (int i = 0; i < x_values.size(); i++) {
        intID = (x_values[i] * (*height) + y_values[i]) / 32;
        intLane = (x_values[i] * (*height) + y_values[i]) % 32;
        //(*C0)[intID] |= 1 << 32 - intLane - 1;
        (*C0)[y_values[i] * (*width) + x_values[i]] = 1;
    }
    cudaMemcpy((*d_C0), (*C0), sizeof(uint32_t) * sizeC, cudaMemcpyHostToDevice);               CUERR
    
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("READING OF DENSE .TSV FILE COMPLETE\n");
    printf("Read height: %i\nRead width: %i\nNon-zero elements: %i\nDensity: %f\n",
           (*height), (*width), nonzeroelements, (*density));
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
bool endsWith(const string& s, const string& suffix) {
    return s.rfind(suffix) == (s.size()-suffix.size());
}

void computeStartError(uint32_t *d_Ab, uint32_t *d_Bb, uint32_t *d_Cb, 
                        int width, int height,
                        int **d_distance_C0_C_start, int *distance_C0_C_start) {
    TIMERSTART(ERRORFIRST)
    
    computeFullError <<< width / THREADSPERBLOCK + 1, THREADSPERBLOCK >>>
                        (d_Ab, d_Bb, d_Cb, width, height, (*d_distance_C0_C_start));                    CUERR
                        
    cudaMemcpy(distance_C0_C_start, (*d_distance_C0_C_start), sizeof(int), cudaMemcpyDeviceToHost);     CUERR

    printf("Starting error between AxB=C and C0: %f \n", 
            (*distance_C0_C_start) / ((double) width * height));
    TIMERSTOP(ERRORFIRST)
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// Only for debugging
void checkDistance(uint32_t *d_Ab, uint32_t *d_Bb, uint32_t *d_C0, int height, int width) {
    int distance_test;
    int *d_distance_test;
    cudaMalloc((void **) &d_distance_test, sizeof(int));                                                CUERR
    distance_test = 0;
    cudaMemcpy(d_distance_test, &distance_test, sizeof(int), cudaMemcpyHostToDevice);                   CUERR
    
    computeFullError <<< width / THREADSPERBLOCK + 1, THREADSPERBLOCK >>>  
                            (d_Ab, d_Bb, d_C0, height, width, d_distance_test);                         CUERR
                                            
    cudaMemcpy(&distance_test, d_distance_test, sizeof(int), cudaMemcpyDeviceToHost);                   CUERR
    printf("Real Error: %f\n", (distance_test/(double)(height*width)));
}

// Initialization of A and B
void initializeFactors( uint32_t **Ab, uint32_t **Bb, 
                        uint32_t **d_Ab, uint32_t **d_Bb, 
                        int width, int height, 
                        float density, fast_kiss_state32_t *state) {

    //(*Ab) = (uint32_t *) malloc(sizeof(uint32_t) * height);
    //(*Bb) = (uint32_t *) malloc(sizeof(uint32_t) * width);
    //cudaMalloc((void **) d_Ab, sizeof(uint32_t) * height);                                              CUERR
    //cudaMalloc((void **) d_Bb, sizeof(uint32_t) * width);                                               CUERR
    (*Ab) = (uint32_t *) malloc(sizeof(uint32_t) * height * DIM_PARAM);
    (*Bb) = (uint32_t *) malloc(sizeof(uint32_t) * width * DIM_PARAM);
    cudaMalloc((void **) d_Ab, sizeof(uint32_t) * height * DIM_PARAM);                                  CUERR
    cudaMalloc((void **) d_Bb, sizeof(uint32_t) * width * DIM_PARAM);                                   CUERR

    // Initialize A and B and copy to device
    bool threshold;
    for (int i = 0; i < height; i++) {
        //(*Ab)[i] = 0;
        #pragma unroll
        for (int j = 0; j < DIM_PARAM; j++) {
            switch(INITIALIZATIONMODE) {
                case 1: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (sqrt(1 - pow(1 - density, 1 / (double) DIM_PARAM)));
                                        break;
                case 2: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (density / (double) 100);
                                        break;
                case 3: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < density;
                                        break;
            }
            //(*Ab)[i] |= threshold ? 1 << (DIM_PARAM - j - 1) : 0 ;
            (*Ab)[i * DIM_PARAM + j] = threshold ? 1 : 0 ;
        }
    }
    for (int i = 0; i < width; i++) {
        //(*Bb)[i] = 0;
        #pragma unroll
        for (int j = 0; j < DIM_PARAM; j++) {
            switch(INITIALIZATIONMODE) {
                case 1: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (sqrt(1 - pow(1 - density, 1 / (double) DIM_PARAM)));
                                        break;
                case 2: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (density / (double) 100);
                                        break;
                case 3: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < density;
                                        break;
            }
            //(*Bb)[i] |= threshold ? 1 << (DIM_PARAM - j - 1) : 0 ;
            (*Bb)[j * width + i] = threshold ? 1 : 0 ;
        }
    }
    
    // copy to device arrays
    //cudaMemcpy((*d_Ab), (*Ab), sizeof(uint32_t) * height, cudaMemcpyHostToDevice);                      CUERR
    //cudaMemcpy((*d_Bb), (*Bb), sizeof(uint32_t) * width, cudaMemcpyHostToDevice);                       CUERR    
    cudaMemcpy((*d_Ab), (*Ab), sizeof(uint32_t) * height * DIM_PARAM, cudaMemcpyHostToDevice);                      CUERR
    cudaMemcpy((*d_Bb), (*Bb), sizeof(uint32_t) * width * DIM_PARAM, cudaMemcpyHostToDevice);                       CUERR

    printf("Initialization of A and B complete\n");
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// Used for debugging and checking for correctness, not optimized
// NOT USED, WRONG
void aftertestGPU(  uint32_t *d_Ab, uint32_t *d_Bb, uint32_t *d_C0b, 
                    int width, int height) {
    TIMERSTART(aftertestGPU)
    uint32_t *d_C_test_GPU;
    uint32_t *C_test_GPU;
    
    int *A, *B, *C0;
    uint32_t *Ab, *Bb, *C0b;
    int *d_A, *d_B;
    float densityA, densityB;
    uint32_t counterDensity;
    A = (int*) malloc(sizeof(int) * DIM_PARAM * height);
    Ab = (uint32_t*) malloc(sizeof(uint32_t) * height);
    B = (int*) malloc(sizeof(int) * width * DIM_PARAM);
    Bb = (uint32_t*) malloc(sizeof(uint32_t) * width);
    C0 = (int*) malloc(sizeof(int) *width * height);
    C0b = (uint32_t*) malloc(sizeof(uint32_t) * ((long long) (height * width) / 32.0 + 1));
    
    cudaMalloc((void**)&d_A, sizeof(int) * DIM_PARAM * height);                                             CUERR
    cudaMalloc((void**)&d_B, sizeof(int) * width * DIM_PARAM);                                              CUERR
    
    cudaMemcpy(Ab, d_Ab, sizeof(uint32_t) * height, cudaMemcpyDeviceToHost);                              CUERR
    cudaMemcpy(Bb, d_Bb, sizeof(uint32_t) * width, cudaMemcpyDeviceToHost);                               CUERR
    cudaMemcpy(C0b, d_C0b, sizeof(uint32_t) * ((long long)(height*width) / 32.0 + 1),
                    cudaMemcpyDeviceToHost);                                                            CUERR

    counterDensity = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < DIM_PARAM; j++){
            A[i * DIM_PARAM + j] = (Ab[i] >> DIM_PARAM - j - 1) & 1;
            if (A[i*DIM_PARAM + j]) counterDensity++;
        }
    }
    densityA = counterDensity / (double) (height*DIM_PARAM);
    
    counterDensity = 0;
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < DIM_PARAM; j++) {
            B[j * width + i] = (Bb[i] >> DIM_PARAM - j - 1) & 1;
            if (B[j * width + i]) counterDensity++;
        }
    }
    densityB = counterDensity / (double) (DIM_PARAM * width);

    int intId;
    int intLane;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
             intId = (i * height + j) / 32;
             intLane = (i * height + j) % 32;
             C0[j * width + i] = (C0b[intId] >> 32 - intLane - 1) & 1;
        }
    }
    
    cudaMemcpy(d_A, A, sizeof(int)*height*DIM_PARAM, cudaMemcpyHostToDevice);                           CUERR
    cudaMemcpy(d_B, B, sizeof(int)*width*DIM_PARAM, cudaMemcpyHostToDevice);                            CUERR   
    
    // Doing a check two times: once with A,B and once with Ab,Bb just to make sure
    // First check
    C_test_GPU = (uint32_t *) malloc(sizeof(uint32_t) * width * height);
    cudaMalloc((void **) &d_C_test_GPU, sizeof(uint32_t) * height * width);                             CUERR
    
    matrixMultiply <<< width / THREADSPERBLOCK + 1, THREADSPERBLOCK >>> 
                        (d_Ab, d_Bb, d_C_test_GPU, width, height);                                      CUERR
                        
    cudaMemcpy(C_test_GPU, d_C_test_GPU, sizeof(uint32_t) * height * width, 
                    cudaMemcpyDeviceToHost);                                                            CUERR
    
    int error_test_GPU = 0;
    for (int i = 0; i < height * width; i++)
        error_test_GPU += (((C0[i] - C_test_GPU[i]) * (C0[i] - C_test_GPU[i])));

    // Second check
    dim3 dimGrid((width - 1) / TILE_WIDTH + 1, (height - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    
    matrixMultiplyInt <<< dimGrid, dimBlock >>> 
                            (d_A, d_B, d_C_test_GPU, height, width, DIM_PARAM);                         CUERR

    cudaMemcpy(C_test_GPU, d_C_test_GPU, sizeof(int) * height * width, cudaMemcpyDeviceToHost);         CUERR
    int error_test_GPU_2 = 0;
    for (int i = 0; i < height * width; i++)
        error_test_GPU_2 += (((C0[i] - C_test_GPU[i]) * (C0[i] - C_test_GPU[i])));
    
    TIMERSTOP(aftertestGPU)
    printf("Aftertest error between C0 and C on GPU (bitwise): %i\n", error_test_GPU);
    printf("Aftertest error between C0 and C on GPU (float): %i\n", error_test_GPU_2);
    printf("Density A: %f, Density B: %f\n", densityA, densityB);
    
    //writeToFiles(A, B, width, height);
}

// Write result matrix in file
void writeToFiles(uint32_t* d_A, uint32_t* d_B, int width, int height){
    uint32_t *A, *B;
    A = (uint32_t*) malloc(sizeof(uint32_t) * DIM_PARAM * height);
    B = (int*) malloc(sizeof(int) * width * DIM_PARAM);
    
    cudaMemcpy(A, d_A, sizeof(uint32_t) * height * DIM_PARAM, cudaMemcpyDeviceToHost);                              CUERR
    cudaMemcpy(B, d_B, sizeof(uint32_t) * width * DIM_PARAM, cudaMemcpyDeviceToHost);                               CUERR

    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%X", timeinfo);
    string str = buffer;
    
    string a = string("A_") + buffer + string(".txt");
    string b = string("B_") + buffer + string(".txt");
    
    ofstream myfile(a);
    if (myfile.is_open()){
        myfile << height << "," << DIM_PARAM << "\n";
        for (int i = 0; i < height; i++){
            for (int j = 0; j < DIM_PARAM; j++){
                myfile << A[i * DIM_PARAM + j] << ((j != DIM_PARAM - 1) ? "," : "");
            }
            myfile << "\n";
        }
        myfile.close();
    }
    
    ofstream myfile2(b);
    if(myfile2.is_open()){
        myfile2 << DIM_PARAM << "," << width << "\n";
        for (int i = 0; i<DIM_PARAM; i++){
            for (int j = 0; j < width; j++){
                myfile2 << B[i * width + j] << ((j != width - 1) ? "," : "");
            }
            myfile2 << "\n";
        }
        myfile2.close();
    }   
    cout << "Writing to files \"" << a << "\" and \"" << b << "\" complete" << endl;
}

// CPU computation
void CPUcomputation(uint32_t *Ab, uint32_t *Bb, uint32_t *C0, 
                    int width, int height, 
                    int startDistance, uint32_t seed, int updateStep,
                    float threshold, int rowsAtOnce) {
                        
    int *hDistance = &startDistance;
    fast_kiss_state32_t state;
    state = get_initial_fast_kiss_state32(seed);
    int toBeChanged;
    int iterations = 0;
    TIMERSTART(CPUcomputation)
    printf("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("- - - - Starting CPU opimization, showing error every %i steps - - - - -\n", updateStep);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    while (*hDistance > threshold && iterations < CPUITERATIONS) {
        if (iterations % updateStep == 0)
            printf("Current Distance: %i\n", *hDistance);
        
        // Change row
        toBeChanged = ((unsigned int) fast_kiss32(&state)) % height;
        CPUvectorMatrixMultCompareRow(Ab, Bb, C0, width, height, toBeChanged, hDistance, &state, rowsAtOnce);
        
        // Change col
        toBeChanged = ((unsigned int) fast_kiss32(&state)) % width;
        CPUvectorMatrixMultCompareCol(Ab, Bb, C0, width, height, toBeChanged, hDistance, &state, rowsAtOnce);
        iterations++;
    }
    printf("- - - - - - - - -\n");
    printf("End Distance on CPU: %i, Number of Iterations: %i,  Error remaining: %f\n", 
                *hDistance, iterations, *hDistance / (double) (height * width));
    TIMERSTOP(CPUcomputation)
}

void CPUvectorMatrixMultCompareRow( uint32_t *Ab, uint32_t *Bb, 
                                    uint32_t *C0, int width, int height, int startrow,
                                    int *hDistance, fast_kiss_state32_t *state, int rowsAtOnce) {
    int rowToBeChanged = startrow;
    int error;
    int cTruthEntry;
    int cEntryOld;
    int cEntryNew;
    uint32_t currentRow;
    uint32_t currentRow_changed;
    uint32_t randomNumber;

    // Change multiple lines from A
    for (int l = 0; l < rowsAtOnce; l++) {
        rowToBeChanged = (startrow + l) % height;
        currentRow = Ab[rowToBeChanged];
        currentRow_changed = currentRow;

        randomNumber = fast_kiss32(state);
        for (int i = 0; i < DIM_PARAM; i++)
            currentRow_changed ^= (randomNumber >> i) & 11 ?(0 << i) : (1 << i);
        
        error = 0;

#pragma omp parallel for private(cEntryOld, cEntryNew, cTruthEntry) reduction(+: error)
        for (int tid = 0; tid < width; tid++) {
            uint32_t currentCol = Bb[tid];
            int intId = (tid * height + rowToBeChanged) / 32;
            int intLane = (tid * height + rowToBeChanged) % 32;
            cTruthEntry = (C0[intId] >> 32 - intLane - 1) & 1;

            cEntryOld = (currentRow         & currentCol) > 0 ? 1 : 0;
            cEntryNew = (currentRow_changed & currentCol) > 0 ? 1 : 0;
            error += ((cEntryNew - cTruthEntry) * (cEntryNew - cTruthEntry)) -
                     ((cEntryOld - cTruthEntry) * (cEntryOld - cTruthEntry));
        }

        if (error < 0) {
            Ab[rowToBeChanged] = currentRow_changed;
            *hDistance = *hDistance + error;
        } 

    }
}

void CPUvectorMatrixMultCompareCol( uint32_t *Ab, uint32_t *Bb, uint32_t *C0, 
                                    int width, int height, int startcol,
                                    int *hDistance, fast_kiss_state32_t *state, int rowsAtOnce) {
    int colToBeChanged = startcol;
    int error;
    int cTruthEntry;
    int cEntryOld;
    int cEntryNew;
    uint32_t currentCol;
    uint32_t currentCol_changed;
    uint32_t randomNumber;

    // Change multiple cols from B
    for (int l = 0; l < rowsAtOnce; l++) {
        colToBeChanged = (colToBeChanged + l) % width;
        currentCol = Bb[colToBeChanged];
        currentCol_changed = currentCol;

        randomNumber = fast_kiss32(state);
        for (int i = 0; i < DIM_PARAM; i++)
            currentCol_changed ^= (randomNumber >> i) & 11 ? (0 << i) : (1 << i);
        
        error = 0;
        #pragma omp parallel for private(cEntryOld, cEntryNew, cTruthEntry) reduction(+: error)
        for (int tid = 0; tid < height; tid++) {
             uint32_t currentRow = Ab[tid]; 
            int intId = (colToBeChanged * height + tid) / 32;
            int intLane = (colToBeChanged * height + tid) % 32;
            cTruthEntry = (C0[intId] >> 32 - intLane - 1) & 1;

            cEntryOld = (currentCol         & currentRow) > 0 ? 1 : 0;
            cEntryNew = (currentCol_changed & currentRow) > 0 ? 1 : 0;
            error += ((cEntryNew - cTruthEntry) * (cEntryNew - cTruthEntry)) - 
                     ((cEntryOld - cTruthEntry) * (cEntryOld - cTruthEntry));
        }

        if (error < 0) {
            Bb[colToBeChanged] = currentCol_changed;
            *hDistance = *hDistance + error;
        }              
    }
}

// Used for debugging and checking, not optimized
void aftertestCPU(  uint32_t *Ab, uint32_t *Bb, uint32_t *d_Ab, uint32_t *d_Bb, uint32_t *C0b, 
                    int width, int height) {    
    TIMERSTART(aftertestCPU)
    int *A, *B, *C0;
    int *d_A, *d_B;
    uint32_t *C_test_CPU;
    uint32_t *d_C_test_CPU;
    A = (int*)malloc(sizeof(int) * DIM_PARAM * height);
    B = (int*)malloc(sizeof(int) * width * DIM_PARAM);
    C0 = (int*)malloc(sizeof(int) * width * height);
    C_test_CPU = (uint32_t *) malloc(sizeof(uint32_t) * width * height);                CUERR
    cudaMalloc((void**)&d_A, sizeof(int) * DIM_PARAM * height);                         CUERR
    cudaMalloc((void**)&d_B, sizeof(int)*width * DIM_PARAM);                            CUERR
    cudaMalloc((void**) &d_C_test_CPU, sizeof(uint32_t) * height * width);              CUERR

    
    for(int i=0; i<height;i++)
        for(int j=0;j<DIM_PARAM;j++)
            A[i*DIM_PARAM + j] = (Ab[i] >> DIM_PARAM-j-1) & 1;

    for(int i=0;i<width;i++)
        for(int j=0;j<DIM_PARAM;j++)
            B[j*width+i] = (Bb[i] >> DIM_PARAM-j-1) & 1;
        
    int intId;
    int intLane;
    for(int i=0; i<width; i++){
        for(int j=0;j<height;j++){
             intId = (i*height + j) / 32;
             intLane = (i*height + j) % 32;
             C0[j*width + i] = (C0b[intId] >> 32 - intLane - 1) & 1;
        }
    }

    
    cudaMemcpy(d_A, A, sizeof(uint32_t) * height * DIM_PARAM, cudaMemcpyHostToDevice);  CUERR
    cudaMemcpy(d_B, B, sizeof(uint32_t) * width * DIM_PARAM, cudaMemcpyHostToDevice);   CUERR
    cudaMemcpy(d_Ab, Ab, sizeof(uint32_t) * height, cudaMemcpyHostToDevice);            CUERR
    cudaMemcpy(d_Bb, Bb, sizeof(uint32_t) * width, cudaMemcpyHostToDevice);             CUERR
    
    // Doing a check two times: once with A,B and once with Ab,Bb just to make sure
    
    matrixMultiply <<< width / THREADSPERBLOCK + 1, THREADSPERBLOCK >>> 
                        (d_Ab, d_Bb, d_C_test_CPU, width, height);                      CUERR
    
    cudaMemcpy(C_test_CPU, d_C_test_CPU, sizeof(uint32_t) * height * width, 
                    cudaMemcpyDeviceToHost);                                            CUERR
                    
    int distance_test_CPU = 0;
    for (int i = 0; i < height * width; i++)
        distance_test_CPU += ((C0[i] - C_test_CPU[i]) * (C0[i] - C_test_CPU[i]));
    
    dim3 dimGrid((width-1)/TILE_WIDTH+1, (height-1)/TILE_WIDTH+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiplyInt <<< dimGrid, dimBlock >>> 
                (d_A, d_B, d_C_test_CPU, height, width, DIM_PARAM);                     CUERR
    cudaMemcpy(C_test_CPU, d_C_test_CPU, sizeof(int) * height * width, 
                    cudaMemcpyDeviceToHost);                                            CUERR
    int distance_test_CPU_2 = 0;
    for (int i = 0; i < height * width; i++) 
        distance_test_CPU_2 += ((C0[i] - C_test_CPU[i]) * (C0[i] - C_test_CPU[i]));
    
    TIMERSTOP(aftertestCPU)
    printf("Aftertest error between C0 and C on CPU (bitwise): %i\n", distance_test_CPU);
    printf("Aftertest error between C0 and C on CPU (float): %i\n", distance_test_CPU_2);
}

/*
void initializeTextureMemory(uint32_t **C0, int width, int height) {
    // Texture Memory Initialization
    //////////////////////////////////////////////////////////////////////////////////////
    /*
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    CUERR
    cudaArray *cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    CUERR
    cudaMemcpyToArray(cuArray, 0, 0, *C0, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    CUERR
    
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    cudaBindTextureToArray(texRef, cuArray, channelDesc);
    
    uint* d_C0_texture;
    size_t pitch;
    cudaMallocPitch((void**)&d_C0_texture, &pitch, width*sizeof(uint32_t), height);
    cudaMemcpy2D(d_C0_texture, pitch, *C0, width*sizeof(uint32_t), width*sizeof(uint32_t), height, cudaMemcpyHostToDevice);
    
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uint>(); 
    cudaBindTexture2D(NULL, texRef, d_C0_texture, desc, width, height, pitch);
    ///////////////////////////////////////////////////////////////////////////////////////
}*/


