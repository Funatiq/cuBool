// #include <stdio.h>
// #include <stdint.h>
// #include <stdlib.h>
// #include <string.h>
// #include <stddef.h>
// #include <sys/types.h>
// #include <math.h>
// #include <time.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <cuda.h>
// #include <device_functions.h>
// #include <curand_kernel.h>
// #include <float.h>
// #include <cuda_profiler_api.h>
// #include <algorithm>
#include <omp.h>
#include <vector>
#include <iostream>
#include <limits>

#include "helper/config.h"
#include "helper/cuda_helpers.cuh"
#include "helper/rngpu.hpp"
#include "helper/io_and_allocation.hpp"

#include "CuBin_cpu.cuh"
#include "CuBin_gpu.cuh"

#include "CuBin_final.cuh"

using namespace std;

using my_bit_vector_t = uint32_t; // only tested uint32_t
// using uint32_max = std::numeric_limits<uint32_t>::max();

int main(int argc, char **argv) {
    // cudaProfilerStart();
    /* ./a.out  [data]
                [updateStep]
                [lines at once]
                [threshold]
                [gpu iterations]
                [number of changed each line gets without change before aborting]
                [startingTemperature]
                [iterations till temperature is reduced]
                [factor the temperature is lowered]
    */
    if(argc == 1) return 0;
    std::string filename = argv[1];
    int updateStep = argc > 2 ? atoi(argv[2]) : 1000;
    int linesAtOnce = argc > 3 ? atoi(argv[3]) : 4000;
    float threshold = argc > 4 ? atof(argv[4]) : 0;
    int gpuiterations = argc > 5 ? atoi(argv[5]) : 10000;
    int everyLineChanged = argc > 6 ? atoi(argv[6]) : 1000;
    float temperature = argc > 7 ?  atof(argv[7]) : 0;
    int iterationsTillReduced = argc > 8 && atoi(argv[8]) > 0 ? atoi(argv[8]) : std::numeric_limits<int>::max();
    float tempFactor = argc > 9 ? atof(argv[9]) : 0.99;
    //int seed = argc > 10 ? atoi(argv[10]) : 41;
    uint32_t seed = (unsigned long)time(NULL) % UINT32_MAX;
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);
    
    // Discard first 100000 entries of PRNG
    for (int i = 0; i < 100000; i++)
        fast_kiss32(&state);
    
    // Read file and save matrix in C0 and d_C0
    my_bit_vector_t *C0b, *d_C0b;
    int width, height;
    float density;
    
    // COO coordinates
    string ending = "data";
    if (endsWith(filename, ending)) {
        readInputFileData(&C0b, &d_C0b, &width, &height, &density, filename);
    } else if (filename.compare("test") == 0) {
        width = 5000;
        height = 5000;
        generate_random_matrix(&C0b, &d_C0b, width, height, &density);
    } else {
        printf("Wrong data file\n");
        return 0;
    }
        
    // Initialize Texture Memory with C0b
    // initializeTextureMemory(&C0b, width, height);

    // Initialize Ab, Bb, d_Ab, d_Bb, all bitwise used matrices
    my_bit_vector_t *Ab, *Bb;
    my_bit_vector_t *d_Ab, *d_Bb;
    initializeFactors(&Ab, &Bb, &d_Ab, &d_Bb, width, height, density, &state);
    // A and B now initialized on device and host

    // Calculate original error, save it two times each one for GPU, one for CPU
    int error_C0_C_start = 0;
    int error_C0_C       = 0;
    int *d_error_C0_C_start, *d_error_C0_C;
    cudaMalloc((void **) &d_error_C0_C, sizeof(int));                                                           CUERR
    cudaMalloc((void **) &d_error_C0_C_start, sizeof(int));                                                     CUERR
    cudaMemcpy(d_error_C0_C_start, &error_C0_C_start, sizeof(int), cudaMemcpyHostToDevice);                     CUERR
    
    computeStartError(d_Ab, d_Bb, d_C0b, width, height, &d_error_C0_C_start, &error_C0_C_start);
    
    cudaMemcpy(d_error_C0_C, d_error_C0_C_start, sizeof(int), cudaMemcpyDeviceToDevice);
    error_C0_C = error_C0_C_start;
    // Now the starting errors are in stored in 4 values
    // error_C0_C_start, error_C0_C on CPU and GPU

    // MAIN PART
    // on GPU
    int iterations = 0;
    int lineToBeChanged;
    int iterationsNoImp = 0;
    int error_C0_C_before = error_C0_C;
    uint32_t gpuSeed;
    threshold *= (width*height);
    #ifdef PERF
    vector<double> errorVector; // vector for error measurement
    vector<int> impVector;
    #endif
    
    // Every line and row changed x times before aborting because of no improvement
    int maxIterationsNoImp = (std::max(width,height) / linesAtOnce + 1) * everyLineChanged; 
    printf("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("- - - - Starting %i GPU iterations, showing error every %i steps - - - -\n", gpuiterations, updateStep);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    TIMERSTART(GPUKERNELLOOP)
    while (
        // threshold < error_C0_C
        // && 
        iterations < gpuiterations
        // && 
        // iterationsNoImp < maxIterationsNoImp
        )
    {
        iterations++;
        if (iterations % iterationsTillReduced == 0)
            temperature *= tempFactor;
        
        // Pull error from GPU to show it
        if (iterations % updateStep == 0) {
            // #ifndef PERF
            printf("Current error: %f\n", error_C0_C / (double) (width * height));
            // #endif
            // For debugging
            // checkDistance(d_Ab, d_Bb, d_C0b, height, width);
        }

        // Change col
        lineToBeChanged = fast_kiss32(&state) % width;
        gpuSeed = ((fast_kiss32(&state) + iterations) % UINT32_MAX);
        vectorMatrixMultCompareCol <<< min(linesAtOnce, width), THREADSPERBLOCK >>>
                                        (d_Ab, d_Bb, d_C0b, width, height, lineToBeChanged, d_error_C0_C, 
                                         gpuSeed, temperature);        CUERR
        cudaDeviceSynchronize();                                                                                CUERR

        // Change row
        lineToBeChanged = fast_kiss32(&state) % height;
        gpuSeed = ((fast_kiss32(&state) + iterations) % UINT32_MAX);
        vectorMatrixMultCompareRow <<< min(linesAtOnce, height), THREADSPERBLOCK >>>
                                        (d_Ab, d_Bb, d_C0b, width, height, lineToBeChanged, d_error_C0_C, 
                                         gpuSeed, temperature);        CUERR
        cudaDeviceSynchronize();                                                                                CUERR
        
        cudaMemcpy(&error_C0_C, d_error_C0_C, sizeof(int), cudaMemcpyDeviceToHost);                             CUERR
        
        // To check how many iterations are needed for improvement
        if (error_C0_C_before - error_C0_C == 0) {
            iterationsNoImp++;
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
    }

    if (!(threshold < error_C0_C))
        printf("error below threshold\n");
    if (!(iterations < gpuiterations))
        printf("reached iteration limit\n");
    if (!(iterationsNoImp < maxIterationsNoImp)) 
        printf("reached limit of iterations without improvement\n");

    // Pull final error from GPU
    cudaMemcpy(&error_C0_C, d_error_C0_C, sizeof(int), cudaMemcpyDeviceToHost);                                 CUERR
    printf("- - - - - - - - -\n");
    printf("Final Error on GPU: %f, %i wrong entries\n", error_C0_C / (double) (height * width), error_C0_C);
    TIMERSTOP(GPUKERNELLOOP)
    
    // Aftertest GPU
    #ifdef TEST
    aftertestGPU(d_Ab, d_Bb, d_C0b, width, height);
    #endif
    
    // Write result matrices to files
    #ifndef PERF
    writeToFiles(d_Ab, d_Bb, width, height);
    #endif
    
    // Output CSV files for plotting
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
    CPUcomputation(Ab, Bb, C0b, width, height, error_C0_C_start, 42, updateStep, threshold, linesAtOnce);
    
    // Aftertest CPU
    aftertestCPU(Ab, Bb, d_Ab, d_Bb, C0b, width, height);
    #endif

    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");

    // Cleaning up
    // cudaProfilerStop();
    cudaDeviceReset();
    free(Ab);
    free(Bb);
    free(C0b);
    cudaFree(d_Ab);
    cudaFree(d_Bb);
    cudaFree(d_C0b);
    cudaFree(d_error_C0_C);
    cudaFree(d_error_C0_C_start);
    return 0;
}



// Start error kernel
template<typename element_t>
__global__ void computeFullError(   element_t *A, element_t *B, element_t *C, 
                                    int width, int height, int *distance_test) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //__shared__ volatile int shared_distance[THREADSPERBLOCK];
    //shared_distance[threadIdx.x] = 0;
    __shared__ int reductionArray[32];
    
    int error_thread = 0;
    if (tid < width) {
        for (int j = 0; j < height; j++) {
            int lineSum = (A[j] & B[tid]) ? 1 : 0;
            int intId = (tid * height + j) / 32;
            int intLane = (tid * height + j) % 32;
            int truthEntry = (C[intId] >> 32 - intLane - 1) & 1; 
            error_thread += lineSum ^ truthEntry;
        }
    }
    __syncthreads();
    
    int error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

   if (threadIdx.x == 0)
        atomicAdd(distance_test, error_block);
}


template<typename bit_vector_t>
void computeStartError(bit_vector_t *d_Ab, bit_vector_t *d_Bb, bit_vector_t *d_Cb, 
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
template<typename element_t>
void checkDistance(element_t *d_Ab, element_t *d_Bb, element_t *d_C0, int height, int width) {
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

