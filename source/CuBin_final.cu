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
// #include "helper/compute_error.cuh"

#include "CuBin_cpu.cuh"
#include "CuBin_gpu.cuh"

// #include "CuBin_final.cuh"

using namespace std;

using my_bit_vector_t = uint32_t; // only tested uint32_t
// using uint32_max = std::numeric_limits<uint32_t>::max();


template<typename bit_vector_t>
void computeError(const bit_vector_t *d_Ab, const bit_vector_t *d_Bb, const bit_vector_t *d_Cb, 
                  const int height, const int width, const int padded_width,
                  int *&d_distance_C0_C, int &distance_C0_C)
{

    cudaMalloc(&d_distance_C0_C, sizeof(int));                                                       CUERR
    cudaMemset(d_distance_C0_C, 0, sizeof(int));                                                       CUERR
    // cudaMemcpy(d_distance_C0_C, &distance_C0_C, sizeof(int), cudaMemcpyHostToDevice);                     CUERR

    computeFullDistance <<< SDIV(width, THREADSPERBLOCK), THREADSPERBLOCK >>>
                        (d_Ab, d_Bb, d_Cb, height, width, padded_width, d_distance_C0_C);                    CUERR
    
    cudaMemcpy(&distance_C0_C, d_distance_C0_C, sizeof(int), cudaMemcpyDeviceToHost);     CUERR
}

// Only for debugging
template<typename bit_vector_t>
void checkDistance(const bit_vector_t *d_Ab, const bit_vector_t *d_Bb, const bit_vector_t *d_C0b,
                   const int height, const int width, const int padded_width)
{
    int distance_test;
    int *d_distance_test;
    // distance_test = 0;

    computeError(d_Ab, d_Bb, d_C0b, height, width, padded_width, d_distance_test, distance_test);

    cudaFree(d_distance_test); CUERR

    printf("Real Error: \t%f\n",
           distance_test / ((double) height * width));
}


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
    linesAtOnce = linesAtOnce / 32 * 32;
    float threshold = argc > 4 ? atof(argv[4]) : 0;
    int gpuiterations = argc > 5 ? atoi(argv[5]) : 10000;
    int everyLineChanged = argc > 6 ? atoi(argv[6]) : 1000;
    float temperature = argc > 7 ?  atof(argv[7]) : 0;
    int iterationsTillReduced = argc > 8 && atoi(argv[8]) > 0 ? atoi(argv[8]) : std::numeric_limits<int>::max();
    float tempFactor = argc > 9 ? atof(argv[9]) : 0.9f;
    //int seed = argc > 10 ? atoi(argv[10]) : 41;
    // uint32_t seed = (unsigned long)time(NULL) % UINT32_MAX;
    uint32_t seed = 47;
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);
    
    // Discard first 100000 entries of PRNG
    for (int i = 0; i < 100000; i++)
        fast_kiss32(&state);
    
    // Read file and save matrix in C0 and d_C0
    // my_bit_vector_t *A0b, *B0b;
    my_bit_vector_t *C0b;
    int height, width;
    float density;
    
    // COO coordinates
    vector<my_bit_vector_t> A0_vec, B0_vec, C0_vec;
    string ending = "data";
    if (endsWith(filename, ending)) {
        readInputFileData(filename, C0_vec, height, width, density);
    } else if (filename.compare("test") == 0) {
        height = 5000;
        width = 5000;
        // height = 5*1024;
        // width = 5*1024;
        generate_random_matrix(height, width, 3, A0_vec, B0_vec, C0_vec, density);
        // free(A0b);
        // free(B0b);
    } else {
        printf("Wrong data file\n");
        return 0;
    }
    C0b = C0_vec.data();

    // int height_padded = SDIV(height, 32) * 32;
    int height_padded_32 = SDIV(height, 32);
    int width_padded = SDIV(width, 32) * 32;
    // int sizeC = height_padded_32 * width;
    int sizeC_padded = height_padded_32 * width_padded;
    // cout << "height: " << height << " width: " << width << endl;
    // cout << "padded height: " << height_padded << " padded width: " << width_padded << endl;

    // Initialize Ab, Bb, d_Ab, d_Bb, all bitwise used matrices
    my_bit_vector_t *Ab, *Bb;
    // Ab = A0b;
    // Bb = B0b;
    // initializeFactors(Ab, Bb, height, width, density, &state);

    vector<my_bit_vector_t> A_vec, B_vec;
    initializeFactors(A_vec, B_vec, height, width, density, &state);

    Ab = A_vec.data();
    Bb = B_vec.data();

    my_bit_vector_t *d_Ab, *d_Bb;
    cudaMalloc((void **) &d_Ab, sizeof(my_bit_vector_t) * height);                                              CUERR
    cudaMalloc((void **) &d_Bb, sizeof(my_bit_vector_t) * width);                                               CUERR
    //
    // cudaMemset(d_Ab, 0, sizeof(my_bit_vector_t) * height_padded);                                     CUERR
    // cudaMemset(d_Bb, 0, sizeof(my_bit_vector_t) * width_padded);                                     CUERR
    // copy to device arrays
    cudaMemcpy(d_Ab, Ab, sizeof(my_bit_vector_t) * height, cudaMemcpyHostToDevice);                      CUERR
    cudaMemcpy(d_Bb, Bb, sizeof(my_bit_vector_t) * width, cudaMemcpyHostToDevice);                       CUERR
    // A and B now initialized on device and host

    my_bit_vector_t *d_C0b;
    cudaMalloc(&d_C0b, sizeof(my_bit_vector_t) * sizeC_padded);                                       CUERR
    // cudaMemset(d_C0b, 0, sizeof(my_bit_vector_t) * sizeC_padded);                                     CUERR
    // cudaMemcpy(d_C0b, C0b, sizeof(my_bit_vector_t) * sizeC, cudaMemcpyHostToDevice);               CUERR
    cudaMemcpy2D(d_C0b, sizeof(my_bit_vector_t) * width_padded,
                 C0b, sizeof(my_bit_vector_t) * width,
                 sizeof(my_bit_vector_t) * width,
                 height_padded_32,
                 cudaMemcpyHostToDevice);               CUERR

    // Calculate original error
    int error_C0_C_start;
    int *d_error_C0_C;
    // TIMERSTART(ERRORFIRST)
    computeError(d_Ab, d_Bb, d_C0b, height, width, width_padded, d_error_C0_C, error_C0_C_start);
    // TIMERSTOP(ERRORFIRST)
    int error_C0_C = error_C0_C_start;
    // Now the starting error is stored in 3 values
    // error_C0_C_start, error_C0_C on CPU and d_error_C0_C on GPU

    printf("Starting error between AxB=C and C0: %f \n", 
           error_C0_C_start / ((double) height * width));
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");

    checkDistance(d_Ab, d_Bb, d_C0b, height, width, width_padded);

    // MAIN PART
    // on GPU
    int iterations = 0;
    int iterationsNoImp = 0;
    int error_C0_C_before = error_C0_C;
    threshold *= (height*width);
    #ifdef PERF
    vector<double> errorVector; // vector for error measurement
    vector<int> impVector;
    #endif
    
    // Every line and row changed x times before aborting because of no improvement
    int maxIterationsNoImp = (std::max(height,width) / linesAtOnce + 1) * everyLineChanged; 
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("- - - - Starting %i GPU iterations, changing %i lines each time - - - - - -\n",
           gpuiterations, linesAtOnce);
    printf("- - - - Showing error every %i steps  - - - - - - - - - - - - - - - - - - -\n",
           updateStep);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");

    TIMERSTART(GPUKERNELLOOP)
    while (
        threshold < error_C0_C
        && 
        iterations < gpuiterations
        && 
        iterationsNoImp < maxIterationsNoImp
        )
    {
        iterations++;
        if (iterations % iterationsTillReduced == 0) {
            temperature *= tempFactor;
            printf("Temperature: %f\t", temperature);
        }
        
        // Pull error from GPU to show it
        if (iterations % updateStep == 0) {
            // #ifndef PERF
            printf("Current error: \t%f\n", error_C0_C / (double) (height * width));
            // #endif
            // For debugging
            // if (error_C0_C < 0)
                // checkDistance(d_Ab, d_Bb, d_C0b, height, width, width_padded);
        }

        // Change row
        int lineToBeChanged = (fast_kiss32(&state) % height) / WARPSPERBLOCK * WARPSPERBLOCK;
        uint32_t gpuSeed = ((fast_kiss32(&state) + iterations) % UINT32_MAX);
        vectorMatrixMultCompareRowWarpShared <<< SDIV(min(linesAtOnce, height), WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                                    (d_Ab, d_Bb, d_C0b,
                                     height, width, width_padded,
                                     lineToBeChanged,
                                     d_error_C0_C, 
                                     gpuSeed, temperature);        CUERR
        cudaDeviceSynchronize();                                                                                CUERR

        // Change col
        lineToBeChanged = (fast_kiss32(&state) % width) / WARPSPERBLOCK * WARPSPERBLOCK;
        gpuSeed = ((fast_kiss32(&state) + iterations) % UINT32_MAX);
        vectorMatrixMultCompareColWarpShared <<< SDIV(min(linesAtOnce, width), WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                                    (d_Ab, d_Bb, d_C0b,
                                     height, width, width_padded,
                                     lineToBeChanged,
                                     d_error_C0_C, 
                                     gpuSeed,
                                     temperature);        CUERR
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
        errorVector.push_back(error_C0_C / (double) (height * width));
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
    printf("Final Error on GPU: \t\t%f, %i wrong entries\n", error_C0_C / (double) (height * width), error_C0_C);
    TIMERSTOP(GPUKERNELLOOP)
    
    // Aftertest GPU
    #ifdef TEST
    aftertestGPU(d_Ab, d_Bb, d_C0b, height, width, width_padded);
    #endif
    
    // Write result matrices to files
    #ifndef PERF
    my_bit_vector_t *A_out = (my_bit_vector_t*) malloc(sizeof(my_bit_vector_t) * height);
    my_bit_vector_t *B_out = (my_bit_vector_t*) malloc(sizeof(my_bit_vector_t) * width);
    
    cudaMemcpy(A_out, d_Ab, sizeof(my_bit_vector_t) * height, cudaMemcpyDeviceToHost);                              CUERR
    cudaMemcpy(B_out, d_Bb, sizeof(my_bit_vector_t) * width, cudaMemcpyDeviceToHost);                               CUERR
    
    writeToFiles(A_out, B_out, height, width);

    free(A_out);
    free(B_out);
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

    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");

    // Cleaning up
    // cudaProfilerStop();
    cudaDeviceReset();
    // free(Ab);
    // free(Bb);
    // free(C0b);
    cudaFree(d_Ab);
    cudaFree(d_Bb);
    cudaFree(d_C0b);
    cudaFree(d_error_C0_C);
    // cudaFree(d_error_C0_C_start);
    return 0;
}

