#include <omp.h>
#include <vector>
#include <iostream>
#include <limits>

#include "helper/config.h"
#include "helper/rngpu.hpp"
#include "helper/io_and_allocation.hpp"

#include "CuBin_gpu.cuh"

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
    linesAtOnce = linesAtOnce;
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

    // int height_padded = SDIV(height, 32) * 32;
    // int height_padded_32 = SDIV(height, 32);

    // Initialize Ab, Bb, d_Ab, d_Bb, all bitwise used matrices
    // my_bit_vector_t *Ab, *Bb;
    // Ab = A0b;
    // Bb = B0b;

    vector<my_bit_vector_t> A_vec, B_vec;
    initializeFactors(A_vec, B_vec, height, width, density, &state);

    // vector<my_bit_vector_t> A_vec(Ab, Ab + height);
    // vector<my_bit_vector_t> B_vec(Bb, Bb + width);
    // vector<my_bit_vector_t> C0_vec(C0b, C0b + height_padded_32 * width);
    auto cubin = CuBin(A_vec, B_vec, C0_vec);
    int distance = cubin.getDistance();
    cout << "Start distance: " << (float) distance / (height*width)
         << " = " << distance << " elements" << endl;
    CuBin::CuBin_config config;
    config.verbosity = 1;
    config.linesAtOnce = linesAtOnce;
    config.maxIterations = gpuiterations;
    config.distanceThreshold = threshold;
    config.distanceShowEvery = updateStep;
    config.seed = 42;
    TIMERSTART(GPUKERNELLOOP)
    cubin.run(config);
    TIMERSTOP(GPUKERNELLOOP)
    cubin.verifyDistance();
    cubin.clear();


    return 0;
}

