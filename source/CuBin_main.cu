#include <vector>
#include <iostream>
#include <limits>

#include "helper/config.h"
#include "helper/rngpu.hpp"
#include "helper/io_and_allocation.hpp"

#include "helper/args_parser.h"

#include "CuBin_gpu.cuh"

using namespace std;

using my_bit_vector_t = uint32_t; // only tested uint32_t

int main(int argc, char **argv) {
    mc::args_parser args{argc, argv};

    auto filename = std::string{};
    if(args.non_prefixed_count() > 0) {
        filename = args.non_prefixed(0);
    } else {
        cerr << "No input file provided. Abort." << endl;
        return 1;
    }

    CuBin::CuBin_config config;
    config.verbosity = args.get<size_t>({"v","verbosity"}, config.verbosity);
    config.linesAtOnce = args.get<size_t>({"l","lines"}, config.linesAtOnce);
    config.maxIterations = args.get<size_t>({"i","iterations"}, config.maxIterations);
    config.distanceThreshold = args.get<int>({"e","d","threshold"}, config.distanceThreshold);
    config.distanceShowEvery = args.get<size_t>({"s","show"}, config.distanceShowEvery);
    config.tempStart = args.get<float>({"t","temp"}, config.tempStart);
    config.tempFactor = args.get<float>({"f","factor"}, config.tempFactor);
    config.tempReduceEvery = args.get<size_t>({"r","reduce"}, config.tempReduceEvery);
    config.seed = args.get<uint32_t>({"seed"}, config.seed);
    config.loadBalance = !args.contains({"b","balanceoff"});

     cout << "verbosity " << config.verbosity << "\n"
        << "linesAtOnce " << config.linesAtOnce << "\n"
        << "maxIterations " << config.maxIterations << "\n"
        << "distanceThreshold " << config.distanceThreshold << "\n"
        << "distanceShowEvery " << config.distanceShowEvery << "\n"
        << "tempStart " << config.tempStart << "\n"
        << "tempFactor " << config.tempFactor << "\n"
        << "tempReduceEvery " << config.tempReduceEvery << "\n"
        << "seed " << config.seed << "\n"
        << "loadBalance " << config.loadBalance << endl;

    // // uint32_t seed = (unsigned long)time(NULL) % UINT32_MAX;
    // uint32_t seed = 46;
        
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);

    // Discard first 100000 entries of PRNG
    for (int i = 0; i < 100000; i++)
        fast_kiss32(&state);
    
    int height, width;
    float density;
    
    vector<my_bit_vector_t> A0_vec, B0_vec, C0_vec;
    string ending = "data";
    if (endsWith(filename, ending)) {
        // Read file and save matrix in C0
        // COO coordinates
        readInputFileData(filename, C0_vec, height, width, density);
    } else if (filename.compare("test") == 0) {
        height = 5000;
        width = 5000;
        // height = 5*1024;
        // width = 5*1024;
        generate_random_matrix(height, width, 3, A0_vec, B0_vec, C0_vec, density);
    } else {
        printf("Wrong data file\n");
        return 0;
    }

    // Initialize Ab, Bb, d_Ab, d_Bb, all bitwise used matrices
    vector<my_bit_vector_t> A_vec, B_vec;
    initializeFactors(A_vec, B_vec, height, width, density, &state);

    auto cubin = CuBin(A_vec, B_vec, C0_vec);
    int distance = cubin.getDistance();
    cout << "Start distance: " << (float) distance / (height*width)
         << " = " << distance << " elements" << endl;
    TIMERSTART(GPUKERNELLOOP)
    cubin.run(config);
    TIMERSTOP(GPUKERNELLOOP)
    cubin.verifyDistance();
    cubin.clear();

    return 0;
}

