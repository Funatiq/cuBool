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

    CuBin<my_bit_vector_t>::CuBin_config config;
    config.verbosity = args.get<size_t>({"v","verbosity"}, config.verbosity);
    uint8_t factorDim = args.get<uint8_t>({"k","dim","dimension","factordim"}, 20);
    config.linesAtOnce = args.get<size_t>({"l","lines","linesperkernel"}, config.linesAtOnce);
    config.maxIterations = args.get<size_t>({"i","iter","iterations"}, config.maxIterations);
    config.distanceThreshold = args.get<int>({"e","d","threshold"}, config.distanceThreshold);
    config.distanceShowEvery = args.get<size_t>({"s","show","showdistance"}, config.distanceShowEvery);
    config.tempStart = args.get<float>({"ts","tempstart","starttemp"}, config.tempStart);
    config.tempEnd = args.get<float>({"te","tempend","endtemp"}, config.tempEnd);
    config.tempFactor = args.get<float>({"tf","factor","tempfactor"}, config.tempFactor);
    config.tempReduceEvery = args.get<size_t>({"tr","reduce","tempiterations"}, config.tempReduceEvery);
    config.seed = args.get<uint32_t>({"seed"}, config.seed);
    config.loadBalance = args.contains({"b","balance","loadbalance"});
    config.flipManyChance = args.get<float>({"fc","chance","flipchance","flipmanychance"}, config.flipManyChance);
    config.flipManyDepth = args.get<uint32_t>({"fd","depth","flipdepth","flipmanydepth"}, config.flipManyDepth);

     cout << "verbosity " << config.verbosity << "\n"
        << "factorDim " << (int)factorDim << "\n"
        << "maxIterations " << config.maxIterations << "\n"
        << "linesAtOnce " << config.linesAtOnce << "\n"
        << "distanceThreshold " << config.distanceThreshold << "\n"
        << "distanceShowEvery " << config.distanceShowEvery << "\n"
        << "tempStart " << config.tempStart << "\n"
        << "tempEnd " << config.tempEnd << "\n"
        << "tempFactor " << config.tempFactor << "\n"
        << "tempReduceEvery " << config.tempReduceEvery << "\n"
        << "seed " << config.seed << "\n"
        << "loadBalance " << config.loadBalance << "\n"
        << "flipManyChance " << config.flipManyChance << "\n"
        << "flipManyDepth " << config.flipManyDepth << endl;

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
        generate_random_matrix(height, width, factorDim, 3, A0_vec, B0_vec, C0_vec, density);
    } else {
        printf("Wrong data file\n");
        return 0;
    }

    // Initialize Ab, Bb bitvector matrices
    vector<my_bit_vector_t> A_vec, B_vec;
    initializeFactors(A_vec, B_vec, height, width, factorDim, density, &state);

    // copy matrices to GPU and run optimization
    auto cubin = CuBin<my_bit_vector_t>(A_vec, B_vec, C0_vec, factorDim);
    int distance = cubin.getDistance();
    TIMERSTART(GPUKERNELLOOP)
    cubin.run(config);
    TIMERSTOP(GPUKERNELLOOP)
    cubin.verifyDistance();
    cubin.clear();

    return 0;
}

