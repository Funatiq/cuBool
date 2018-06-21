#include <vector>
#include <iostream>
#include <string>
// #include <limits>

#include "helper/rngpu.hpp"
#include "helper/args_parser.h"

#include "config.h"
#include "io_and_allocation.hpp"
#include "CuBin_gpu.cuh"
#include "CuBin_cpu.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

using my_bit_vector_t = uint32_t; // only tested uint32_t

using my_cubin = CuBin<my_bit_vector_t>;
// using my_cubin = Cubin_CPU<my_bit_vector_t>;

int main(int argc, char **argv) {
    mc::args_parser args{argc, argv};

    auto filename = string{};
    if(args.non_prefixed_count() > 0) {
        filename = args.non_prefixed(0);
    } else {
        cerr << "No input file provided. Abort." << endl;
        return 1;
    }

    my_cubin::CuBin_config config;
    config.verbosity = args.get<size_t>({"v","verbosity"}, config.verbosity);
    uint8_t factorDim = args.get<uint8_t>({"k","dim","dimension","factordim"}, 20);
    config.linesAtOnce = args.get<size_t>({"l","lines","linesperkernel"}, config.linesAtOnce);
    config.maxIterations = args.get<size_t>({"i","iter","iterations"}, config.maxIterations);
    config.distanceThreshold = args.get<int>({"e","d","threshold"}, config.distanceThreshold);
    config.distanceShowEvery = args.get<size_t>({"s","show","showdistance"}, config.distanceShowEvery);
    config.tempStart = args.get<float>({"ts","tempstart","starttemp"}, config.tempStart);
    config.tempEnd = args.get<float>({"te","tempend","endtemp"}, config.tempEnd);
    config.tempFactor = args.get<float>({"tf","factor","tempfactor"}, config.tempFactor);
    config.tempStep = args.get<size_t>({"tm","step","tempstep","move","tempmove"}, config.tempStep);
    config.seed = args.get<uint32_t>({"seed"}, config.seed);
    config.loadBalance = args.contains({"b","balance","loadbalance"});
    config.flipManyChance = args.get<float>({"fc","chance","flipchance","flipmanychance"}, config.flipManyChance);
    config.flipManyDepth = args.get<uint32_t>({"fd","depth","flipdepth","flipmanydepth"}, config.flipManyDepth);

    config.stuckIterationsBeforeBreak = args.get<size_t>({"stuck"}, config.stuckIterationsBeforeBreak);

    cout << "verbosity " << config.verbosity << "\n"
        << "factorDim " << (int)factorDim << "\n"
        << "maxIterations " << config.maxIterations << "\n"
        << "linesAtOnce " << config.linesAtOnce << "\n"
        << "distanceThreshold " << config.distanceThreshold << "\n"
        << "distanceShowEvery " << config.distanceShowEvery << "\n"
        << "stuckIterationsBeforeBreak " << config.stuckIterationsBeforeBreak << "\n"
        << "tempStart " << config.tempStart << "\n"
        << "tempEnd " << config.tempEnd << "\n"
        << "tempFactor " << config.tempFactor << "\n"
        << "tempStep " << config.tempStep << "\n"
        << "seed " << config.seed << "\n"
        << "loadBalance " << config.loadBalance << "\n"
        << "flipManyChance " << config.flipManyChance << "\n"
        << "flipManyDepth " << config.flipManyDepth << endl;

    // // uint32_t seed = (unsigned long)time(NULL) % UINT32_MAX;
    // uint32_t seed = 46;
        
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);

    // Discard first 100000 entries of PRNG
    for (int i = 0; i < 100000; i++)
        fast_kiss32(state);
    
    int height, width;
    float density;
    
    vector<my_bit_vector_t> A0_vec, B0_vec, C0_vec;
    string ending = ".in";
    if (endsWith(filename, ending)) {
        // Read file and save matrix in C0
        // COO coordinates
        readInputFileData(filename, C0_vec, height, width, density);
    } else if (filename.compare("test") == 0) {
        height = 5000;
        width = 4000;
        // height = 5*1024;
        // width = 5*1024;
        generate_random_matrix(height, width, factorDim, 4, A0_vec, B0_vec, C0_vec, density);
    } else {
        cerr << "Wrong data file" << endl;
        return 0;
    }

    // Initialize Ab, Bb bitvector matrices
    vector<my_bit_vector_t> A_vec, B_vec;
    // initializeFactors(A_vec, B_vec, height, width, factorDim, density, state);

    // computeHammingDistanceCPU(A_vec, B_vec, C0_vec, height, width);
    // writeToFiles(filename + "_start", A_vec, B_vec, height, width, factorDim);

    // copy matrices to GPU and run optimization
    // auto cubin = my_cubin(A_vec, B_vec, C0_vec, factorDim, density);
    auto cubin = my_cubin(C0_vec, height, width, density, 4);
    // cubin.initializeFactors(A_vec, B_vec);
    // cubin.initializeFactors(0, factorDim, fast_kiss32(state));

    // auto distance = cubin.getDistance();
    // cubin.verifyDistance();
    TIMERSTART(GPUKERNELLOOP)
    // cubin.run(0, config);
    cubin.runAllParallel(8, config);
    TIMERSTOP(GPUKERNELLOOP)
    // cubin.verifyDistance();

    // cubin.getFactors(0, A_vec, B_vec);
    cubin.getBestFactors(A_vec, B_vec);
    cubin.clear();

    writeToFiles(filename + "_end", A_vec, B_vec, height, width, factorDim);
    
    // auto error = computeHammingDistanceCPU(A_vec, B_vec, C0_vec, height, width);
    // std::cout << "cpu error: " << error << std::endl;

    // auto C = computeProductCOO(A_vec, B_vec, height, width);
    // cout << "Nonzeros in product: " << C.size() << endl;

    auto confusion = computeErrorsCPU(A_vec, B_vec, C0_vec, height, width);

    cout << "true_positives: \t" << confusion.TP << '\t';
    cout << "true_negatives: \t" << confusion.TN << '\n';
    cout << "false_positives:\t" << confusion.FP << '\t';
    cout << "false_negatives:\t" << confusion.FN << '\n';
    cout << "total error:\t" << confusion.total_error() << '\t';
    cout << "rel error:\t" << confusion.rel_error() << endl;
    cout << "precision:\t" << confusion.precision() << endl;
    cout << "recall:   \t" << confusion.sensitivity() << endl;
    return 0;
}

