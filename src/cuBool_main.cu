#include <vector>
#include <iostream>
#include <string>
#include <algorithm> // min
// #include <limits>

#include "helper/rngpu.hpp"
#include "helper/args_parser.h"

#include "config.h"
#include "io_and_allocation.hpp"
#include "cuBool_gpu.cuh"
#include "cuBool_cpu.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

using my_bit_vector_t = uint32_t; // only tested uint32_t

using my_cuBool = cuBool<my_bit_vector_t>;
// using my_cuBool = cuBool_CPU<my_bit_vector_t>;

int main(int argc, char **argv) {
    mc::args_parser args{argc, argv};

    string filename;
    if(args.non_prefixed_count() > 0) {
        filename = args.non_prefixed(0);
    } else {
        cerr << "No input file provided. Abort." << endl;
        return 1;
    }

    size_t numRuns = args.get<size_t>({"r","runs"}, 1);

    my_cuBool::cuBool_config config;
    config.verbosity = args.get<size_t>({"v","verbosity"}, config.verbosity);
    config.factorDim = args.get<uint8_t>({"d","dim","dimension","factordim"}, config.factorDim);
    config.linesAtOnce = args.get<size_t>({"l","lines","linesperkernel"}, config.linesAtOnce);
    config.maxIterations = args.get<size_t>({"i","iter","iterations"}, config.maxIterations);
    config.distanceThreshold = args.get<int>({"e","threshold"}, config.distanceThreshold);
    config.distanceShowEvery = args.get<size_t>({"show","showdistance"}, config.distanceShowEvery);
    config.tempStart = args.get<float>({"ts","tempstart","starttemp"}, config.tempStart);
    config.tempEnd = args.get<float>({"te","tempend","endtemp"}, config.tempEnd);
    config.tempFactor = args.get<float>({"tf","factor","tempfactor"}, config.tempFactor);
    config.tempStep = args.get<size_t>({"tm","step","tempstep","move","tempmove"}, config.tempStep);
    config.seed = args.get<uint32_t>({"seed"}, config.seed);
    config.loadBalance = args.contains({"b","balance","loadbalance"});
    config.flipManyChance = args.get<float>({"fc","chance","flipchance","flipmanychance"}, config.flipManyChance);
    config.flipManyDepth = args.get<uint32_t>({"fd","depth","flipdepth","flipmanydepth"}, config.flipManyDepth);
    config.weight = args.get<int>({"w","weight"}, config.weight);

    config.stuckIterationsBeforeBreak = args.get<size_t>({"stuck"}, config.stuckIterationsBeforeBreak);

    cout << "verbosity " << config.verbosity << "\n"
        << "factorDim " << int(config.factorDim) << "\n"
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
        << "flipManyDepth " << config.flipManyDepth << "\n"
        << "weight " << config.weight << endl;

    // // uint32_t seed = (unsigned long)time(NULL) % UINT32_MAX;
    // uint32_t seed = 46;
        
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);

    // Discard first 100000 entries of PRNG
    for (int i = 0; i < 100000; i++)
        fast_kiss32(state);

    config.seed = fast_kiss32(state);
    
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
        generate_random_matrix(height, width, config.factorDim, 4, A0_vec, B0_vec, C0_vec, density);
    } else {
        cerr << "Bad input file" << endl;
        return 0;
    }

    vector<my_bit_vector_t> A_vec, B_vec;

    size_t numSlots = std::min(size_t(2), numRuns);
    auto cuBool = my_cuBool(C0_vec, height, width, density, numSlots);

    TIMERSTART(GPUKERNELLOOP)
    cuBool.runAllParallel(numRuns, config);
    TIMERSTOP(GPUKERNELLOOP)

    cuBool.getBestFactors(A_vec, B_vec);

    const auto& distances = cuBool.getDistances();
    writeDistancesToFile(filename, distances);

    writeFactorsToFiles(filename + "_best", A_vec, B_vec, config.factorDim);

    auto confusion = computeErrorsCPU(A_vec, B_vec, C0_vec, height, width);

    cout << "true_positives: \t" << confusion.TP << '\t';
    cout << "true_negatives: \t" << confusion.TN << '\n';
    cout << "false_positives:\t" << confusion.FP << '\t';
    cout << "false_negatives:\t" << confusion.FN << '\n';
    cout << "total error:\t" << confusion.total_error() << '\t';
    cout << "rel error:\t" << confusion.rel_error() << '\n';
    cout << "precision:\t" << confusion.precision()*100 << " %\n";
    cout << "recall:   \t" << confusion.sensitivity()*100 << " %\n";
    cout << "F1 score: \t" << confusion.f1score() << endl;

    int count = nonzeroDimension(A_vec);
    cout << "A uses " << count << " of " << int(config.factorDim) << " columns" << endl;
    count = nonzeroDimension(B_vec);
    cout << "B uses " << count << " of " << int(config.factorDim) << " columns" << endl;

    return 0;
}

