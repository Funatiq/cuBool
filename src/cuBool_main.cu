#include <vector>
#include <iostream>
#include <string>
#include <algorithm> // min
// #include <limits>

#include "helper/rngpu.hpp"
#include "helper/clipp.h"

#include "config.h"
#include "io_and_allocation.hpp"
#include "bit_vector_functions.h"
#ifdef USE_CPU
#include "cuBool_cpu.h"
#else
#include "cuBool_gpu.cuh"
#endif

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

using clipp::value;
using clipp::option;

using my_bit_vector_t = uint32_t; // only tested uint32_t
using my_cuBool = cuBool<my_bit_vector_t>;

int main(int argc, char **argv) {
    string filename;
    size_t numRuns = 1;
    my_cuBool::cuBool_config config;

    auto cli = (
        value("dataset file", filename),
        (option("-r") & value("runs", numRuns)) % "number of runs",
        (option("-v") & value("verbosity", config.verbosity)) % "verbosity",
        (option("-d") & value("dim", config.factorDim)) % "latent dimension",
        (option("-l") & value("lines", config.linesAtOnce)) % "number of lines to update per iteration",
        (option("-i") & value("iter", config.maxIterations)) % "maximum number of iterations",
        (option("-e") & value("err", config.distanceThreshold)) % "error threshold",
        (option("--show") & value("s", config.distanceShowEvery)) % "show distance every <s> iterations",
        (option("--ts") & value("start temp", config.tempStart)) % "start temperature",
        (option("--te") & value("end temp", config.tempEnd)) % "end temperature",
        (option("--tf") & value("factor", config.tempFactor)) % "temperature reduction factor",
        (option("--tm") & value("move", config.tempStep)) % "reduce temperature every <tm> iterations",
        (option("--seed") & value("seed", config.seed)) % "seed for pseudo random numbers",
        (option("--fc") & value("flip chance", config.flipManyChance)) % "chance to flip multiple bits",
        (option("--fd") & value("flip depth", config.flipManyDepth)) % "flip chance for each bit in multi flip (negative power of two)",
        (option("-w", "--weight") & value("weight", config.weight)) % "weight in error measure",
        (option("--stuck") & value("s", config.stuckIterationsBeforeBreak)) % "stop if stuck for <s> iterations"
    );

    auto parseResult = clipp::parse(argc, argv, cli);
    if(!parseResult) {
        auto fmt = clipp::doc_formatting{}.doc_column(30);
        cout << clipp::make_man_page(cli, argv[0], fmt);
        return 1;
    }

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

    if (filename.compare("test") == 0) {
        height = 5000;
        width = 5000;
        generate_random_matrix(height, width, config.factorDim, 4, A0_vec, B0_vec, C0_vec, density);
    } else {
        readInputFileData(filename, C0_vec, height, width, density);
    }

    vector<my_bit_vector_t> A_vec, B_vec;

    size_t numSlots = std::min(size_t(2), numRuns);
    auto cuBool = my_cuBool(C0_vec, height, width, density, numSlots);

    TIMERSTART(GPUKERNELLOOP)
    cuBool.runMultiple(numRuns, config);
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

