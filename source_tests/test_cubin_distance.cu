#include <vector>
#include <iostream>
#include <limits>

#include "../source/helper/config.h"
#include "../source/helper/rngpu.hpp"
#include "../source/helper/io_and_allocation.hpp"

#include "../source/CuBin_gpu.cuh"

using namespace std;

using my_bit_vector_t = uint32_t; // only tested uint32_t
using my_factor_t = float;

template<typename factor_t>
void test(typename CuBin<factor_t>::CuBin_config config) {
    if(std::is_same<factor_t, uint32_t>::value) {
        cout << "testing uint32_t" << endl;
    }
    else if(std::is_same<factor_t, float>::value) {
        cout << "testing float" << endl;
    }
    else {
        cout << "warning: unknown type" << endl;
    }

    const uint8_t factorDim = 20;

    int distanceIterations = 100;

    // // uint32_t seed = (unsigned long)time(NULL) % UINT32_MAX;
    uint32_t seed = 46;
        
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);

    // Discard first 100000 entries of PRNG
    for (int i = 0; i < 100000; i++)
        fast_kiss32(state);
    
    int height, width;
    float density;
    
    vector<factor_t> A0_vec, B0_vec;
    vector<my_bit_vector_t> C0_vec;

    height = 5000;
    width = 5000;
    // height = 5*1024;
    // width = 5*1024;
    TIMERSTART(GENERATE)
    generate_random_matrix(height, width, factorDim, 3, A0_vec, B0_vec, C0_vec, density);
    TIMERSTOP(GENERATE)

    // copy matrices to GPU and run optimization
    auto cubin0 = CuBin<factor_t>(A0_vec, B0_vec, C0_vec, factorDim);
    int distance = cubin0.getDistance();
    cout << "Start distance: " << (float) distance / (height*width)
         << " = " << distance << " elements" << endl;
    TIMERSTART(DISTANCE)
    for(int i=0; i<distanceIterations; ++i)
        cubin0.verifyDistance();
    TIMERSTOP(DISTANCE)  
    cubin0.clear();

    if(distance > 0) {
        cout << "Distance should be 0! Abort." << endl;
        return;
    }

    // Initialize Ab, Bb bitvector matrices
    vector<factor_t> A_vec, B_vec;
    TIMERSTART(INIT)
    initializeFactors(A_vec, B_vec, height, width, factorDim, density, state);
    TIMERSTOP(INIT)

    // copy matrices to GPU and run optimization
    auto cubin = CuBin<factor_t>(A_vec, B_vec, C0_vec, factorDim);
    distance = cubin.getDistance();
    cout << "Start distance: " << (float) distance / (height*width)
         << " = " << distance << " elements" << endl;
    TIMERSTART(DISTANCE2)
    for(int i=0; i<distanceIterations; ++i)
        cubin.verifyDistance();
    TIMERSTOP(DISTANCE2)


    // typename CuBin<factor_t>::CuBin_config config;
    // config.verbosity = 2;
    // // config.linesAtOnce = 2000;
    // config.maxIterations = maxIterations;
    // // config.distanceThreshold = -1;
    // // config.distanceShowEvery = 2500;
    // config.distanceShowEvery = 5000;
    // config.tempStart = 50.0f / height;
    // // config.tempStart = 100.0f;
    // config.tempEnd = .1f / height;
    // // config.tempEnd = 9.0f;
    // config.tempFactor = 0.98;
    // config.tempStep = 10000;
    // // config.seed = 40; // 0 error for float, w=h=5*1024 or 5000
    // // config.seed = 42; // T=100-9:0.98every5000 works for float 5*1024
    // // config.seed = 41; // local min for float, w=h=5*1024
    // config.seed = 44;
    // config.loadBalance = true;
    // // config.flipManyChance = 0;
    // // config.flipManyDepth = 0;

    cout << "verbosity " << config.verbosity << "\n"
        << "maxIterations " << config.maxIterations << "\n"
        << "linesAtOnce " << config.linesAtOnce << "\n"
        << "distanceThreshold " << config.distanceThreshold << "\n"
        << "distanceShowEvery " << config.distanceShowEvery << "\n"
        << "tempStart " << config.tempStart << "\n"
        << "tempEnd " << config.tempEnd << "\n"
        << "tempFactor " << config.tempFactor << "\n"
        << "tempStep " << config.tempStep << "\n"
        << "seed " << config.seed << "\n"
        << "loadBalance " << config.loadBalance << "\n"
        << "flipManyChance " << config.flipManyChance << "\n"
        << "flipManyDepth " << config.flipManyDepth << endl;

    TIMERSTART(GPUKERNELLOOP)
    cubin.run(config);
    TIMERSTOP(GPUKERNELLOOP)
    cubin.verifyDistance();
    cubin.clear();

    cout << endl;
}

int main(int argc, char **argv) {

    CuBin<uint32_t>::CuBin_config config_i;
    config_i.verbosity = 2;
    // config_i.linesAtOnce = 2000;
    // config_i.maxIterations = -1;
    config_i.maxIterations = 10000;
    // config_i.distanceThreshold = -1;
    // config_i.distanceShowEvery = 2500;
    config_i.distanceShowEvery = 1000;
    // config_i.tempStart = 100.0f / 5120;
    // config_i.tempStart = 100.0f;
    // config_i.tempEnd = .1f / 5120;
    // config_i.tempEnd = 9.0f;
    // config_i.tempFactor = 0.98;
    // config_i.tempStep = 10000;
    config_i.seed = 40; // 0 error for float, w=h=5*1024 or 5000
    // config_i.seed = 42; // T=100-9:0.98every5000 works for float 5*1024
    // config_i.seed = 41; // local min for float, w=h=5*1024
    // config_i.seed = 44;
    config_i.loadBalance = true;
    // config_i.flipManyChance = 0;
    // config_i.flipManyDepth = 0;

    test<uint32_t>(config_i);

    CuBin<float>::CuBin_config config_f;
    config_f.verbosity = 2;
    // // config_f.linesAtOnce = 2000;
    config_f.maxIterations = 10000;
    // // config_f.distanceThreshold = -1;
    // config_f.distanceShowEvery = 2500;
    config_f.distanceShowEvery = 1000;
    // config_f.tempStart = 50.0f / 5120;
    // // config_f.tempStart = 100.0f;
    // config_f.tempEnd = .1f / 5120;
    // // config_f.tempEnd = 9.0f;
    // config_f.tempFactor = 0.98;
    // config_f.tempStep = 10000;
    config_f.seed = 40; // 0 error for float, w=h=5*1024 or 5000
    // // config_f.seed = 42; // T=100-9:0.98every5000 works for float 5*1024
    // // config_f.seed = 41; // local min for float, w=h=5*1024
    // config_f.seed = 44;
    config_f.loadBalance = true;
    // // config_f.flipManyChance = 0;
    // // config_f.flipManyDepth = 0;

    test<float>   (config_f);

    return 0;
}

