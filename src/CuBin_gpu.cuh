#ifndef CUBIN_GPU_CUH
#define CUBIN_GPU_CUH

#include <vector>
#include <iostream>
#include <limits>
#include <type_traits>
#include <omp.h>

#include "helper/rngpu.hpp"
#include "helper/cuda_helpers.cuh"

#include "config.h"
#include "updates_and_measures.cuh"
#include "bit_vector_kernels.cuh"
#include "float_kernels.cuh"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

template<typename factor_t = uint32_t>
class CuBin
{
    using factor_matrix_t = vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = vector<bit_vector_t>;

    struct factor_handler {
        factor_t *d_A;
        factor_t *d_B;
        int *distance_;
        int *d_distance_;
        uint8_t factorDim_ = 20;
        size_t lineSize_ = 1;
        bool initialized_ = false;
    };

public:
    CuBin(const bit_matrix_t& C,
          const size_t height,
          const size_t width,
          const float density,
          const size_t numActiveExperriments = 1)
    {
        cout << "~~~ GPU CuBin ~~~" << endl; 

        int device_id = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        cout << "Using device " << device_id << ": " << prop.name << endl;

        max_parallel_lines_ = prop.multiProcessorCount * WARPSPERBLOCK;

        density_ = density;
        inverse_density_ = 1 / density;

        if(std::is_same<factor_t, uint32_t>::value) {
            lineSize_padded_ = 1;
        }
        else if(std::is_same<factor_t, float>::value) {
            lineSize_padded_ = 32;
        }

        omp_set_num_threads(numActiveExperriments);
        activeExperiments.resize(numActiveExperriments);
        bestFactors = {};

        initializeMatrix(C, height, width);
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << endl;
    }

    ~CuBin() {
        clear();
    }

    // allocate memory for matrix, factors and distances, copy matrix to device
    bool initializeMatrix(const bit_matrix_t& C, const size_t height, const size_t width)
    {
        if( SDIV(height,32) * width != C.size()) {
            cerr << "CuBin construction: Matrix dimension mismatch." << endl;
            return false;
        }

        if(initialized_) {
            cerr << "CuBin already initialized. Please clear CuBin before reinitialization." << endl;
            return false;
        }

        size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        height_ = height;
        // size_t height_padded = SDIV(height_, WARPSPERBLOCK) * WARPSPERBLOCK;
        for(auto& e : activeExperiments) {
            cudaMalloc(&e.d_A, lineBytes_padded * height_); CUERR
        }
        cudaMallocHost(&bestFactors.d_A, lineBytes_padded * height_); CUERR

        width_ = width;
        // size_t width_padded = SDIV(width_, WARPSPERBLOCK) * WARPSPERBLOCK;
        for(auto& e : activeExperiments) {
            cudaMalloc(&e.d_B, lineBytes_padded * width_); CUERR
        }
        cudaMallocHost(&bestFactors.d_B, lineBytes_padded * width_); CUERR
        
        size_t height_C = SDIV(height_, 32);
        width_C_padded_ = SDIV(width_, 32) * 32;
        cudaMalloc(&d_C, sizeof(bit_vector_t) * height_C * width_C_padded_); CUERR

        cudaMemcpy2D(d_C, sizeof(bit_vector_t) * width_C_padded_,
                     C.data(), sizeof(bit_vector_t) * width_,
                     sizeof(bit_vector_t) * width_,
                     height_C,
                     cudaMemcpyHostToDevice); CUERR

        for(auto& e : activeExperiments) {
            cudaMallocHost(&e.distance_, sizeof(int)); CUERR
            cudaMalloc(&e.d_distance_, sizeof(int)); CUERR
            cudaMemset(e.d_distance_, 0, sizeof(int)); CUERR
        }
        cudaMallocHost(&bestFactors.distance_, sizeof(int)); CUERR
        *bestFactors.distance_ = std::numeric_limits<int>::max();

        cout << "CuBin initialization complete." << endl;

        cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;

        return initialized_ = true;
    }

    // initialize factors as copy of host vectors
    bool initializeFactors(const size_t activeId,
                           const factor_matrix_t& A,
                           const factor_matrix_t& B,
                           const uint8_t factorDim,
                           const cudaStream_t stream = 0)
    {
        auto& handler = activeExperiments[activeId];

        handler.factorDim_ = factorDim;

        if(std::is_same<factor_t, uint32_t>::value) {
            handler.lineSize_ = 1;
        }
        else if(std::is_same<factor_t, float>::value) {
            handler.lineSize_ = handler.factorDim_;
        }

        if( A.size() != height_ * handler.lineSize_ || B.size() != width_ * handler.lineSize_) {
            cerr << "CuBin initialization: Factor dimension mismatch." << endl;
            return false;
        }

        size_t lineBytes = sizeof(factor_t) * handler.lineSize_;
        size_t lineBytes_padded = sizeof(factor_t) * handler.lineSize_padded_;

        cudaMemcpy2DAsync(handler.d_A, lineBytes_padded,
                     A.data(), lineBytes,
                     lineBytes,
                     height_,
                     cudaMemcpyHostToDevice,
                     stream);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpy2DAsync(handler.d_B, lineBytes_padded,
                     B.data(), lineBytes,
                     lineBytes,
                     width_,
                     cudaMemcpyHostToDevice,
                     stream);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemsetAsync(handler.d_distance_, 0, sizeof(int), stream);
        // cudaStreamSynchronize(stream); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                        (handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_,
                         handler.factorDim_, inverse_density_, handler.d_distance_);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpyAsync(handler.distance_, handler.d_distance_, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); CUERR

        cout << "Factor initialization complete." << endl;

        cout << "Start distance: "
             << "\tabs_err: " << *handler.distance_
             << "\trel_err: " << (float) *handler.distance_ / (height_ * width_)
             << endl;

        return handler.initialized_ = true;
    }

    // initialize factors on device according to INITIALIZATIONMODE
    bool initializeFactors(const size_t activeId, const uint8_t factorDim, uint32_t seed, const cudaStream_t stream = 0) {
        auto& handler = activeExperiments[activeId];

        handler.factorDim_ = factorDim;

        if(std::is_same<factor_t, uint32_t>::value) {
            handler.lineSize_ = 1;
        }
        else if(std::is_same<factor_t, float>::value) {
            handler.lineSize_ = handler.factorDim_;
        }

        float threshold = getInitChance(density_, handler.factorDim_);
        
        initFactor <<< SDIV(height_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32, 0, stream >>>
                    (handler.d_A, height_, handler.factorDim_, seed, threshold);
        // cudaStreamSynchronize(stream); CUERR

        seed += height_;
        initFactor <<< SDIV(width_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32, 0, stream >>>
                    (handler.d_B, width_, handler.factorDim_, seed, threshold);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemsetAsync(handler.d_distance_, 0, sizeof(int), stream);
        // cudaStreamSynchronize(stream); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                        (handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_,
                         handler.factorDim_, inverse_density_, handler.d_distance_);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpyAsync(handler.distance_, handler.d_distance_, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); CUERR

        cout << "Factor initialization complete." << endl;

        cout << "Start distance: "
             << "\tabs_err: " << *handler.distance_
             << "\trel_err: " << (float) *handler.distance_ / (height_ * width_)
             << endl;

        return handler.initialized_ = true;
    }

    bool verifyDistance(const size_t activeId) {
        auto& handler = activeExperiments[activeId];

        if(!initialized_) {
            cerr << "CuBin not initialized." << endl;
            return false;
        }

        int* distance_proof;
        int* d_distance_proof;

        cudaMallocHost(&distance_proof, sizeof(int)); CUERR
        cudaMalloc(&d_distance_proof, sizeof(int)); CUERR
        cudaMemset(d_distance_proof, 0, sizeof(int)); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                        (handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_,
                         handler.factorDim_, inverse_density_, d_distance_proof);

        cudaDeviceSynchronize(); CUERR

        cudaMemcpy(distance_proof, d_distance_proof, sizeof(int), cudaMemcpyDeviceToHost); CUERR

        bool equal = *handler.distance_ == *distance_proof;
        if(!equal) {
            cout << "----- !Distances differ! -----\n";
            cout << "Running distance:  " << *handler.distance_ << "\n";
            cout << "Real distance:     " << *distance_proof << endl;
        } else {
            cout << "Distance verified" << endl;
        }

        cudaFreeHost(distance_proof);
        cudaFree(d_distance_proof);
        return equal;
    } 

    void clear() {
        if(initialized_) {
            cudaFree(d_C);
            for(auto& e : activeExperiments) {
                cudaFree(e.d_A);
                cudaFree(e.d_B);
                cudaFreeHost(e.distance_);
                cudaFree(e.d_distance_);
            }
            cudaFreeHost(bestFactors.d_A);
            cudaFreeHost(bestFactors.d_B);
            cudaFreeHost(bestFactors.distance_);

            initialized_ = false;
        }
    }

    void getFactors(const size_t activeId, factor_matrix_t& A, factor_matrix_t& B, const cudaStream_t stream = 0) {
        auto& handler = activeExperiments[activeId];

        if(!initialized_) {
            cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t lineBytes = sizeof(factor_t) * handler.lineSize_;
        size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        A.resize(height_);
        cudaMemcpy2DAsync(A.data(), lineBytes,
                     handler.d_A, lineBytes_padded,
                     lineBytes,
                     height_,
                     cudaMemcpyDeviceToHost,
                     stream);
        // cudaStreamSynchronize(stream); CUERR

        B.resize(width_);
        cudaMemcpy2DAsync(B.data(), lineBytes,
                     handler.d_B, lineBytes_padded,
                     lineBytes,
                     width_,
                     cudaMemcpyDeviceToHost,
                     stream);
        // cudaStreamSynchronize(stream); CUERR
    }

    void getBestFactors(factor_matrix_t& A, factor_matrix_t& B, const cudaStream_t stream = 0) {
        if(!initialized_) {
            cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t lineBytes = sizeof(factor_t) * bestFactors.lineSize_;

        A.resize(height_);
        cudaMemcpyAsync(A.data(),
                     bestFactors.d_A,
                     lineBytes * height_,
                     cudaMemcpyHostToHost,
                     stream);
        // cudaStreamSynchronize(stream); CUERR

        B.resize(width_);
        cudaMemcpyAsync(B.data(),
                     bestFactors.d_B,
                     lineBytes * width_,
                     cudaMemcpyHostToHost,
                     stream);
        // cudaStreamSynchronize(stream); CUERR
    }

    // int getDistance() {
    //     if(!initialized_) {
    //         cerr << "CuBin not initialized." << endl;
    //         return -1;
    //     }
    //     return *distance_;
    // }

    struct CuBin_config {
        size_t verbosity = 1;
        size_t linesAtOnce = 0;
        size_t maxIterations = 0;
        int distanceThreshold = 0;
        size_t distanceShowEvery = std::numeric_limits<size_t>::max();
        float tempStart = 0.0f;
        float tempEnd = -1.0f;
        float tempFactor = 0.98f;
        size_t tempStep = std::numeric_limits<size_t>::max();
        uint32_t seed = 0;
        bool loadBalance = false;
        float flipManyChance = 0.1f;
        uint32_t flipManyDepth = 2;
        size_t stuckIterationsBeforeBreak = std::numeric_limits<size_t>::max();
    };

    void runAllParallel(const size_t numExperiments, const CuBin_config& config) {
        uint8_t factorDim = 20;
        uint32_t seed = 123;

        #pragma omp parallel for //schedule(dynamic,1)
        for(size_t i=0; i<numExperiments; ++i) {
            unsigned id = omp_get_thread_num();
            #pragma omp ciritcal
            cout << "Starting run " << i << " in slot " << id << endl;
            initializeFactors(id, factorDim, seed+id);
            run(id, config);
        }
    }

    void run(const size_t activeId, const CuBin_config& config, const cudaStream_t stream = 0) {
        auto& handler = activeExperiments[activeId];

        if(!initialized_ || !handler.initialized_) {
            cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t linesAtOnce = SDIV(config.linesAtOnce, WARPSPERBLOCK) * WARPSPERBLOCK;
        if(config.loadBalance) {
            linesAtOnce = linesAtOnce / max_parallel_lines_ * max_parallel_lines_;
            if (!linesAtOnce) linesAtOnce = max_parallel_lines_;
        }

        if(config.verbosity > 0) {
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
            cout << "- - - - Starting " << config.maxIterations
                 << " GPU iterations, changing " << linesAtOnce
                 << " lines each time\n";
            cout << "- - - - Showing error every " << config.distanceShowEvery
                 << " steps\n";
            if(config.tempStart > 0) {
                cout << "- - - - Start temperature " << config.tempStart
                     << " multiplied by " << config.tempFactor
                     << " every " << config.tempStep
                     << " steps\n";

            }
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -";
            cout << endl;
        }

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
        float temperature = config.tempStart;
        size_t iteration = 0;
        size_t stuckIterations = 0;
        auto distancePrev = *handler.distance_;
        while(
                *handler.distance_ > config.distanceThreshold &&
                iteration++ < config.maxIterations
                && temperature > config.tempEnd
                && stuckIterations < config.stuckIterationsBeforeBreak) {

            // Change rows
            int lineToBeChanged = (fast_kiss32(state) % height_) / WARPSPERBLOCK * WARPSPERBLOCK;
            uint32_t gpuSeed = fast_kiss32(state) + iteration;

            vectorMatrixMultCompareRowWarpShared 
                <<< SDIV(min(linesAtOnce, height_), WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                (handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_, handler.factorDim_,
                 lineToBeChanged, handler.d_distance_, gpuSeed, temperature/10,
                 config.flipManyChance, config.flipManyDepth, inverse_density_);
            // cudaStreamSynchronize(stream); CUERR

            // Change cols
            lineToBeChanged = (fast_kiss32(state) % width_) / WARPSPERBLOCK * WARPSPERBLOCK;
            gpuSeed = fast_kiss32(state) + iteration;

            vectorMatrixMultCompareColWarpShared 
                <<< SDIV(min(linesAtOnce, width_), WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                (handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_, handler.factorDim_,
                 lineToBeChanged, handler.d_distance_, gpuSeed, temperature/10,
                 config.flipManyChance, config.flipManyDepth, inverse_density_);
            // cudaStreamSynchronize(stream); CUERR

            cudaMemcpyAsync(handler.distance_, handler.d_distance_, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream); CUERR

            if(config.verbosity > 0 && iteration % config.distanceShowEvery == 0) {
                cout << "Iteration: " << iteration
                     << "\tabs_err: " << *handler.distance_
                     << "\trel_err: " << (float) *handler.distance_ / (height_*width_)
                     << "\ttemp: " << temperature;
                cout << endl;
            }
            if(iteration % config.tempStep == 0) {
                temperature *= config.tempFactor;
            }
            if(*handler.distance_ == distancePrev)
                stuckIterations++;
            else
                stuckIterations = 0;
            distancePrev = *handler.distance_;
        }

        if(config.verbosity > 0) {
            if (!(iteration < config.maxIterations))
                cout << "Reached iteration limit: " << config.maxIterations << '\n';
            if (!(*handler.distance_ > config.distanceThreshold))
                cout << "Distance below threshold.\n";
            if (!(temperature > config.tempEnd))
                cout << "Temperature below threshold.\n";
            if (!(stuckIterations < config.stuckIterationsBeforeBreak))
                cout << "Stuck for " << stuckIterations << " iterations.\n";
        }
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        cout << "Final result: "
             << "\tabs_err: " << *handler.distance_
             << "\trel_err: " << (float) *handler.distance_ / (height_ * width_)
             << endl;

        if(*handler.distance_ < *bestFactors.distance_) {

            #pragma omp critical
            if(*handler.distance_ < *bestFactors.distance_) {
                cout << "Result is better than previous best. Copying to host." << endl;

                *bestFactors.distance_ = *handler.distance_;
                bestFactors.lineSize_ = handler.lineSize_;
                bestFactors.factorDim_ = handler.factorDim_;

                size_t lineBytes = sizeof(factor_t) * handler.lineSize_;
                size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

                cudaMemcpy2DAsync(bestFactors.d_A, lineBytes,
                             handler.d_A, lineBytes_padded,
                             lineBytes,
                             height_,
                             cudaMemcpyDeviceToHost,
                             stream);

                cudaMemcpy2DAsync(bestFactors.d_B, lineBytes,
                             handler.d_B, lineBytes_padded,
                             lineBytes,
                             width_,
                             cudaMemcpyDeviceToHost,
                             stream);
            }
        }
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << endl;
    }  

private:
    bool initialized_ = false;

    bit_vector_t *d_C;
    float density_;
    int inverse_density_;
    size_t height_ = 0;
    size_t width_ = 0;
    size_t width_C_padded_ = 0;
    size_t lineSize_padded_ = 1;
    int max_parallel_lines_;

    factor_handler bestFactors;
    vector<factor_handler> activeExperiments;
};

#endif
