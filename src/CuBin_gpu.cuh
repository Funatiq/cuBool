#ifndef CUBIN_GPU_CUH
#define CUBIN_GPU_CUH

#include <vector>
#include <iostream>
#include <limits>
#include <type_traits>

#include "helper/rngpu.hpp"
#include "helper/cuda_helpers.cuh"

#include "config.h"
#include "updates_and_measures.cuh"
#include "bit_vector_kernels.cuh"
#include "float_kernels.cuh"

using std::cout;
using std::endl;


template<typename factor_t = uint32_t>
class CuBin
{
    using factor_matrix_t = std::vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = std::vector<bit_vector_t>;

public:
    CuBin(const factor_matrix_t& A,
          const factor_matrix_t& B,
          const bit_matrix_t& C,
          const uint8_t factorDim,
          const float density)
    {
        cout << "~~~ GPU CuBin ~~~" << endl; 

        int device_id = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        cout << "Using device " << device_id << ": " << prop.name << endl;

        max_parallel_lines_ = prop.multiProcessorCount * WARPSPERBLOCK;

        if(factorDim > 32) {
            std::cerr << "Factor dimension too big! Maximum is 32." << endl;
            factorDim_ = 32;
        }
        else factorDim_ = factorDim;

        density_ = density;
        inverse_density_ = 1 / density;

        if(std::is_same<factor_t, uint32_t>::value) {
            lineSize_ = 1;
            lineSize_padded_ = 1;
        }
        if(std::is_same<factor_t, float>::value) {
            lineSize_ = factorDim_;
            lineSize_padded_ = 32;
        }

        initialize(A, B, C);
    }

    ~CuBin() {
        clear();
    }

    bool initialize(const bit_matrix_t& C, const size_t height, const size_t width) {

        if( SDIV(height,32) * width != C.size()) {
            std::cerr << "CuBin construction: Matrix dimension mismatch." << endl;
            return false;
        }

        if(initialized_) {
            std::cerr << "CuBin already initialized. Please clear CuBin before reinitialization." << endl;
            return false;
        }

        size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        height_ = height;
        // size_t height_padded = SDIV(height_, WARPSPERBLOCK) * WARPSPERBLOCK;
        cudaMalloc(&d_A, lineBytes_padded * height_); CUERR

        width_ = width;
        // size_t width_padded = SDIV(width_, WARPSPERBLOCK) * WARPSPERBLOCK;
        cudaMalloc(&d_B, lineBytes_padded * width_); CUERR
        
        size_t height_C = SDIV(height_, 32);
        width_C_padded_ = SDIV(width_, 32) * 32;
        cudaMalloc(&d_C, sizeof(bit_vector_t) * height_C * width_C_padded_); CUERR

        cudaMemcpy2D(d_C, sizeof(bit_vector_t) * width_C_padded_,
                     C.data(), sizeof(bit_vector_t) * width_,
                     sizeof(bit_vector_t) * width_,
                     height_C,
                     cudaMemcpyHostToDevice); CUERR

        cudaMallocHost(&distance_, sizeof(int)); CUERR
        cudaMalloc(&d_distance_, sizeof(int)); CUERR
        cudaMemset(d_distance_, 0, sizeof(int)); CUERR

        cout << "CuBin initialization complete." << endl;

        cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;
        cout << "Factor dimension:\t" << (int) factorDim_ << endl;

        return initialized_ = true;
    }

    // initialize factors as copy of host vectors
    bool initializeFactors(const factor_matrix_t& A, const factor_matrix_t& B, cudaStream_t stream = 0) {
        if( A.size() != height_ * lineSize_ || B.size() != width_ * lineSize_) {
            std::cerr << "CuBin initialization: Factor dimension mismatch." << endl;
            return false;
        }

        size_t lineBytes = sizeof(factor_t) * lineSize_;
        size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        cudaMemcpy2DAsync(d_A, lineBytes_padded,
                     A.data(), lineBytes,
                     lineBytes,
                     height_,
                     cudaMemcpyHostToDevice,
                     stream);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpy2DAsync(d_B, lineBytes_padded,
                     B.data(), lineBytes,
                     lineBytes,
                     width_,
                     cudaMemcpyHostToDevice,
                     stream);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemsetAsync(d_distance_, 0, sizeof(int), stream);
        // cudaStreamSynchronize(stream); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                        (d_A, d_B, d_C, height_, width_, width_C_padded_,
                         factorDim_, inverse_density_, d_distance_);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpyAsync(distance_, d_distance_, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); CUERR

        cout << "Factor initialization complete." << endl;

        cout << "Start distance: "
             << "\tabs_err: " << *distance_
             << "\trel_err: " << (float) *distance_ / (height_ * width_)
             << endl;

        return true;
    }

    // initialize factors on device according to INITIALIZATIONMODE
    bool initializeFactors(cudaStream_t stream = 0) {
        float threshold = getInitChance(density_, factorDim_);
        
        uint32_t seed = 0;
        initFactor <<< SDIV(height_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32, 0, stream >>>
                    (d_A, height_, factorDim_, seed, threshold);
        // cudaStreamSynchronize(stream); CUERR

        seed += height_;
        initFactor <<< SDIV(width_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32, 0, stream >>>
                    (d_B, width_, factorDim_, seed, threshold);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemsetAsync(d_distance_, 0, sizeof(int), stream);
        // cudaStreamSynchronize(stream); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                        (d_A, d_B, d_C, height_, width_, width_C_padded_,
                         factorDim_, inverse_density_, d_distance_);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpyAsync(distance_, d_distance_, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); CUERR

        cout << "Factor initialization complete." << endl;

        cout << "Start distance: "
             << "\tabs_err: " << *distance_
             << "\trel_err: " << (float) *distance_ / (height_ * width_)
             << endl;

        return true;
    }


    bool initialize(const factor_matrix_t& A, const factor_matrix_t& B, const bit_matrix_t& C) {
        initialize(C, A.size()/lineSize_, B.size()/lineSize_);
        initializeFactors(A, B);
        // initializeFactors();
        return true;
    }

    bool verifyDistance() {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return false;
        }

        int* distance_proof;
        int* d_distance_proof;

        cudaMallocHost(&distance_proof, sizeof(int)); CUERR
        cudaMalloc(&d_distance_proof, sizeof(int)); CUERR
        cudaMemset(d_distance_proof, 0, sizeof(int)); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                        (d_A, d_B, d_C, height_, width_, width_C_padded_,
                         factorDim_, inverse_density_, d_distance_proof);

        cudaDeviceSynchronize(); CUERR

        cudaMemcpy(distance_proof, d_distance_proof, sizeof(int), cudaMemcpyDeviceToHost); CUERR

        bool equal = *distance_ == *distance_proof;
        if(!equal) {
            cout << "----- !Distances differ! -----\n";
            cout << "Running distance:  " << *distance_ << "\n";
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
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            cudaFreeHost(distance_);
            cudaFree(d_distance_);
            initialized_ = false;
        }
    }

    void getFactors(factor_matrix_t& A, factor_matrix_t& B) {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t lineBytes = sizeof(factor_t) * lineSize_;
        size_t lineBytes_padded = sizeof(factor_t) * lineSize_padded_;

        A.resize(height_);
        cudaMemcpy2D(A.data(), lineBytes,
                     d_A, lineBytes_padded,
                     lineBytes,
                     height_,
                     cudaMemcpyDeviceToHost); CUERR
        
        B.resize(width_);
        cudaMemcpy2D(B.data(), lineBytes,
                     d_B, lineBytes_padded,
                     lineBytes,
                     width_,
                     cudaMemcpyDeviceToHost); CUERR
    }

    int getDistance() {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return -1;
        }
        return *distance_;
    }

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

    void run(const CuBin_config& config, cudaStream_t stream = 0) {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
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
        auto distancePrev = *distance_;
        while(
                *distance_ > config.distanceThreshold &&
                iteration++ < config.maxIterations
                && temperature > config.tempEnd
                && stuckIterations < config.stuckIterationsBeforeBreak) {

            // Change rows
            int lineToBeChanged = (fast_kiss32(state) % height_) / WARPSPERBLOCK * WARPSPERBLOCK;
            uint32_t gpuSeed = fast_kiss32(state) + iteration;

            vectorMatrixMultCompareRowWarpShared 
                <<< SDIV(min(linesAtOnce, height_), WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                (d_A, d_B, d_C, height_, width_, width_C_padded_, factorDim_,
                 lineToBeChanged, d_distance_, gpuSeed, temperature/10,
                 config.flipManyChance, config.flipManyDepth, inverse_density_);

            // cudaDeviceSynchronize(); CUERR

            // Change cols
            lineToBeChanged = (fast_kiss32(state) % width_) / WARPSPERBLOCK * WARPSPERBLOCK;
            gpuSeed = fast_kiss32(state) + iteration;

            vectorMatrixMultCompareColWarpShared 
                <<< SDIV(min(linesAtOnce, width_), WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                (d_A, d_B, d_C, height_, width_, width_C_padded_, factorDim_,
                 lineToBeChanged, d_distance_, gpuSeed, temperature/10,
                 config.flipManyChance, config.flipManyDepth, inverse_density_);

            // cudaDeviceSynchronize(); CUERR

            cudaMemcpyAsync(distance_, d_distance_, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream); CUERR

            if(config.verbosity > 0 && iteration % config.distanceShowEvery == 0) {
                cout << "Iteration: " << iteration
                     << "\tabs_err: " << *distance_
                     << "\trel_err: " << (float) *distance_ / (height_*width_)
                     << "\ttemp: " << temperature;
                cout << endl;
            }
            if(iteration % config.tempStep == 0) {
                temperature *= config.tempFactor;
            }
            if(*distance_ == distancePrev)
                stuckIterations++;
            else
                stuckIterations = 0;
            distancePrev = *distance_;
        }

        if(config.verbosity > 0) {
            if (!(iteration < config.maxIterations))
                cout << "Reached iteration limit: " << config.maxIterations << endl;
            if (!(*distance_ > config.distanceThreshold))
                cout << "Distance below threshold." << endl;
            if (!(temperature > config.tempEnd))
                cout << "Temperature below threshold." << endl;
            if (!(stuckIterations < config.stuckIterationsBeforeBreak))
                cout << "Stuck for " << stuckIterations << " iterations." << endl;
        }
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        cout << "Final result: "
             << "\tabs_err: " << *distance_
             << "\trel_err: " << (float) *distance_ / (height_ * width_)
             << endl;
    }  

private:
    bool initialized_ = false;
    // factor_matrix_t A;
    // factor_matrix_t B;
    // bit_matrix_st C;
    factor_t *d_A;
    factor_t *d_B;
    bit_vector_t *d_C;
    float density_;
    int inverse_density_;
    int *distance_;
    int *d_distance_;
    // size_t height_padded;
    uint8_t factorDim_ = 20;
    size_t height_ = 0;
    size_t width_ = 0;
    size_t width_C_padded_ = 0;
    size_t lineSize_ = 1;
    size_t lineSize_padded_ = 1;
    int max_parallel_lines_;
};

#endif
