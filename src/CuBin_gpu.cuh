#ifndef CUBIN_GPU_CUH
#define CUBIN_GPU_CUH

#include <vector>
#include <iostream>
#include <sstream>
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
using std::ostringstream;
using std::vector;

template<typename factor_t = uint32_t>
class CuBin
{
    using factor_matrix_t = vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = vector<bit_vector_t>;

    using index_t = uint32_t;
    using error_t = int;

    struct factor_handler {
        factor_t *d_A;
        factor_t *d_B;
        error_t *distance_;
        error_t *d_distance_;
        uint8_t factorDim_ = 20;
        size_t lineSize_ = 1;
        bool initialized_ = false;
    };

public:
    CuBin(const bit_matrix_t& C,
          const index_t height,
          const index_t width,
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
    bool initializeMatrix(const bit_matrix_t& C, const index_t height, const index_t width)
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
        // index_t height_padded = SDIV(height_, WARPSPERBLOCK) * WARPSPERBLOCK;
        for(auto& e : activeExperiments) {
            cudaMalloc(&e.d_A, lineBytes_padded * height_); CUERR
        }
        cudaMallocHost(&bestFactors.d_A, lineBytes_padded * height_); CUERR

        width_ = width;
        // index_t width_padded = SDIV(width_, WARPSPERBLOCK) * WARPSPERBLOCK;
        for(auto& e : activeExperiments) {
            cudaMalloc(&e.d_B, lineBytes_padded * width_); CUERR
        }
        cudaMallocHost(&bestFactors.d_B, lineBytes_padded * width_); CUERR
        
        index_t height_C = SDIV(height_, 32);
        width_C_padded_ = SDIV(width_, 32) * 32;
        cudaMalloc(&d_C, sizeof(bit_vector_t) * height_C * width_C_padded_); CUERR

        cudaMemcpy2D(d_C, sizeof(bit_vector_t) * width_C_padded_,
                     C.data(), sizeof(bit_vector_t) * width_,
                     sizeof(bit_vector_t) * width_,
                     height_C,
                     cudaMemcpyHostToDevice); CUERR

        // cudaMemcpyToSymbol(height_c, &height_, sizeof(index_t));
        // cudaMemcpyToSymbol(width_c, &width_, sizeof(index_t));
        // cudaMemcpyToSymbol(width_padded_c, &width_C_padded_, sizeof(index_t));
        // cudaMemcpyToSymbol(inverse_density_c, &inverse_density_, sizeof(int));

        for(auto& e : activeExperiments) {
            cudaMallocHost(&e.distance_, sizeof(error_t)); CUERR
            cudaMalloc(&e.d_distance_, sizeof(error_t)); CUERR
            cudaMemset(e.d_distance_, 0, sizeof(error_t)); CUERR
        }
        cudaMallocHost(&bestFactors.distance_, sizeof(error_t)); CUERR
        *bestFactors.distance_ = std::numeric_limits<error_t>::max();

        cout << "CuBin initialization complete." << endl;

        cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;

        return initialized_ = true;
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

private:
    void calculateDistance(const factor_handler& handler, const int weight = 1, const cudaStream_t stream = 0) {
        cudaMemsetAsync(handler.d_distance_, 0, sizeof(error_t), stream);
        // cudaStreamSynchronize(stream); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                        (handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_,
                         handler.factorDim_, weight, handler.d_distance_);
        // cudaStreamSynchronize(stream); CUERR

        cudaMemcpyAsync(handler.distance_, handler.d_distance_, sizeof(error_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); CUERR
    }

public:
    // initialize factors as copy of host vectors
    bool initializeFactors(const size_t activeId,
                           const factor_matrix_t& A,
                           const factor_matrix_t& B,
                           const uint8_t factorDim,
                           const cudaStream_t stream = 0)
    {
        return initializeFactors(activeId, factorDim, [&,this](factor_handler& handler){
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
        });
    }

    // initialize factors on device according to INITIALIZATIONMODE
    bool initializeFactors(const size_t activeId, const uint8_t factorDim, uint32_t seed, const cudaStream_t stream = 0) {
        return initializeFactors(activeId, factorDim, [&,this](factor_handler& handler){
            float threshold = getInitChance(density_, handler.factorDim_);

            initFactor <<< SDIV(height_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32, 0, stream >>>
                        (handler.d_A, height_, handler.factorDim_, seed, threshold);
            // cudaStreamSynchronize(stream); CUERR

            seed += height_;
            initFactor <<< SDIV(width_, WARPSPERBLOCK*32/lineSize_padded_), WARPSPERBLOCK*32, 0, stream >>>
                        (handler.d_B, width_, handler.factorDim_, seed, threshold);
            // cudaStreamSynchronize(stream); CUERR
        });
    }

    template<class Initializer>
    bool initializeFactors(const size_t activeId, const uint8_t factorDim, Initializer&& initilize, const cudaStream_t stream = 0) {
        auto& handler = activeExperiments[activeId];

        handler.factorDim_ = factorDim;

        if(std::is_same<factor_t, uint32_t>::value) {
            handler.lineSize_ = 1;
        }
        else if(std::is_same<factor_t, float>::value) {
            handler.lineSize_ = handler.factorDim_;
        }

        initilize(handler);

        return handler.initialized_ = true;
    }

    bool verifyDistance(const size_t activeId, const int weight = 1) {
        auto& handler = activeExperiments[activeId];

        if(!initialized_) {
            cerr << "CuBin not initialized." << endl;
            return false;
        }

        error_t* distance_proof;
        error_t* d_distance_proof;

        cudaMallocHost(&distance_proof, sizeof(error_t)); CUERR
        cudaMalloc(&d_distance_proof, sizeof(error_t)); CUERR
        cudaMemset(d_distance_proof, 0, sizeof(error_t)); CUERR

        computeDistanceRowsShared <<< SDIV(height_, WARPSPERBLOCK), WARPSPERBLOCK*32 >>>
                        (handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_,
                         handler.factorDim_, weight, d_distance_proof);

        cudaDeviceSynchronize(); CUERR

        cudaMemcpy(distance_proof, d_distance_proof, sizeof(error_t), cudaMemcpyDeviceToHost); CUERR

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

    void getFactors(const size_t activeId, factor_matrix_t& A, factor_matrix_t& B, const cudaStream_t stream = 0) const {
        auto& handler = activeExperiments[activeId];

        if(!initialized_ || !handler.initialized_) {
            cerr << "Factors in slot " << activeId << " not initialized." << endl;
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

    error_t getDistance(const size_t activeId) const {
        auto& handler = activeExperiments[activeId];

        if(!initialized_ || !handler.initialized_) {
            cerr << "Factors in slot " << activeId << " not initialized." << endl;
            return -1;
        }
        return *handler.distance_;
    }

    void getBestFactors(factor_matrix_t& A, factor_matrix_t& B, const cudaStream_t stream = 0) const {
        if(!initialized_ || !bestFactors.initialized_) {
            cerr << "Best result not initialized." << endl;
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

    error_t getBestDistance() const {
        if(!initialized_ || !bestFactors.initialized_) {
            cerr << "Best result not initialized." << endl;
            return -1;
        }
        return *bestFactors.distance_;
    }

    struct CuBin_config {
        size_t verbosity = 1;
        index_t linesAtOnce = 0;
        size_t maxIterations = 0;
        error_t distanceThreshold = 0;
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
        uint8_t factorDim = 20;
        int weight = 1;
    };

    void runAllParallel(const size_t numExperiments, const CuBin_config& config) {
        finalDistances.resize(numExperiments);

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
        // uint8_t factorDim = 20;
        // uint32_t seed = 123;

        // cudaMemcpyToSymbol(flipManyChance_c, &config.flipManyChance, sizeof(float));
        // cudaMemcpyToSymbol(flipManyDepth_c, &config.flipManyDepth, sizeof(uint32_t));
        // cudaMemcpyToSymbol(factorDim_c, &config.factorDim, sizeof(uint8_t));

        #pragma omp parallel for schedule(dynamic,1)
        for(size_t i=0; i<numExperiments; ++i) {
            unsigned id = omp_get_thread_num();
            auto config_i = config;
            uint32_t seed = fast_kiss32(state);
            config_i.seed = fast_kiss32(state);
            #pragma omp critical
            cout << "Starting run " << i << " in slot " << id << " with seed " << config_i.seed << endl;
            initializeFactors(id, config_i.factorDim, seed);
            finalDistances[i] = run(id, config_i);
        }
    }

    float run(const size_t activeId, const CuBin_config& config, const cudaStream_t stream = 0) {
        auto& handler = activeExperiments[activeId];

        if(!initialized_ || !handler.initialized_) {
            cerr << "CuBin not initialized." << endl;
            return -1;
        }

        ostringstream out;

        calculateDistance(handler, config.weight, stream);

        if(config.verbosity > 0) {
            out << "\tStart distance for slot " << activeId
                 << "\tabs_err: " << *handler.distance_
                 << "\trel_err: " << (float) *handler.distance_ / (height_ * width_)
                 << '\n';
        }

        index_t linesAtOnce = SDIV(config.linesAtOnce, WARPSPERBLOCK) * WARPSPERBLOCK;
        if(config.loadBalance) {
            linesAtOnce = linesAtOnce / max_parallel_lines_ * max_parallel_lines_;
            if (!linesAtOnce) linesAtOnce = max_parallel_lines_;
        }

        if(config.verbosity > 1) {
            out << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
            out << "- - - - Starting " << config.maxIterations
                 << " GPU iterations, changing " << linesAtOnce
                 << " lines each time\n";
            out << "- - - - Showing error every " << config.distanceShowEvery
                 << " steps\n";
            if(config.tempStart > 0) {
                out << "- - - - Start temperature " << config.tempStart
                     << " multiplied by " << config.tempFactor
                     << " every " << config.tempStep
                     << " steps\n";
                out << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
                     << endl;
            }
        }

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
        float temperature = config.tempStart;
        size_t iteration = 0;
        size_t stuckIterations = 0;
        auto distancePrev = *handler.distance_;
        size_t syncStep = 100;
        while(
                *handler.distance_ > config.distanceThreshold &&
                iteration++ < config.maxIterations
                && temperature > config.tempEnd
                && stuckIterations < config.stuckIterationsBeforeBreak) {

            // Change rows
            index_t lineToBeChanged = (fast_kiss32(state) % height_) / WARPSPERBLOCK * WARPSPERBLOCK;
            uint32_t gpuSeed = fast_kiss32(state) + iteration;

            vectorMatrixMultCompareRowWarpShared
                <<< SDIV(min(linesAtOnce, height_), WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                (handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_, handler.factorDim_,
                 lineToBeChanged, handler.d_distance_, gpuSeed, temperature/10,
                 config.flipManyChance, config.flipManyDepth, config.weight);
            // cudaStreamSynchronize(stream); CUERR

            // Change cols
            lineToBeChanged = (fast_kiss32(state) % width_) / WARPSPERBLOCK * WARPSPERBLOCK;
            gpuSeed = fast_kiss32(state) + iteration;

            vectorMatrixMultCompareColWarpShared
                <<< SDIV(min(linesAtOnce, width_), WARPSPERBLOCK), WARPSPERBLOCK*32, 0, stream >>>
                (handler.d_A, handler.d_B, d_C, height_, width_, width_C_padded_, handler.factorDim_,
                 lineToBeChanged, handler.d_distance_, gpuSeed, temperature/10,
                 config.flipManyChance, config.flipManyDepth, config.weight);
            cudaStreamSynchronize(stream); CUERR

            if(iteration % syncStep == 0) {
                cudaMemcpyAsync(handler.distance_, handler.d_distance_, sizeof(error_t), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream); CUERR

                if(*handler.distance_ == distancePrev)
                    stuckIterations += syncStep;
                else
                    stuckIterations = 0;
                distancePrev = *handler.distance_;
            }

            if(config.verbosity > 1 && iteration % config.distanceShowEvery == 0) {
                out << "Iteration: " << iteration
                     << "\tabs_err: " << *handler.distance_
                     << "\trel_err: " << float(*handler.distance_) / (height_*width_)
                     << "\ttemp: " << temperature;
                out << endl;
            }
            if(iteration % config.tempStep == 0) {
                temperature *= config.tempFactor;
            }
        }

        if(config.verbosity > 0) {
            out << "\tBreak condition for slot " << activeId << ": ";
            if (!(iteration < config.maxIterations))
                out << "Reached iteration limit: " << config.maxIterations << '\n';
            if (!(*handler.distance_ > config.distanceThreshold))
                out << "Distance below threshold.\n";
            if (!(temperature > config.tempEnd))
                out << "Temperature below threshold.\n";
            if (!(stuckIterations < config.stuckIterationsBeforeBreak))
                out << "Stuck for " << stuckIterations << " iterations.\n";
            // out << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
            out << "\tFinal distance for slot " << activeId
                 << "\tabs_err: " << *handler.distance_
                 << "\trel_err: " << float(*handler.distance_) / (height_ * width_)
                 << endl;
        }

        if(*handler.distance_ < *bestFactors.distance_) {

            #pragma omp critical
            if(*handler.distance_ < *bestFactors.distance_) {
                if(config.verbosity > 0) {
                    out << "\tResult is better than previous best. Copying to host." << endl;
                }

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

                bestFactors.initialized_ = true;
            }
        }
        // cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << endl;

        #pragma omp critical
        cout << out.str();

        return float(*handler.distance_) / (height_ * width_);
    }

    const vector<float>& getDistances() const {
        return finalDistances;
    }

private:
    bool initialized_ = false;

    bit_vector_t *d_C;
    float density_;
    int inverse_density_;
    index_t height_;
    index_t width_;
    index_t width_C_padded_;
    size_t lineSize_padded_;
    int max_parallel_lines_;

    factor_handler bestFactors;
    vector<factor_handler> activeExperiments;

    vector<float> finalDistances;
};

#endif
