#ifndef cuBool_CPU_H
#define cuBool_CPU_H

#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#include <omp.h>

#include "helper/rngpu.hpp"
#include "helper/confusion.h"

#include "config.h"
#include "io_and_allocation.hpp"
#include "bit_vector_functions.h"
// #include "float_functions.h"

using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::min;

template<typename factor_t = uint32_t>
class cuBool
{
    using factor_matrix_t = vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = vector<bit_vector_t>;

    using index_t = uint32_t;
    using my_error_t = float;

    struct factor_handler {
        factor_matrix_t A_;
        factor_matrix_t B_;
        error_t distance_;
        uint8_t factorDim_ = 20;
        size_t lineSize_ = 1;
        bool initialized_ = false;
    };

public:
    cuBool(const bit_matrix_t& C,
          const index_t height,
          const index_t width,
          const float density,
          const size_t numActiveExperriments = 1)
    {
        cout << "~~~ CPU cuBool ~~~" << endl; 

        height_ = height;
        width_ = width;

        density_ = density;
        inverse_density_ = 1 / density;

        if(std::is_same<factor_t, uint32_t>::value) {
            lineSize_padded_ = 1;
        }
        else if(std::is_same<factor_t, float>::value) {
            lineSize_padded_ = 32;
        }

        max_parallel_lines_ = omp_get_max_threads();

        bestFactors = {};
        resetBest();
        cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;

        initializeMatrix(C);

        if(initialized_) {
            cout << "cuBool initialization complete." << endl;
        } else {
            exit(1);
        }
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << endl;
    }

    ~cuBool() = default;

    bool resetBest() {
        bestFactors.distance_ = std::numeric_limits<error_t>::max();
        bestFactors.initialized_ = false;
        return true;
    }

    // allocate memory for matrix, factors and distances, copy matrix to device
    bool initializeMatrix(const bit_matrix_t& C)
    {
        if( SDIV(height_,32) * width_ != C.size()) {
            cerr << "cuBool construction: Matrix dimension mismatch." << endl;
            return false;
        }

        C_ = C;

        // weights_rows_ = computeInverseDensitiesRows(C_, height_, width_);
        // weights_cols_ = computeInverseDensitiesCols(C_, height_, width_);

        // weights_rows_ = computeDensitiesRows(C_, height_, width_);
        // weights_cols_ = computeDensitiesCols(C_, height_, width_);

        // my_error_t max = 0;
        // my_error_t min = std::numeric_limits<float>::max();
        // for (const auto& w : weights_rows_) {
        //     // cout << w << " ";
        //     if(w > max) max = w;
        //     if(w < min) min = w;
        // }
        // cout << endl;
        // cout << "rows weight min: " << min << " weight max: " << max << endl;

        // max = 0;
        // min = std::numeric_limits<float>::max();
        // for (const auto& w : weights_cols_) {
        //     // cout << w << " ";
        //     if(w > max) max = w;
        //     if(w < min) min = w;
        // }
        // // cout << endl;
        // cout << "cols weight min: " << min << " weight max: " << max << endl;

        return initialized_ = true;
    }

public:
    // initialize factors with custom Initializer function
    template<class Initializer>
    bool initializeFactors(const uint8_t factorDim, Initializer&& initilize) {
        auto& handler = activeFactors;

        handler.factorDim_ = factorDim;

        if(std::is_same<factor_t, uint32_t>::value) {
            handler.lineSize_ = 1;
        }
        else if(std::is_same<factor_t, float>::value) {
            handler.lineSize_ = handler.factorDim_;
        }

        initilize(handler);

        handler.distance_ = -1;

        return handler.initialized_ = true;
    }

    // initialize factors as copy of vectors
    bool initializeFactors(const factor_matrix_t& A, const factor_matrix_t& B, const uint8_t factorDim) {
        return initializeFactors(factorDim, [&,this](factor_handler& handler){
            if( A.size() != height_ * handler.lineSize_ || B.size() != width_ * handler.lineSize_) {
                cerr << "cuBool initialization: Factor dimension mismatch." << endl;
                return false;
            }

            activeFactors.A_ = A;
            activeFactors.B_ = B;
        
        });
    }

    // initialize factors on device according to INITIALIZATIONMODE
    bool initializeFactors(const uint8_t factorDim, uint32_t seed) {
        return initializeFactors(factorDim, [&,this](factor_handler& handler){
            float threshold = getInitChance(density_, handler.factorDim_);
            // float threshold = 1.0f;
            initFactor(activeFactors.A_, height_, activeFactors.factorDim_, seed, threshold);
            seed += height_;
            initFactor(activeFactors.B_, width_, activeFactors.factorDim_, seed, threshold);
        });
    }

    // initialize first factor with random base vectors and optimize other factor
    bool initializeFactorsRandomBase(const uint8_t factorDim, uint32_t seed) {
        return initializeFactors(factorDim, [&,this](factor_handler& handler){
            float threshold = getInitChance(density_, handler.factorDim_);
            initFactor(activeFactors.A_, height_, activeFactors.factorDim_, seed, threshold);
            
            // activeFactors.A_ = factor_matrix_t(height_, 0);
            activeFactors.B_ = factor_matrix_t(width_, 0);
            // #pragma omp parallel
            for(int k=0; k<activeFactors.factorDim_; ++k) {
                // updateWholeColumn(activeFactors.A_, height_, activeFactors.factorDim_, k, density_, seed);
                optimizeWholeColumn<true>(activeFactors.B_, width_, activeFactors.A_, height_, C_, activeFactors.factorDim_, k);
            }
            activeFactors.distance_ = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
            cout << "Start distance: "
                      << "\tabs_err: " << activeFactors.distance_
                      << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                      << endl;

            // #pragma omp parallel
            for(int k=0; k<activeFactors.factorDim_; ++k) {
                // updateWholeColumn(activeFactors.A_, height_, activeFactors.factorDim_, k, density_, seed);
                optimizeWholeColumn<true>(activeFactors.B_, width_, activeFactors.A_, height_, C_, activeFactors.factorDim_, k);
            }
            activeFactors.distance_ = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
            cout << "Start distance: "
                      << "\tabs_err: " << activeFactors.distance_
                      << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                      << endl;
        });
    }

    bool verifyDistance() {
        if(!initialized_ && activeFactors.initialized_) {
            cerr << "cuBool not initialized." << endl;
            return false;
        }

        my_error_t distance_proof;

        distance_proof = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
        // distance_proof = computeDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_, weights_rows_, weights_cols_);

        bool equal = fabs(activeFactors.distance_- distance_proof) < 1e-3; // std::numeric_limits<float>::epsilon();
        if(!equal) {
            cout << "----- !Distances differ! -----\n";
            cout << "Running distance:  " << activeFactors.distance_ << "\n";
            cout << "Real distance:     " << distance_proof << endl;
        } else {
            cout << "Distance verified" << endl;
        }
        return equal;
    } 

    void getFactors(factor_matrix_t& A, factor_matrix_t& B) const {
        if(!activeFactors.initialized_) {
            cerr << "cuBool not initialized." << endl;
            return;
        }

        A = activeFactors.A_;
        B = activeFactors.B_;
    }

    my_error_t getDistance() const {
        if(!activeFactors.initialized_) {
            cerr << "cuBool not initialized." << endl;
            return -1;
        }
        return activeFactors.distance_;
    }

    void getBestFactors(factor_matrix_t& A, factor_matrix_t& B) const {
        if(!bestFactors.initialized_) {
            cerr << "cuBool not initialized." << endl;
            return;
        }

        A = bestFactors.A_;
        B = bestFactors.B_;
    }

    my_error_t getBestDistance() const {
        if(!bestFactors.initialized_) {
            cerr << "cuBool not initialized." << endl;
            return -1;
        }
        return bestFactors.distance_;
    }

    struct cuBool_config {
        size_t verbosity = 1;
        index_t linesAtOnce = 0;
        size_t maxIterations = 0;
        error_t distanceThreshold = std::numeric_limits<error_t>::min();
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

    void runMultiple(const size_t numExperiments, const cuBool_config& config) {
        finalDistances.resize(numExperiments);

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);

        for(size_t i=0; i<numExperiments; ++i) {
            auto config_i = config;
            uint32_t seed;
            seed = fast_kiss32(state);
            config_i.seed = fast_kiss32(state);
            cout << "Starting run " << i << " with seed " << config_i.seed << endl;
            initializeFactors(config_i.factorDim, seed);
            finalDistances[i] = run(config_i);
        }
    }

    float run(const cuBool_config& config) {
        if(!initialized_) {
            cerr << "cuBool not initialized." << endl;
            return -1;
        }

        if(!activeFactors.initialized_) {
            cerr << "cuBool active factors not initialized." << endl;
            return -1;
        }

        activeFactors.distance_ = computeDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_, config.weight);

        if(config.verbosity > 0) {
            cout << "\tStart distance"
                 << "\tabs_err: " << activeFactors.distance_
                 << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                 << '\n';
        }

        index_t linesAtOnce = config.linesAtOnce;
        if(config.loadBalance) {
            linesAtOnce = linesAtOnce / max_parallel_lines_ * max_parallel_lines_;
            if (!linesAtOnce) linesAtOnce = max_parallel_lines_;
        }

        if(config.verbosity > 1) {
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
            cout << "- - - - Starting " << config.maxIterations
                  << " CPU iterations, changing " << linesAtOnce
                  << " lines each time\n";
            cout << "- - - - Showing error every " << config.distanceShowEvery
                      << " steps\n";
            if(config.tempStart > 0) {
                cout << "- - - - Start temperature " << config.tempStart
                      << " multiplied by " << config.tempFactor
                      << " every " << config.tempStep
                      << " steps\n";

            }
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
                << endl;
        }

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
        float temperature = config.tempStart;
        float weight = config.weight;
        size_t iteration = 0;
        size_t stuckIterations = 0;
        auto distancePrev = activeFactors.distance_;
        index_t lineToBeChanged;
        uint32_t cpuSeed;
        // int all_true_positives = computeTruePositiveCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
        // int all_true_positives = computeTruePositiveCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
        confusion_matrix confusion = computeErrorsCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
        confusion_matrix confusion_new;

        factor_matrix_t A_new, B_new;

        #pragma omp parallel firstprivate(iteration)
        while( activeFactors.distance_ > config.distanceThreshold
                && iteration++ < config.maxIterations
                && temperature > config.tempEnd
                && stuckIterations < config.stuckIterationsBeforeBreak)
        {
            #pragma omp single
            {
                lineToBeChanged = (fast_kiss32(state) % height_);
                cpuSeed = fast_kiss32(state) + iteration;

                confusion_new.TP = 0;
                confusion_new.FP = 0;
                confusion_new.FN = 0;

                A_new = activeFactors.A_;
                B_new = activeFactors.B_;
            }
            uint8_t k = iteration % activeFactors.factorDim_;
            // Change rows
            // updateWholeColumn(A_new, height_, activeFactors.factorDim_, k, density_, cpuSeed);
            updateColumnPart(A_new, height_, activeFactors.factorDim_, k, density_,
                             lineToBeChanged, min(linesAtOnce, height_), cpuSeed);

            // Change cols
            auto confusion_update = optimizeWholeColumn<true>(B_new, width_, A_new, height_, C_, activeFactors.factorDim_, k);
            // implicit barrier

            #pragma omp atomic
            confusion_new.TP += confusion_update.TP;
            #pragma omp atomic
            confusion_new.FP += confusion_update.FP;
            #pragma omp atomic
            confusion_new.FN += confusion_update.FN;
            #pragma omp barrier

            #pragma omp single
            {
                // confusion_new = computeErrorsCPU(A_new, B_new, C_, height_, width_);

                if(confusion_new.total_error() < confusion.total_error()) {
                // if(confusion_new.precision() > confusion.precision()) {
                // if(confusion_new.jaccard() > confusion.jaccard()) {
                // if(metro(state, confusion.precision() - confusion_new.precision(), temperature)) {
                    activeFactors.A_ = A_new;
                    activeFactors.B_ = B_new;

                    confusion = confusion_new;

                    // cout << "update accepted" << endl;
                }

                // int hamming;
                // if(iteration % config.distanceShowEvery == 0) {
                    // activeFactors.distance_ = computeDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_, weights_rows_, weights_cols_);
                    // activeFactors.distance_ = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
                // }
                activeFactors.distance_ = confusion.total_error();

                if(config.verbosity > 1 && iteration % config.distanceShowEvery == 0) {
                    cout << "Iteration: " << iteration
                              << "\tTP: " << confusion.TP
                              // << "\terrors: " << confusion.total_error()
                              // << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                              << "\thamming: " << activeFactors.distance_
                              << "\ttemp: " << temperature;
                    cout << endl;
                }
                if(iteration % config.tempStep == 0) {
                    temperature *= config.tempFactor;
                    if(weight > 1)
                        weight *= config.tempFactor;
                    if(weight < 1)
                        weight = 1;
                }
                if(activeFactors.distance_ == distancePrev)
                    stuckIterations++;
                else
                    stuckIterations = 0;
                distancePrev = activeFactors.distance_;
            }
        }

        if(config.verbosity > 0) {
            cout << "\tBreak condition:\t";
            if (!(iteration < config.maxIterations))
                cout << "Reached iteration limit: " << config.maxIterations;
            if (!(activeFactors.distance_ > config.distanceThreshold))
                cout << "Distance below threshold: " << config.distanceThreshold;
            if (!(temperature > config.tempEnd))
                cout << "Temperature below threshold";
            if (!(stuckIterations < config.stuckIterationsBeforeBreak))
                cout << "Stuck for " << stuckIterations << " iterations";
            cout << " after " << iteration << " iterations.\n";
        }

        // use hamming distance for final judgement
        activeFactors.distance_ = computeDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_, 1);

        if(config.verbosity > 0) {
            cout << "\tFinal distance"
                 << "\tabs_err: " << activeFactors.distance_
                 << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                 << endl;
        }

        if(activeFactors.distance_ < bestFactors.distance_) {
                if(config.verbosity > 0) {
                    cout << "\tResult is better than previous best. Saving new best." << endl;
                }

                bestFactors = activeFactors;
        }

        return float(activeFactors.distance_) / height_ / width_;
    }  

    const vector<float>& getDistances() const {
        return finalDistances;
    }

private:
    bool initialized_ = false;

    bit_matrix_t C_;
    vector<my_error_t> weights_rows_;
    vector<my_error_t> weights_cols_;
    float density_;
    int inverse_density_;

    index_t height_ = 0;
    index_t width_ = 0;
    size_t lineSize_padded_ = 1;

    int max_parallel_lines_;

    factor_handler bestFactors;
    factor_handler activeFactors;

    vector<float> finalDistances;
};

#endif
