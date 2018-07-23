#ifndef cuBool_CPU_H
#define cuBool_CPU_H

#include <vector>
#include <iostream>
#include <limits>
#include <cmath>

#include "helper/rngpu.hpp"
#include "helper/confusion.h"

#include "config.h"
#include "bit_vector_functions.h"
// #include "float_functions.h"

using std::vector;
using std::cout;
using std::cerr;
using std::endl;


template<typename factor_t = uint32_t>
class cuBool_CPU
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
    cuBool_CPU(const bit_matrix_t& C,
          const index_t height,
          const index_t width,
          const float density)
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

    ~cuBool_CPU() {}

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

        return initialized_ = true;
    }

public:
    bool initializeFactors(const factor_matrix_t& A, const factor_matrix_t& B, const uint8_t factorDim) {
        activeFactors.factorDim_ = factorDim;

        if (std::is_same<factor_t, uint32_t>::value) {
            activeFactors.lineSize_ = 1;
        }
        if(std::is_same<factor_t, float>::value) {
            activeFactors.lineSize_ = activeFactors.factorDim_;
        }

        // if( SDIV(A.size()/lineSize_,32) * B.size()/lineSize_ != C_.size()) {
        if( A.size() != height_ * activeFactors.lineSize_ || B.size() != width_ * activeFactors.lineSize_) {
            cerr << "cuBool construction: Factor dimension mismatch." << endl;
            return false;
        }
        
        activeFactors.A_ = A;
        activeFactors.B_ = B;

        weights_rows_ = computeInverseDensitiesRows(C_, height_, width_);
        weights_cols_ = computeInverseDensitiesCols(C_, height_, width_);

        // weights_rows_ = computeDensitiesRows(C_, height_, width_);
        // weights_cols_ = computeDensitiesCols(C_, height_, width_);

        my_error_t max = 0;
        my_error_t min = std::numeric_limits<float>::max();
        for (const auto& w : weights_rows_) {
            // cout << w << " ";
            if(w > max) max = w;
            if(w < min) min = w;
        }
        // cout << endl;
        cout << "rows weight min: " << min << " weight max: " << max << endl;

        max = 0;
        min = std::numeric_limits<float>::max();
        for (const auto& w : weights_cols_) {
            // cout << w << " ";
            if(w > max) max = w;
            if(w < min) min = w;
        }
        // cout << endl;
        cout << "cols weight min: " << min << " weight max: " << max << endl;

        activeFactors.activeFactors.B_ = factor_matrix_t(height_, 0);
        for(int k=0; k<activeFactors.factorDim_; ++k) {
            // updateWholeColumn(activeFactors.A_, height_, activeFactors.factorDim_, k, density_, seed);
            optimizeWholeColumn<true>(activeFactors.B_, width_, activeFactors.A_, height_, C_, activeFactors.factorDim_, k);
        }
        activeFactors.distance_ = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
        cout << "Start distance: "
                  << "\tabs_err: " << activeFactors.distance_
                  << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                  << endl;

        for(int k=0; k<activeFactors.factorDim_; ++k) {
            // updateWholeColumn(activeFactors.A_, height_, activeFactors.factorDim_, k, density_, seed);
            optimizeWholeColumn<true>(activeFactors.B_, width_, activeFactors.A_, height_, C_, activeFactors.factorDim_, k);
        }
        activeFactors.distance_ = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
        // activeFactors.distance_ = computeDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_, weights_rows_, weights_cols_);

        cout << "cuBool initialization complete." << endl;

        cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;
        cout << "Factor dimension:\t" << int(activeFactors.factorDim_) << endl;

        cout << "Start distance: "
                  << "\tabs_err: " << activeFactors.distance_
                  << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                  << endl;

        return initialized_ = true;
    }

    bool verifyDistance() {
        if(!initialized_) {
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
        }
        return equal;
    } 

    void getFactors(factor_matrix_t& A, factor_matrix_t& B) {
        if(!initialized_) {
            cerr << "cuBool not initialized." << endl;
            return;
        }

        A = activeFactors.A_;
        B = activeFactors.B_;
    }

    my_error_t getDistance() {
        if(!initialized_) {
            cerr << "cuBool not initialized." << endl;
            return -1;
        }
        return activeFactors.distance_;
    }

    struct cuBool_config {
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

    void run(const cuBool_config& config) {
        if(!initialized_) {
            cerr << "cuBool not initialized." << endl;
            return;
        }

        size_t linesAtOnce = config.linesAtOnce;

        if(config.verbosity > 0) {
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
            cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -";
            cout << endl;
        }

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
        // float temperature = 0;
        float temperature = config.tempStart;
        size_t iteration = 0;
        size_t stuckIterations = 0;
        auto distancePrev = activeFactors.distance_;
        // my_error_t tempStep_distance = 1;
        // my_error_t update_sum = 0;
        int lineToBeChanged;
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

                if(config.verbosity > 0 && iteration % config.distanceShowEvery == 0) {
                    cout << "Iteration: " << iteration
                              // << "\tupdate: " << update_sum / config.distanceShowEvery
                              << "\tTP: " << confusion.TP
                              // << "\terrors: " << confusion.total_error()
                              // << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                              << "\thamming: " << activeFactors.distance_
                              << "\ttemp: " << temperature;
                    cout << endl;

                    // cout << "\tseed: " << (int) cpuSeed << endl;
                    // update_sum = 0;

                    // tempStep_distance = activeFactors.distance_;
                }
                if(iteration % config.tempStep == 0) {
                    //delay temperature
                    // if(temperature <= 0) temperature = config.tempStart;
                    temperature *= config.tempFactor;
                    // if((float) activeFactors.distance_ / tempStep_distance < 0.9f)
                    //     temperature /= config.tempFactor;
                    // else
                    //     temperature *= config.tempFactor;
                    // tempStep_distance = activeFactors.distance_;
                }
                if(activeFactors.distance_ == distancePrev)
                    stuckIterations++;
                else
                    stuckIterations = 0;
                distancePrev = activeFactors.distance_;
            }
        }

        if(config.verbosity > 0) {
            if (!(iteration < config.maxIterations))
                cout << "Reached iteration limit: " << config.maxIterations << endl;
            if (!(activeFactors.distance_ > config.distanceThreshold))
                cout << "Distance below threshold." << endl;
            if (!(temperature > config.tempEnd))
                cout << "Temperature below threshold." << endl;
            if (!(stuckIterations < config.stuckIterationsBeforeBreak))
                cout << "Stuck for " << stuckIterations << " iterations." << endl;
        }
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        cout << "Final result: "
                  << "\tabs_err: " << activeFactors.distance_
                  << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                  << endl;
    }  

private:
    bool initialized_ = false;

    bit_matrix_t C_;
    vector<my_error_t> weights_rows_;
    vector<my_error_t> weights_cols_;
    float density_;
    int inverse_density_;

    size_t height_ = 0;
    size_t width_ = 0;
    size_t lineSize_padded_ = 1;

    factor_handler bestFactors;
    factor_handler activeFactors;

    vector<float> finalDistances;
};

#endif
