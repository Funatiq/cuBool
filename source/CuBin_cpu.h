#ifndef CUBIN_CPU_H
#define CUBIN_CPU_H

#include <vector>
#include <iostream>
#include <cmath>

using std::vector;

template<typename bit_vector_t>
int computeHammingDistanceCPU(const vector<bit_vector_t> &Ab,
                        const vector<bit_vector_t> &Bb,
                        const vector<bit_vector_t> &Cb,
                        const int height,
                        const int width)
{
    int error = 0;

    #pragma omp parallel for reduction(+:error)
    for(int j=0; j < width; ++j) {
        uint32_t B_j = Bb[j];
        for(int i=0; i < height; ++i) {
            const int product = (Ab[i] & B_j) ? 1 : 0;

            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            error += product ^ C_ij;
        }
    }

    return error;
}

template<typename bit_vector_t>
void computeErrorsCPU(const vector<bit_vector_t> &Ab,
                        const vector<bit_vector_t> &Bb,
                        const vector<bit_vector_t> &Cb,
                        const int height,
                        const int width)
{
    int true_positives = 0;
    int true_negatives = 0;
    int false_positives = 0;
    int false_negatives = 0;

    #pragma omp parallel for reduction(+:true_positives) reduction(+:true_negatives) reduction(+:false_positives) reduction(+:false_negatives)
    for(int j=0; j < width; ++j) {
        uint32_t B_j = Bb[j];
        for(int i=0; i < height; ++i) {
            const int product = (Ab[i] & B_j) ? 1 : 0;

            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            if(product) {
                if(C_ij)
                    true_positives++;
                else
                    false_positives++;
            } else {
                if(C_ij)
                    false_negatives++;
                else
                    true_negatives++;
            }
        }
    }

    std::cout << "true_positives: " << true_positives << endl;
    std::cout << "true_negatives: " << true_negatives << endl;
    std::cout << "false_positives: " << false_positives << endl;
    std::cout << "false_negatives: " << false_negatives << endl;
    std::cout << "total error: " << false_positives + false_negatives << endl;
}

template<typename bit_vector_t, typename error_t>
error_t computeDistanceCPU(const vector<bit_vector_t> &Ab,
                           const vector<bit_vector_t> &Bb,
                           const vector<bit_vector_t> &Cb,
                           const int height,
                           const int width,
                           const vector<error_t>& weights_rows,
                           const vector<error_t>& weights_cols)
{
    error_t error = 0;

    #pragma omp parallel for reduction(+:error)
    for(int i=0; i < height; ++i) {
        uint32_t A_i = Ab[i];
        for(int j=0; j < width; ++j) {
            const int product = (A_i & Bb[j]) ? 1 : 0;

            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            const error_t weight = weights_rows[i] + weights_cols[j];

            error += error_measure(product, C_ij, weight);
        }
    }

    return error;
}

template<typename bit_vector_t, typename error_t = float>
vector<error_t> computeDensitiesRows(const vector<bit_vector_t> &Cb,
                                            const int height,
                                            const int width)
{
    vector<error_t> density_rows(height);

    #pragma omp parallel for
    for(int i=0; i<height; ++i) {
        int nonZeroCount = 0;
        for(int j=0; j<width; ++j) {
            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            nonZeroCount += C_ij;
        }
        density_rows[i] = (error_t) nonZeroCount / width;
    }

    return density_rows;
}

template<typename bit_vector_t, typename error_t = float>
vector<error_t> computeDensitiesCols(const vector<bit_vector_t> &Cb,
                                        const int height,
                                        const int width)
{
    vector<error_t> density_cols(width);

    #pragma omp parallel for
    for(int j=0; j<width; ++j) {
        int nonZeroCount = 0;
        for(int i=0; i<height; ++i) {
            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            nonZeroCount += C_ij;
        }
        density_cols[j] = (error_t) nonZeroCount / height;
    }

    return density_cols;
}

template<typename bit_vector_t, typename error_t = float>
vector<error_t> computeInverseDensitiesRows(const vector<bit_vector_t> &Cb,
                                            const int height,
                                            const int width)
{
    vector<error_t> inverse_density_rows(height);

    #pragma omp parallel for
    for(int i=0; i<height; ++i) {
        int nonZeroCount = 0;
        for(int j=0; j<width; ++j) {
            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            nonZeroCount += C_ij;
        }
        inverse_density_rows[i] = (error_t) width / nonZeroCount;
    }

    return inverse_density_rows;
}

template<typename bit_vector_t, typename error_t = float>
vector<error_t> computeInverseDensitiesCols(const vector<bit_vector_t> &Cb,
                                        const int height,
                                        const int width)
{
    vector<error_t> inverse_density_cols(width);

    #pragma omp parallel for
    for(int j=0; j<width; ++j) {
        int nonZeroCount = 0;
        for(int i=0; i<height; ++i) {
            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            nonZeroCount += C_ij;
        }
        inverse_density_cols[j] = (error_t) height / nonZeroCount;
    }

    return inverse_density_cols;
}

template<bool transpose, typename bit_vector_t, typename error_t>
int vectorMatrixMultCompareLineCPU(vector<bit_vector_t> &Ab,
                                   const int size_A,
                                   const vector<bit_vector_t> &Bb,
                                   const int size_B,
                                   const vector<bit_vector_t> &Cb,
                                   const uint8_t factorDim,
                                   const int startline,
                                   const int numlines,
                                   const uint32_t seed, 
                                   const float temperature,
                                   const float flipManyChance,
                                   const uint32_t flipManyDepth,
                                   const vector<error_t>& weights_rows,
                                   const vector<error_t>& weights_cols)
{
    error_t error_update = 0;

    // // const uint8_t numTests = factorDim+1;
    const uint8_t numTests = factorDim;
    // // const uint8_t numTests = 1;
    bit_vector_t A_i_tests[numTests];
    error_t error_tests[numTests];

    #pragma omp for
    // #pragma omp parallel for reduction(+:error_update)
    for(int id=0; id < numlines; ++id) {
        const int i = (startline + id) % size_A;

        fast_kiss_state32_t state;
        state = get_initial_fast_kiss_state32(seed + id);

        const bit_vector_t A_i = Ab[i];
        // bit_vector_t A_i_changed = Ab[i] ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);

        for(int k=0; k<factorDim; ++k) {
            A_i_tests[k] = A_i ^ (1 << k);
            error_tests[k] = 0;
            // A_i_tests[k] = A_i ^ get_flip_mask_many(factorDim, state, flipManyDepth);
        //     error_tests[k] = 0;
        }
        // A_i_tests[numTests] = A_i ^ get_flip_mask_many(factorDim, state, flipManyDepth);
        // A_i_tests[numTests] = fast_kiss32(state) >> (32 - factorDim);
        // error_tests[numTests] = 0;

        // error_t error = 0;
        for(int j=0; j < size_B; ++j) {
            const int vecId = transpose ? j / 32 * size_A + i : i / 32 * size_B + j;
            const int vecLane = transpose ? j % 32 : i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;
            
            // const error_t weight = (weights_rows[i] + weights_cols[j]) / 2;
            const error_t weight = weights_rows[i];

            const int product_old = (A_i         & Bb[j]) ? 1 : 0;
            // const int product_new = (A_i_changed & Bb[j]) ? 1 : 0;

            for(int k=0; k<numTests; ++k) {
                const int product_new = (A_i_tests[k] & Bb[j]) ? 1 : 0;

                error_tests[k] += error_measure(product_new, C_ij, weight)
                                - error_measure(product_old, C_ij, weight);
            }

            // error += error_measure(product_new, C_ij, weight)
            //        - error_measure(product_old, C_ij, weight);
        }

        error_t error = std::numeric_limits<float>::max();
        bit_vector_t A_i_changed;
        for(int k=0; k<numTests; ++k) {
            if(error_tests[k] != 0 && error_tests[k] < error) {
                error = error_tests[k];
                A_i_changed = A_i_tests[k];
            }
        }
        // if(error == INT_MAX) {
        if(error > 0) {
            const uint32_t k = fast_kiss32(state) % numTests;
            A_i_changed = A_i_tests[k];
            error = error_tests[k];
        }

        if (metro(state, error, temperature, size_B)) {
            Ab[i] = A_i_changed;
            error_update += error;
        }
    }

    return error_update;
}

// template<typename bit_vector_t>
// int vectorMatrixMultCompareRowCPU(vector<bit_vector_t> &Ab,
//                                   const vector<bit_vector_t> &Bb,
//                                   const vector<bit_vector_t> &Cb,
//                                   const int height,
//                                   const int width,
//                                   const uint8_t factorDim,
//                                   const int startrow,
//                                   const int numrows,
//                                   const uint32_t seed, 
//                                   const float temperature,
//                                   const float flipManyChance,
//                                   const uint32_t flipManyDepth,
//                                   const int inverse_density)
// {
//     int error_update = 0;

//     // const uint8_t numTests = factorDim+1;
//     const uint8_t numTests = factorDim;
//     // const uint8_t numTests = 1;
//     bit_vector_t A_i_tests[numTests];
//     int error_tests[numTests];

//     #pragma omp for
//     // #pragma omp parallel for reduction(+:error_update)
//     for(int id=0; id < numrows; ++id) {
//         const int i = (startrow + id) % height;

//         fast_kiss_state32_t state;
//         state = get_initial_fast_kiss_state32(seed + id);

//         bit_vector_t A_i = Ab[i];
//         // bit_vector_t A_i_changed = Ab[i] ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
//         for(int k=0; k<factorDim; ++k) {
//             // A_i_tests[k] = A_i ^ (1 << k);
//             // error_tests[k] = 0;
//             A_i_tests[k] = A_i ^ get_flip_mask_many(factorDim, state, flipManyDepth);
//             error_tests[k] = 0;
//         }
//         // A_i_tests[numTests] = A_i ^ get_flip_mask_many(factorDim, state, flipManyDepth);
//         // A_i_tests[numTests] = fast_kiss32(state) >> (32 - factorDim);
//         // error_tests[numTests] = 0;

//         // int error = 0;
//         for(int j=0; j < width; ++j) {
//             // const int product_new = (A_i_changed & Bb[j]) ? 1 : 0;
//             const int product_old = (A_i         & Bb[j]) ? 1 : 0;

//             const int vecId = i / 32 * width + j;
//             const int vecLane = i % 32;
//             const int C_ij = (Cb[vecId] >> vecLane) & 1;

//             for(int k=0; k<numTests; ++k) {
//                 const int product_new = (A_i_tests[k] & Bb[j]) ? 1 : 0;
//                 error_tests[k] += error_measure(product_new, C_ij, inverse_density)
//                              - error_measure(product_old, C_ij, inverse_density);
//             }

//             // error += error_measure(product_new, C_ij, inverse_density)
//                    // - error_measure(product_old, C_ij, inverse_density);
//         }

//         int error = INT_MAX;
//         bit_vector_t A_i_changed;
//         for(int k=0; k<numTests; ++k) {
//             if(error_tests[k] != 0 && error_tests[k] < error) {
//                 error = error_tests[k];
//                 A_i_changed = A_i_tests[k];
//             }
//         }
//         // if(error == INT_MAX) {
//         if(error > 0) {
//             const uint32_t k = fast_kiss32(state) % numTests;
//             A_i_changed = A_i_tests[k];
//             error = error_tests[k];
//         }

//         if (metro(state, error, temperature, width)) {
//             Ab[i] = A_i_changed;
//             error_update += error;
//         }
//     }

//     return error_update;
// }

// template<typename bit_vector_t>
// int vectorMatrixMultCompareColCPU(const vector<bit_vector_t> &Ab,
//                                   vector<bit_vector_t> &Bb,
//                                   const vector<bit_vector_t> &Cb,
//                                   const int height,
//                                   const int width,
//                                   const uint8_t factorDim,
//                                   const int startcol,
//                                   const int numcols,
//                                   const uint32_t seed, 
//                                   const float temperature,
//                                   const float flipManyChance,
//                                   const uint32_t flipManyDepth,
//                                   const int inverse_density)
// {
//     int error_update = 0;

//     // const uint8_t numTests = factorDim+1;
//     const uint8_t numTests = factorDim;
//     bit_vector_t B_j_tests[numTests];
//     int error_tests[numTests];

//     #pragma omp for
//     // #pragma omp parallel for reduction(+:error_update)
//     for(int id=0; id < numcols; ++id) {
//         const int j = (startcol + id) % width;

//         fast_kiss_state32_t state;
//         state = get_initial_fast_kiss_state32(seed + id);

//         bit_vector_t B_j = Bb[j];
//         // bit_vector_t B_j_changed = Bb[j] ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
//         for(int k=0; k<factorDim; ++k) {
//             // B_j_tests[k] = B_j ^ (1 << k);
//             // error_tests[k] = 0;
//             B_j_tests[k] = B_j ^ get_flip_mask_many(factorDim, state, flipManyDepth);
//             error_tests[k] = 0;
//         }
//         // B_j_tests[factorDim] = fast_kiss32(state) >> (32 - factorDim);
//         // error_tests[factorDim] = 0;

//         // int error = 0;
//         for(int i=0; i < height; ++i) {
//             // const int product_new = (Ab[i] & B_j_changed) ? 1 : 0;
//             const int product_old = (Ab[i] & B_j        ) ? 1 : 0;

//             const int vecId = i / 32 * width + j;
//             const int vecLane = i % 32;
//             const int C_ij = (Cb[vecId] >> vecLane) & 1;

//             for(int k=0; k<numTests; ++k) {
//                 const int product_new = (Ab[i] & B_j_tests[k]) ? 1 : 0;
//                 error_tests[k] += error_measure(product_new, C_ij, inverse_density)
//                              - error_measure(product_old, C_ij, inverse_density);
//             }

//             // error += error_measure(product_new, C_ij, inverse_density)
//                    // - error_measure(product_old, C_ij, inverse_density);
//         }

//         int error = INT_MAX;
//         bit_vector_t B_j_changed;
//         for(int k=0; k<numTests; ++k) {
//             if(error_tests[k] != 0 && error_tests[k] < error) {
//                 error = error_tests[k];
//                 B_j_changed = B_j_tests[k];
//             }
//         }
//         // if(error == INT_MAX) {
//         if(error > 0) {
//             const uint32_t k = fast_kiss32(state) % numTests;
//             B_j_changed = B_j_tests[k];
//             error = error_tests[k];
//         }

//         if (metro(state, error, temperature, height)) {
//             Bb[j] = B_j_changed;
//             error_update += error;
//         }
//     }

//     return error_update;
// }

struct coo {
    coo(uint32_t x, uint32_t y) : x_{x}, y_{y} {}

    uint32_t x_;
    uint32_t y_;
};

vector<coo> computeProductCOO(const vector<uint32_t> &Ab,
                              const vector<uint32_t> &Bb,
                              const int height,
                              const int width)
{
    vector<coo> C;

    #pragma omp parallel for ordered schedule(static,1)
    for(int i=0; i < height; ++i) {
        uint32_t row = Ab[i];
        vector<coo> Ci;
        for(int j=0; j < width; ++j) {
            if(row & Bb[j])
                Ci.emplace_back(i,j);
        }
        #pragma omp ordered
        C.insert(C.end(), Ci.begin(), Ci.end());
    }
    return C;
}


template<typename factor_t = uint32_t>
class Cubin_CPU
{
    using factor_matrix_t = vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = vector<bit_vector_t>;

    using my_error_t = float;

public:
    Cubin_CPU(const factor_matrix_t& A,
          const factor_matrix_t& B,
          const bit_matrix_t& C,
          const uint8_t factorDim = 20,
          const float density = 0.99f)
    {
        cout << "~~~ CPU CuBin ~~~" << endl; 

        if(factorDim > 32) {
            std::cerr << "Factor dimension too big! Maximum is 32." << endl;
            factorDim_ = 32;
        }
        else factorDim_ = factorDim;

        // inverse_density_ = 1 / density;

        initialize(A, B, C);
    }

    ~Cubin_CPU() {
        clear();
    }

    bool initialize(const factor_matrix_t& A, const factor_matrix_t& B, const bit_matrix_t& C) {
        if (std::is_same<factor_t, uint32_t>::value) {
            lineSize_ = 1;
            lineSize_padded_ = 1;
        }
        if(std::is_same<factor_t, float>::value) {
            lineSize_ = factorDim_;
            lineSize_padded_ = factorDim_;
        }

        if( SDIV(A.size()/lineSize_,32) * B.size()/lineSize_ != C.size()) {
            std::cerr << "CuBin construction: Matrix dimension mismatch." << std::endl;
            return false;
        }

        if(initialized_) {
            std::cerr << "CuBin already initialized. Please clear CuBin before reinitialization." << std::endl;
            return false;
        }

        height_ = A.size() / lineSize_;

        width_ = B.size() / lineSize_;
        
        A_ = A;
        B_ = B;
        C_ = C;

        inverse_density_rows_ = computeInverseDensitiesRows(C_, height_, width_);
        inverse_density_cols_ = computeInverseDensitiesCols(C_, height_, width_);

        // for (int i = 0; i < height_; ++i) {
        //     cout << inverse_density_rows_[i] << " ";
        // }
        // cout << endl;

        // for (int j = 0; j < width_; ++j) {
        //     cout << inverse_density_cols_[j] << " ";
        // }
        // cout << endl;

        distance_ = computeDistanceCPU(A_, B_, C_, height_, width_, inverse_density_rows_, inverse_density_cols_);

        std::cout << "CuBin initialization complete." << endl;

        std::cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;
        std::cout << "Factor dimension:\t" << (int) factorDim_ << endl;

        std::cout << "Start distance: "
                  << "\tabs_err: " << distance_
                  << "\trel_err: " << (float) distance_ / (height_ * width_)
                  << std::endl;

        return initialized_ = true;
    }

    bool verifyDistance() {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return false;
        }

        my_error_t distance_proof;

        distance_proof = computeDistanceCPU(A_, B_, C_, height_, width_, inverse_density_rows_, inverse_density_cols_);

        bool equal = fabs(distance_- distance_proof) < 1e-3; // std::numeric_limits<float>::epsilon();
        if(!equal) {
            std::cout << "----- !Distances differ! -----\n";
            std::cout << "Running distance:  " << distance_ << "\n";
            std::cout << "Real distance:     " << distance_proof << std::endl;
        }
        return equal;
    } 

    void clear() {
        if(initialized_) {
            A_.clear();
            B_.clear();
            C_.clear();
            distance_ = 0;
            initialized_ = false;
        }
    }

    void getFactors(factor_matrix_t& A, factor_matrix_t& B) {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        A = A_;
        B = B_;
    }

    my_error_t getDistance() {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return -1;
        }
        return distance_;
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

    void run(const CuBin_config& config) {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t linesAtOnce = config.linesAtOnce;

        if(config.verbosity > 0) {
            std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
            std::cout << "- - - - Starting " << config.maxIterations
                      << " CPU iterations, changing " << linesAtOnce
                      << " lines each time\n";
            std::cout << "- - - - Showing error every " << config.distanceShowEvery
                      << " steps\n";
            if(config.tempStart > 0) {
                std::cout << "- - - - Start temperature " << config.tempStart
                          << " multiplied by " << config.tempFactor
                          << " every " << config.tempStep
                          << " steps\n";

            }
            std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -";
            std::cout << std::endl;
        }

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
        float temperature = config.tempStart;
        size_t iteration = 0;
        size_t stuckIterations = 0;
        auto distancePrev = distance_;
        int lineToBeChanged;
        uint32_t cpuSeed;
        #pragma omp parallel firstprivate(iteration)
        while( distance_ > config.distanceThreshold
                && iteration++ < config.maxIterations
                && temperature > config.tempEnd
                && stuckIterations < config.stuckIterationsBeforeBreak)
        {
            // Change rows
            #pragma omp single
            {
                lineToBeChanged = (fast_kiss32(state) % height_) / WARPSPERBLOCK * WARPSPERBLOCK;
                cpuSeed = fast_kiss32(state) + iteration;
            }

            my_error_t distance_update = vectorMatrixMultCompareLineCPU<false>(A_, height_, B_, width_, C_, factorDim_,
                                          lineToBeChanged, min(linesAtOnce, height_), cpuSeed, temperature/10,
                                          config.flipManyChance, config.flipManyDepth,
                                          inverse_density_rows_, inverse_density_cols_);
            // int distance_update = vectorMatrixMultCompareRowCPU(A_, B_, C_, height_, width_, factorDim_,
            //                               lineToBeChanged, min(linesAtOnce, height_), cpuSeed, temperature/10,
            //                               config.flipManyChance, config.flipManyDepth, inverse_density_);
            // implicit barrier

            // Change cols
            #pragma omp single
            {
                lineToBeChanged = (fast_kiss32(state) % width_) / WARPSPERBLOCK * WARPSPERBLOCK;
                cpuSeed = fast_kiss32(state) + iteration;
            }

            distance_update += vectorMatrixMultCompareLineCPU<true>(B_, width_, A_, height_, C_, factorDim_,
                                          lineToBeChanged, min(linesAtOnce, height_), cpuSeed, temperature/10,
                                          config.flipManyChance, config.flipManyDepth,
                                          inverse_density_cols_, inverse_density_rows_);
            // distance_update += vectorMatrixMultCompareColCPU(A_, B_, C_, height_, width_, factorDim_,
            //                               lineToBeChanged, min(linesAtOnce, height_), cpuSeed, temperature/10,
            //                               config.flipManyChance, config.flipManyDepth, inverse_density_);
            // implicit barrier

            // #pragma omp atomic
            // distance_ += distance_update;
            // #pragma omp barrier

            #pragma omp single
            {
                int hamming;
                if(iteration % config.distanceShowEvery == 0) {
                    distance_ = computeDistanceCPU(A_, B_, C_, height_, width_, inverse_density_rows_, inverse_density_cols_);
                    hamming = computeHammingDistanceCPU(A_, B_, C_, height_, width_);
                }


                if(config.verbosity > 0 && iteration % config.distanceShowEvery == 0) {
                    std::cout << "Iteration: " << iteration
                              << "\terror: " << distance_
                              // << "\trel_err: " << (float) distance_ / (height_*width_)
                              << "\thamming: " << hamming
                              << "\ttemp: " << temperature;
                    std::cout << std::endl;

                    // std::cout << "\tseed: " << (int) cpuSeed << std::endl;
                }
                if(iteration % config.tempStep == 0) {
                    temperature *= config.tempFactor;
                }
                if(distance_ == distancePrev)
                    stuckIterations++;
                else
                    stuckIterations = 0;
                distancePrev = distance_;
            }
        }

        if(config.verbosity > 0) {
            if (!(iteration < config.maxIterations))
                std::cout << "Reached iteration limit: " << config.maxIterations << std::endl;
            if (!(distance_ > config.distanceThreshold))
                std::cout << "Distance below threshold." << std::endl;
            if (!(temperature > config.tempEnd))
                std::cout << "Temperature below threshold." << std::endl;
            if (!(stuckIterations < config.stuckIterationsBeforeBreak))
                std::cout << "Stuck for " << stuckIterations << " iterations." << std::endl;
        }
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        std::cout << "Final result: "
                  << "\tabs_err: " << distance_
                  << "\trel_err: " << (float) distance_ / (height_ * width_)
                  << std::endl;
    }  

private:
    bool initialized_ = false;
    factor_matrix_t A_;
    factor_matrix_t B_;
    bit_matrix_t C_;
    vector<my_error_t> inverse_density_rows_;
    vector<my_error_t> inverse_density_cols_;
    // int inverse_density_;
    my_error_t distance_;
    // size_t height_padded;
    uint8_t factorDim_ = 20;
    size_t height_ = 0;
    size_t width_ = 0;
    size_t lineSize_ = 1;
    size_t lineSize_padded_ = 1;
};

#endif
