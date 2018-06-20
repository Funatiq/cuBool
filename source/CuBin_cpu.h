#ifndef CUBIN_CPU_H
#define CUBIN_CPU_H

#include <vector>
#include <iostream>
#include <limits>
#include <cmath>

#include "helper/config.h"
#include "helper/rngpu.hpp"
#include "helper/updates_and_measures.cuh"

using std::vector;
using std::cout;
using std::cerr;
using std::endl;

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

struct confusion_matrix {
    int TP;
    int TN;
    int FP;
    int FN;

    confusion_matrix() : TP(0), TN(0), FP(0), FN(0) {};

    confusion_matrix(int tp, int tn, int fp, int fn) : TP(tp), TN(tn), FP(fp), FN(fn) {};

    float precision() {
        return 1.0f*TP / (TP + FP);
    }

    float sensitivity() {
        return 1.0f*TP / (TP + FN);
    }

    float f1score() {
        return 2.0f*TP / (2*TP + FP + FN);
    }

    float jaccard() {
        return 1.0f*TP / (TP + FP + FN);
    }

    int total_error() {
        return FP + FN;
    }

    int problem_size() {
        return TP + TN + FP + FN;
    }

    float rel_error() {
        return float(total_error()) / problem_size();
    }
};

template<typename bit_vector_t>
confusion_matrix computeErrorsCPU(const vector<bit_vector_t> &Ab,
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

            // if(product) {
            //     if(C_ij)
            //         true_positives++;
            //     else
            //         false_positives++;
            // } else {
            //     if(C_ij)
            //         false_negatives++;
            //     else
            //         true_negatives++;
            // }
            true_positives  +=  C_ij &  product;
            true_negatives  += !(C_ij | product);
            false_positives += (!C_ij) &  product;
            false_negatives +=  C_ij & !product;
        }
    }

    return confusion_matrix(true_positives, true_negatives, false_positives, false_negatives);
}

template<typename bit_vector_t>
float computeTruePositiveCPU(const vector<bit_vector_t> &Ab,
                       const vector<bit_vector_t> &Bb,
                       const vector<bit_vector_t> &Cb,
                       const int height,
                       const int width)
{
    float true_positives = 0;

    #pragma omp parallel for reduction(+:true_positives)
    for(int j=0; j < width; ++j) {
        uint32_t B_j = Bb[j];
        for(int i=0; i < height; ++i) {
            const int product = (Ab[i] & B_j) ? 1 : 0;

            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            if(product & C_ij) {
                true_positives++;
            }
        }
    }

    return true_positives;
}

template<typename bit_vector_t>
float computeJaccardCPU(const vector<bit_vector_t> &Ab,
                       const vector<bit_vector_t> &Bb,
                       const vector<bit_vector_t> &Cb,
                       const int height,
                       const int width)
{
    float jaccard = 0;

    #pragma omp parallel for reduction(+:jaccard)
    for(int j=0; j < width; ++j) {
        uint32_t B_j = Bb[j];
        int true_positives = 0;
        int false_positives = 0;
        int false_negatives = 0;
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
            }
        }
        jaccard += (float) true_positives / (true_positives + false_positives + false_negatives);
    }

    return jaccard;
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

            // const error_t weight_average = (weights_rows[i] + weights_cols[j]) / 2;

            // const error_t weight = 1 / weights_rows[i] - 1 + 1 / weights_cols[j] - 1;
            // const error_t weight = (weights_rows[i] - 1 + weights_cols[j] - 1);
            // const error_t weight = 2 + log(weights_rows[i] - 1) + log(weights_cols[j] - 1);
            // weight = 1 + log(weight);
            // weight = sqrt(weight);
            // const error_t weight = sqrt(weights_rows[i] - 1) + sqrt(weights_cols[j] - 1);
            // const error_t counterweight = 2;
            const error_t weight = 4;
            // const error_t weight = 3;
            // const error_t weight = 1;
            const error_t counterweight = 1;

            error += error_measure(product, C_ij, weight, counterweight);
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
        if(nonZeroCount == 0) nonZeroCount++;
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
        if(nonZeroCount == 0) nonZeroCount++;
        inverse_density_cols[j] = (error_t) height / nonZeroCount;
    }

    return inverse_density_cols;
}


template<typename bit_vector_t>
void updateWholeColumn(vector<bit_vector_t> &Ab,
                                   const int size_A,
                                   const uint8_t factorDim,
                                    const uint8_t k,
                                    const float density,
                                    const uint32_t seed)
{
    #pragma omp for
    for (int i = 0; i < size_A; ++i) {
        fast_kiss_state32_t state;
        state = get_initial_fast_kiss_state32(seed + i);

        bool set_one = (fast_kiss32(state) / double(UINT32_MAX)) 
                       < (sqrt(1 - pow(1 - density, 1 / double(factorDim))));

        if (set_one)
            Ab[i] |= 1 << k;
        else //set 0
            Ab[i] &= ~(1 << k);
    }
}

template<typename bit_vector_t>
void updateColumnPart(vector<bit_vector_t> &Ab,
                                   const int size_A,
                                   const uint8_t factorDim,
                                   const uint8_t k,
                                   const float density,
                                   const int startline,
                                   const int numlines,
                                   const uint32_t seed)
{
    #pragma omp for
    for (int id = 0; id < numlines; ++id) {
        const int i = (startline + id) % size_A;

        fast_kiss_state32_t state;
        state = get_initial_fast_kiss_state32(seed + i);

        bool set_one = (fast_kiss32(state) / double(UINT32_MAX))
                       < (sqrt(1 - pow(1 - density, 1 / double(factorDim))));

        if (set_one)
            Ab[i] |= 1 << k;
        else //set 0
            Ab[i] &= ~(1 << k);
    }
}

template<bool transpose, typename bit_vector_t>
confusion_matrix optimizeWholeColumn(vector<bit_vector_t> &Ab,
                                   const int size_A,
                                   const vector<bit_vector_t> &Bb,
                                   const int size_B,
                                   const vector<bit_vector_t> &Cb,
                                   const uint8_t factorDim,
                                   const uint8_t k)
{
    confusion_matrix confusion_new;

    #pragma omp for
    for (int i = 0; i < size_A; ++i) {

        const bit_vector_t A_i_0 = Ab[i] & ~(1 << k);
        const bit_vector_t A_i_1 = Ab[i] | (1 << k);

        confusion_matrix confusion_0;
        confusion_matrix confusion_1;

        for(int j=0; j < size_B; ++j) {
            const int vecId = transpose ? j / 32 * size_A + i : i / 32 * size_B + j;
            const int vecLane = transpose ? j % 32 : i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            // const int product_0 = (A_i_0 & Bb[j] & (1<<k)) ? 1 : 0;
            // const int product_1 = (A_i_1 & Bb[j] & (1<<k)) ? 1 : 0;
            const int product_0 = (A_i_0 & Bb[j]) ? 1 : 0;
            const int product_1 = (A_i_1 & Bb[j]) ? 1 : 0;

            confusion_0.TP += C_ij & product_0;
            confusion_1.TP += C_ij & product_1;

            confusion_0.FN  += C_ij & !product_0;
            confusion_1.FN += C_ij & !product_1;

            confusion_0.FP += (!C_ij) & product_0;
            confusion_1.FP += (!C_ij) & product_1;
        }

        // if(4*confusion_0.FN + confusion_0.FP <= 4*confusion_1.FN + confusion_1.FP) {
        if(confusion_0.total_error() <= confusion_1.total_error()) {
        // if(confusion_0.precision() >= confusion_1.precision()) {
        // if(confusion_0.jaccard() > confusion_1.jaccard()) {
            Ab[i] = A_i_0;
            confusion_new.TP += confusion_0.TP;
            confusion_new.FN += confusion_0.FN;
            confusion_new.FP += confusion_0.FP;
        }
        else {
            Ab[i] = A_i_1;
            confusion_new.TP += confusion_1.TP;
            confusion_new.FN += confusion_1.FN;
            confusion_new.FP += confusion_1.FP;
        }
    }
    return confusion_new;
}

template<bool transpose, typename bit_vector_t>
confusion_matrix updateLinesJaccardCPU(vector<bit_vector_t> &Ab,
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
                                   // const int all_true_positives)
                                   const confusion_matrix confusion)
{
    // int update = 0;
    confusion_matrix confusion_update;

    #pragma omp for
    // #pragma omp parallel for reduction(+:update)
    for(int id=0; id < numlines; ++id) {
        const int i = (startline + id) % size_A;

        fast_kiss_state32_t state;
        state = get_initial_fast_kiss_state32(seed + id);

        const bit_vector_t A_i = Ab[i];
        const bit_vector_t A_i_draw = get_flip_mask_many(factorDim, state, flipManyDepth);
        const bit_vector_t A_i_flip = A_i ^ A_i_draw;
        // const bit_vector_t A_i_new = Ab[i] ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);
        // const bit_vector_t A_i_new = get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);

        confusion_matrix confusion_old;
        confusion_matrix confusion_draw;
        confusion_matrix confusion_flip;
        // int true_positives_old = 0;
        // int true_positives_new = 0;
        // // int false_xs_old = 0;
        // // int false_xs_new = 0;
        // int false_negatives_old = 0;
        // int false_negatives_new = 0;
        // int false_positives_old = 0;
        // int false_positives_new = 0;
        for(int j=0; j < size_B; ++j) {
            const int vecId = transpose ? j / 32 * size_A + i : i / 32 * size_B + j;
            const int vecLane = transpose ? j % 32 : i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            const int product_old  = (A_i      & Bb[j]) ? 1 : 0;
            const int product_draw = (A_i_draw & Bb[j]) ? 1 : 0;
            const int product_flip = (A_i_flip & Bb[j]) ? 1 : 0;

            confusion_old.TP  += C_ij & product_old;
            confusion_draw.TP += C_ij & product_draw;
            confusion_flip.TP += C_ij & product_flip;

            // false_xs_old       += C_ij ^ product_old;
            // false_xs_new       += C_ij ^ product_new;

            confusion_old.FN  += C_ij & !product_old;
            confusion_draw.FN += C_ij & !product_draw;
            confusion_flip.FN += C_ij & !product_flip;

            confusion_old.FP  += (!C_ij) & product_old;
            confusion_draw.FP += (!C_ij) & product_draw;
            confusion_flip.FP += (!C_ij) & product_flip;
        }
        const int all_tp_draw = confusion.TP - confusion_old.TP + confusion_draw.TP;
        const int all_tp_flip = confusion.TP - confusion_old.TP + confusion_flip.TP;
        // const int all_false_positives_new = confusion.FP - false_positives_old + false_positives_new;
        // const int all_false_negatives_new = confusion.FN - false_negatives_old + false_negatives_new;
        // const confusion_matrix confusion_new(confusion.TP - true_positives_old + true_positives_new,
        //                                      0,
        //                                      confusion.FP - false_positives_old + false_positives_new,
        //                                      confusion.FN - false_negatives_old + false_negatives_new);
        // const int jaccard_numerator = confusion.TP * false_xs_new - all_true_positives_new * false_xs_old;
        // const int jaccard_numerator = confusion.TP * (false_negatives_new + false_positives_new)
        //                             - all_true_positives_new * (false_negatives_old + false_positives_old);
        // const int jaccard_numerator = confusion.TP * false_negatives_new - all_true_positives_new * false_negatives_old;
        // const int jaccard_numerator = confusion.TP * false_positives_new - all_true_positives_new * false_positives_old;
        // const int jaccard_numerator = true_positives_old * (true_positives_new + false_xs_new)
                                    // - true_positives_new * (true_positives_old + false_xs_old);
        // const float jaccard = 1.0f * confusion.TP / (confusion.TP + 3*false_negatives_old + false_positives_old)
        //                     - 1.0f * all_true_positives_new / (all_true_positives_new + 3*false_negatives_new + false_positives_new);
        // const float jaccard = 1.0f * confusion.TP / (confusion.TP + 3*confusion.FN + confusion.FP)
        //                     - 1.0f * confusion_new.TP / (confusion_new.TP + 3*confusion_new.FN + confusion_new.FP);

        // const float f1score = 2.0f * confusion.TP / (2*confusion.TP + false_negatives_old + false_positives_old)
        //                     - 2.0f * all_true_positives_new / (2*all_true_positives_new + false_negatives_new + false_positives_new);

        const float jaccard_old  = 1.0f * confusion.TP / (confusion.TP + 3*confusion_old.FN + confusion_old.FP);
        const float jaccard_draw = 1.0f * all_tp_draw / (all_tp_draw + 3*confusion_draw.FN + confusion_draw.FP);
        const float jaccard_flip = 1.0f * all_tp_flip / (all_tp_flip + 3*confusion_flip.FN + confusion_flip.FP);

        bit_vector_t A_i_new = A_i_draw;
        float jaccard_new = jaccard_draw;
        confusion_matrix& confusion_new = confusion_draw;
        if(jaccard_draw > jaccard_old) {
            if(jaccard_flip > jaccard_draw) {
                A_i_new = A_i_flip;
                jaccard_new = jaccard_flip;
                confusion_new = confusion_flip;
            }
        } else {
            if(jaccard_flip > jaccard_old) {
                A_i_new = A_i_flip;
                jaccard_new = jaccard_flip;
                confusion_new = confusion_flip;
            } else {
                const uint32_t coin = fast_kiss32(state) % 2;
                if(coin) {
                    A_i_new = A_i_flip;
                    jaccard_new = jaccard_flip;
                    confusion_new = confusion_flip;
                }
            }
        }

        if (metro(state, jaccard_old - jaccard_new, temperature)) {
            Ab[i] = A_i_new;
            // update += true_positives_new - true_positives_old;
            confusion_update.TP += confusion_new.TP - confusion_old.TP;
            confusion_update.FP += confusion_new.FP - confusion_old.FP;
            confusion_update.FN += confusion_new.FN - confusion_old.FN;
        }
    }

    return confusion_update;
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
    const uint8_t numTests = factorDim*2;
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
            A_i_tests[factorDim+k] = A_i ^ get_flip_mask_many(factorDim, state, flipManyDepth);
            error_tests[k] = 0;
            error_tests[factorDim+k] = 0;
        }
        // A_i_tests[numTests] = A_i ^ get_flip_mask_many(factorDim, state, flipManyDepth);
        // A_i_tests[numTests] = fast_kiss32(state) >> (32 - factorDim);
        // error_tests[numTests] = 0;

        error_t error = 0;
        for(int j=0; j < size_B; ++j) {
            const int vecId = transpose ? j / 32 * size_A + i : i / 32 * size_B + j;
            const int vecLane = transpose ? j % 32 : i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;
            
            // const error_t weight = sqrt(weights_rows[i] - 1);
            // const error_t counterweight = 1;
            // const error_t weight = 1 / weights_rows[i] - 1 + 1 / weights_cols[j] - 1;
            // const error_t weight = (weights_rows[i] - 1 + weights_cols[j] - 1);
            // const error_t weight = 1 + log(weights_rows[i] - 1);
            // const error_t weight = sqrt(weights_rows[i] - 1) + sqrt(weights_cols[j] - 1);
            // const error_t counterweight = 2;
            const error_t weight = 4;
            // const error_t weight = 3;
            // const error_t weight = 1;
            const error_t counterweight = 1;

            if(weight < 0 || counterweight < 0)
                cout << "weight < 0" << endl;

            const int product_old = (A_i         & Bb[j]) ? 1 : 0;
            // const int product_new = (A_i_changed & Bb[j]) ? 1 : 0;

            // error += error_measure(product_new, C_ij, weight, counterweight)
            //        - error_measure(product_old, C_ij, weight, counterweight);

            for(int k=0; k<numTests; ++k) {
                const int product_new = (A_i_tests[k] & Bb[j]) ? 1 : 0;

                error_tests[k] += error_measure(product_new, C_ij, weight, counterweight)
                                - error_measure(product_old, C_ij, weight, counterweight);
            }
        }

        // error_t error = 0;
        bit_vector_t A_i_changed;
        for(int k=0; k<numTests; ++k) {
            // if(error_tests[k] != 0 && error_tests[k] < error) {
            if(error_tests[k] < error) {
                error = error_tests[k];
                A_i_changed = A_i_tests[k];
            }
        }
        if(error < 0) {
            Ab[i] = A_i_changed;
            error_update += error;
            continue;
        }
        // if(error == INT_MAX) {
        // if(error >= 0) {
        else {
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
            cerr << "Factor dimension too big! Maximum is 32." << endl;
            factorDim_ = 32;
        }
        else
            factorDim_ = factorDim;

        // inverse_density_ = 1 / density;
        density_ = density;

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
            cerr << "CuBin construction: Matrix dimension mismatch." << endl;
            return false;
        }

        if(initialized_) {
            cerr << "CuBin already initialized. Please clear CuBin before reinitialization." << endl;
            return false;
        }

        height_ = A.size() / lineSize_;

        width_ = B.size() / lineSize_;
        
        A_ = A;
        B_ = B;
        C_ = C;

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

        B_ = factor_matrix_t(height_, 0);
        for(int k=0; k<factorDim_; ++k) {
            // updateWholeColumn(A_, height_, factorDim_, k, density_, seed);
            optimizeWholeColumn<true>(B_, width_, A_, height_, C_, factorDim_, k);
        }
        distance_ = computeHammingDistanceCPU(A_, B_, C_, height_, width_);
        cout << "Start distance: "
                  << "\tabs_err: " << distance_
                  << "\trel_err: " << (float) distance_ / (height_ * width_)
                  << endl;

        for(int k=0; k<factorDim_; ++k) {
            // updateWholeColumn(A_, height_, factorDim_, k, density_, seed);
            optimizeWholeColumn<true>(B_, width_, A_, height_, C_, factorDim_, k);
        }
        distance_ = computeHammingDistanceCPU(A_, B_, C_, height_, width_);
        // distance_ = computeDistanceCPU(A_, B_, C_, height_, width_, weights_rows_, weights_cols_);

        cout << "CuBin initialization complete." << endl;

        cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;
        cout << "Factor dimension:\t" << (int) factorDim_ << endl;

        cout << "Start distance: "
                  << "\tabs_err: " << distance_
                  << "\trel_err: " << (float) distance_ / (height_ * width_)
                  << endl;

        return initialized_ = true;
    }

    bool verifyDistance() {
        if(!initialized_) {
            cerr << "CuBin not initialized." << endl;
            return false;
        }

        my_error_t distance_proof;

        distance_proof = computeHammingDistanceCPU(A_, B_, C_, height_, width_);
        // distance_proof = computeDistanceCPU(A_, B_, C_, height_, width_, weights_rows_, weights_cols_);

        bool equal = fabs(distance_- distance_proof) < 1e-3; // std::numeric_limits<float>::epsilon();
        if(!equal) {
            cout << "----- !Distances differ! -----\n";
            cout << "Running distance:  " << distance_ << "\n";
            cout << "Real distance:     " << distance_proof << endl;
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
            cerr << "CuBin not initialized." << endl;
            return;
        }

        A = A_;
        B = B_;
    }

    my_error_t getDistance() {
        if(!initialized_) {
            cerr << "CuBin not initialized." << endl;
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
            cerr << "CuBin not initialized." << endl;
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
        auto distancePrev = distance_;
        // my_error_t tempStep_distance = 1;
        // my_error_t update_sum = 0;
        int lineToBeChanged;
        uint32_t cpuSeed;
        // int all_true_positives = computeTruePositiveCPU(A_, B_, C_, height_, width_);
        // int all_true_positives = computeTruePositiveCPU(A_, B_, C_, height_, width_);
        confusion_matrix confusion = computeErrorsCPU(A_, B_, C_, height_, width_);
        confusion_matrix confusion_new;

        factor_matrix_t A_new, B_new;

        #pragma omp parallel firstprivate(iteration)
        while( distance_ > config.distanceThreshold
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

                A_new = A_;
                B_new = B_;
            }
            uint8_t k = iteration % factorDim_;
            // Change rows
            // updateWholeColumn(A_new, height_, factorDim_, k, density_, cpuSeed);
            updateColumnPart(A_new, height_, factorDim_, k, density_,
                             lineToBeChanged, min(linesAtOnce, height_), cpuSeed);

            // Change cols
            auto confusion_update = optimizeWholeColumn<true>(B_new, width_, A_new, height_, C_, factorDim_, k);
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
                    A_ = A_new;
                    B_ = B_new;

                    confusion = confusion_new;

                    // cout << "update accepted" << endl;
                }

                // int hamming;
                // if(iteration % config.distanceShowEvery == 0) {
                    // distance_ = computeDistanceCPU(A_, B_, C_, height_, width_, weights_rows_, weights_cols_);
                    // distance_ = computeHammingDistanceCPU(A_, B_, C_, height_, width_);
                // }
                distance_ = confusion.total_error();

                if(config.verbosity > 0 && iteration % config.distanceShowEvery == 0) {
                    cout << "Iteration: " << iteration
                              // << "\tupdate: " << update_sum / config.distanceShowEvery
                              << "\tTP: " << confusion.TP
                              // << "\terrors: " << confusion.total_error()
                              // << "\trel_err: " << (float) distance_ / (height_*width_)
                              << "\thamming: " << distance_
                              << "\ttemp: " << temperature;
                    cout << endl;

                    // cout << "\tseed: " << (int) cpuSeed << endl;
                    // update_sum = 0;

                    // tempStep_distance = distance_;
                }
                if(iteration % config.tempStep == 0) {
                    //delay temperature
                    // if(temperature <= 0) temperature = config.tempStart;
                    temperature *= config.tempFactor;
                    // if((float) distance_ / tempStep_distance < 0.9f)
                    //     temperature /= config.tempFactor;
                    // else
                    //     temperature *= config.tempFactor;
                    // tempStep_distance = distance_;
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
                cout << "Reached iteration limit: " << config.maxIterations << endl;
            if (!(distance_ > config.distanceThreshold))
                cout << "Distance below threshold." << endl;
            if (!(temperature > config.tempEnd))
                cout << "Temperature below threshold." << endl;
            if (!(stuckIterations < config.stuckIterationsBeforeBreak))
                cout << "Stuck for " << stuckIterations << " iterations." << endl;
        }
        cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        cout << "Final result: "
                  << "\tabs_err: " << distance_
                  << "\trel_err: " << (float) distance_ / (height_ * width_)
                  << endl;
    }  

private:
    bool initialized_ = false;
    factor_matrix_t A_;
    factor_matrix_t B_;
    bit_matrix_t C_;
    vector<my_error_t> weights_rows_;
    vector<my_error_t> weights_cols_;
    // int inverse_density_;
    float density_;
    my_error_t distance_;
    // size_t height_padded;
    uint8_t factorDim_ = 20;
    size_t height_ = 0;
    size_t width_ = 0;
    size_t lineSize_ = 1;
    size_t lineSize_padded_ = 1;
};

#endif
