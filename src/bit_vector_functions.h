#ifndef BIT_VECTOR_FUNCTIONS_H
#define BIT_VECTOR_FUNCTIONS_H

#include <vector>
#include <bitset>

#include "helper/confusion.h"

#include "config.h"
#include "io_and_allocation.hpp"
#include "updates_and_measures.cuh"

using std::vector;


template<typename bit_vector_t, typename index_t>
size_t computeHammingDistanceCPU(const vector<bit_vector_t> &Ab,
                        const vector<bit_vector_t> &Bb,
                        const vector<bit_vector_t> &Cb,
                        const index_t height,
                        const index_t width)
{
    size_t error = 0;

    #pragma omp parallel for reduction(+:error)
    for(index_t j=0; j < width; ++j) {
        uint32_t B_j = Bb[j];
        for(index_t i=0; i < height; ++i) {
            const int product = (Ab[i] & B_j) ? 1 : 0;

            const index_t vecId = i / 32 * width + j;
            const index_t vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            error += product ^ C_ij;
        }
    }

    return error;
}

template<typename bit_vector_t>
int nonzeroDimension(vector<bit_vector_t>& Ab){
    bit_vector_t columns = 0;
    for(auto& a : Ab)
        columns |= a;
    std::bitset<std::numeric_limits<bit_vector_t>::digits> bits(columns);
    return bits.count();
}


template<typename bit_vector_t, typename index_t>
confusion_matrix computeErrorsCPU(const vector<bit_vector_t> &Ab,
                        const vector<bit_vector_t> &Bb,
                        const vector<bit_vector_t> &Cb,
                        const index_t height,
                        const index_t width)
{
    size_t true_positives = 0;
    size_t true_negatives = 0;
    size_t false_positives = 0;
    size_t false_negatives = 0;

    #pragma omp parallel for reduction(+:true_positives) reduction(+:true_negatives) reduction(+:false_positives) reduction(+:false_negatives)
    for(index_t j=0; j < width; ++j) {
        uint32_t B_j = Bb[j];
        for(index_t i=0; i < height; ++i) {
            const int product = (Ab[i] & B_j) ? 1 : 0;

            const index_t vecId = i / 32 * width + j;
            const index_t vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            true_positives  +=  C_ij &  product;
            true_negatives  += !(C_ij | product);
            false_positives += (!C_ij) &  product;
            false_negatives +=  C_ij & !product;
        }
    }

    return confusion_matrix(true_positives, true_negatives, false_positives, false_negatives);
}

template<typename bit_vector_t, typename index_t>
float computeTruePositiveCPU(const vector<bit_vector_t> &Ab,
                       const vector<bit_vector_t> &Bb,
                       const vector<bit_vector_t> &Cb,
                       const index_t height,
                       const index_t width)
{
    float true_positives = 0;

    #pragma omp parallel for reduction(+:true_positives)
    for(index_t j=0; j < width; ++j) {
        uint32_t B_j = Bb[j];
        for(index_t i=0; i < height; ++i) {
            const int product = (Ab[i] & B_j) ? 1 : 0;

            const index_t vecId = i / 32 * width + j;
            const index_t vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            if(product & C_ij) {
                true_positives++;
            }
        }
    }

    return true_positives;
}

template<typename bit_vector_t, typename index_t>
float computeJaccardCPU(const vector<bit_vector_t> &Ab,
                       const vector<bit_vector_t> &Bb,
                       const vector<bit_vector_t> &Cb,
                       const index_t height,
                       const index_t width)
{
    float jaccard = 0;

    #pragma omp parallel for reduction(+:jaccard)
    for(index_t j=0; j < width; ++j) {
        uint32_t B_j = Bb[j];
        size_t true_positives = 0;
        size_t false_positives = 0;
        size_t false_negatives = 0;
        for(index_t i=0; i < height; ++i) {
            const int product = (Ab[i] & B_j) ? 1 : 0;

            const index_t vecId = i / 32 * width + j;
            const index_t vecLane = i % 32;
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

template<typename bit_factor_t, typename bit_matrix_t, typename index_t, typename error_t>
error_t computeDistanceCPU(const vector<bit_factor_t> &Ab,
                           const vector<bit_factor_t> &Bb,
                           const vector<bit_matrix_t> &Cb,
                           const index_t height,
                           const index_t width,
                           const vector<error_t>& weights_rows,
                           const vector<error_t>& weights_cols)
{
    error_t error = 0;

    #pragma omp parallel for reduction(+:error)
    for(index_t i=0; i < height; ++i) {
        uint32_t A_i = Ab[i];
        for(index_t j=0; j < width; ++j) {
            const int product = (A_i & Bb[j]) ? 1 : 0;

            const index_t vecId = i / 32 * width + j;
            const index_t vecLane = i % 32;
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

template<typename bit_vector_t, typename index_t, typename error_t = float>
vector<error_t> computeDensitiesRows(const vector<bit_vector_t> &Cb,
                                            const index_t height,
                                            const index_t width)
{
    vector<error_t> density_rows(height);

    #pragma omp parallel for
    for(index_t i=0; i<height; ++i) {
        size_t nonZeroCount = 0;
        for(index_t j=0; j<width; ++j) {
            const index_t vecId = i / 32 * width + j;
            const index_t vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            nonZeroCount += C_ij;
        }
        density_rows[i] = (error_t) nonZeroCount / width;
    }

    return density_rows;
}

template<typename bit_vector_t, typename index_t, typename error_t = float>
vector<error_t> computeDensitiesCols(const vector<bit_vector_t> &Cb,
                                        const index_t height,
                                        const index_t width)
{
    vector<error_t> density_cols(width);

    #pragma omp parallel for
    for(index_t j=0; j<width; ++j) {
        size_t nonZeroCount = 0;
        for(index_t i=0; i<height; ++i) {
            const index_t vecId = i / 32 * width + j;
            const index_t vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            nonZeroCount += C_ij;
        }
        density_cols[j] = (error_t) nonZeroCount / height;
    }

    return density_cols;
}

template<typename bit_vector_t, typename index_t, typename error_t = float>
vector<error_t> computeInverseDensitiesRows(const vector<bit_vector_t> &Cb,
                                            const index_t height,
                                            const index_t width)
{
    vector<error_t> inverse_density_rows(height);

    #pragma omp parallel for
    for(index_t i=0; i<height; ++i) {
        size_t nonZeroCount = 0;
        for(index_t j=0; j<width; ++j) {
            const index_t vecId = i / 32 * width + j;
            const index_t vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            nonZeroCount += C_ij;
        }
        if(nonZeroCount == 0) nonZeroCount++;
        inverse_density_rows[i] = (error_t) width / nonZeroCount;
    }

    return inverse_density_rows;
}

template<typename bit_vector_t, typename index_t, typename error_t = float>
vector<error_t> computeInverseDensitiesCols(const vector<bit_vector_t> &Cb,
                                        const index_t height,
                                        const index_t width)
{
    vector<error_t> inverse_density_cols(width);

    #pragma omp parallel for
    for(index_t j=0; j<width; ++j) {
        size_t nonZeroCount = 0;
        for(index_t i=0; i<height; ++i) {
            const index_t vecId = i / 32 * width + j;
            const index_t vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            nonZeroCount += C_ij;
        }
        if(nonZeroCount == 0) nonZeroCount++;
        inverse_density_cols[j] = (error_t) height / nonZeroCount;
    }

    return inverse_density_cols;
}


template<typename bit_vector_t, typename index_t>
void updateWholeColumn(vector<bit_vector_t> &Ab,
                                   const index_t size_A,
                                   const uint8_t factorDim,
                                    const uint8_t column,
                                    const float density,
                                    const uint32_t seed)
{
    updateColumnPart(Ab, size_A, factorDim, column, density, 0, size_A, seed);
}

template<typename bit_vector_t, typename index_t>
void updateColumnPart(vector<bit_vector_t> &Ab,
                                   const index_t size_A,
                                   const uint8_t factorDim,
                                   const uint8_t column,
                                   const float density,
                                   const index_t startline,
                                   const index_t numlines,
                                   const uint32_t seed)
{
    const double threshold = getInitChance(density, factorDim);

    #pragma omp for
    for (index_t id = 0; id < numlines; ++id) {
        const index_t i = (startline + id) % size_A;

        fast_kiss_state32_t state;
        state = get_initial_fast_kiss_state32(seed + i);

        const bool set_one = fast_kiss32(state) < threshold * UINT32_MAX;

        if (set_one)
            Ab[i] |= 1 << column;
        else //set 0
            Ab[i] &= ~(1 << column);
    }
}

template<bool transpose, typename bit_vector_t, typename index_t>
confusion_matrix optimizeWholeColumn(vector<bit_vector_t> &Ab,
                                   const index_t size_A,
                                   const vector<bit_vector_t> &Bb,
                                   const index_t size_B,
                                   const vector<bit_vector_t> &Cb,
                                   const uint8_t factorDim,
                                   const uint8_t k)
{
    confusion_matrix confusion_new;

    #pragma omp for
    for (index_t i = 0; i < size_A; ++i) {

        const bit_vector_t A_i_0 = Ab[i] & ~(1 << k);
        const bit_vector_t A_i_1 = Ab[i] | (1 << k);

        confusion_matrix confusion_0;
        confusion_matrix confusion_1;

        for(index_t j=0; j < size_B; ++j) {
            const index_t vecId = transpose ? j / 32 * size_A + i : i / 32 * size_B + j;
            const index_t vecLane = transpose ? j % 32 : i % 32;
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

template<bool transpose, typename bit_vector_t, typename index_t>
confusion_matrix updateLinesJaccardCPU(vector<bit_vector_t> &Ab,
                                   const index_t size_A,
                                   const vector<bit_vector_t> &Bb,
                                   const index_t size_B,
                                   const vector<bit_vector_t> &Cb,
                                   const uint8_t factorDim,
                                   const index_t startline,
                                   const index_t numlines,
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
    for(index_t id=0; id < numlines; ++id) {
        const index_t i = (startline + id) % size_A;

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
        // size_t true_positives_old = 0;
        // size_t true_positives_new = 0;
        // // size_t false_xs_old = 0;
        // // size_t false_xs_new = 0;
        // size_t false_negatives_old = 0;
        // size_t false_negatives_new = 0;
        // size_t false_positives_old = 0;
        // size_t false_positives_new = 0;
        for(index_t j=0; j < size_B; ++j) {
            const index_t vecId = transpose ? j / 32 * size_A + i : i / 32 * size_B + j;
            const index_t vecLane = transpose ? j % 32 : i % 32;
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
        const size_t all_tp_draw = confusion.TP - confusion_old.TP + confusion_draw.TP;
        const size_t all_tp_flip = confusion.TP - confusion_old.TP + confusion_flip.TP;
        // const size_t all_false_positives_new = confusion.FP - false_positives_old + false_positives_new;
        // const size_t all_false_negatives_new = confusion.FN - false_negatives_old + false_negatives_new;
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

template<bool transpose, typename bit_vector_t, typename index_t, typename error_t>
int vectorMatrixMultCompareLineCPU(vector<bit_vector_t> &Ab,
                                   const index_t size_A,
                                   const vector<bit_vector_t> &Bb,
                                   const index_t size_B,
                                   const vector<bit_vector_t> &Cb,
                                   const uint8_t factorDim,
                                   const index_t startline,
                                   const index_t numlines,
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
    for(index_t id=0; id < numlines; ++id) {
        const index_t i = (startline + id) % size_A;

        fast_kiss_state32_t state;
        state = get_initial_fast_kiss_state32(seed + id);

        const bit_vector_t A_i = Ab[i];
        // bit_vector_t A_i_changed = Ab[i] ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);

        for(uint8_t k=0; k<factorDim; ++k) {
            A_i_tests[k] = A_i ^ (1 << k);
            A_i_tests[factorDim+k] = A_i ^ get_flip_mask_many(factorDim, state, flipManyDepth);
            error_tests[k] = 0;
            error_tests[factorDim+k] = 0;
        }
        // A_i_tests[numTests] = A_i ^ get_flip_mask_many(factorDim, state, flipManyDepth);
        // A_i_tests[numTests] = fast_kiss32(state) >> (32 - factorDim);
        // error_tests[numTests] = 0;

        error_t error = 0;
        for(index_t j=0; j < size_B; ++j) {
            const index_t vecId = transpose ? j / 32 * size_A + i : i / 32 * size_B + j;
            const index_t vecLane = transpose ? j % 32 : i % 32;
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

            const int product_old = (A_i         & Bb[j]) ? 1 : 0;
            // const int product_new = (A_i_changed & Bb[j]) ? 1 : 0;

            // error += error_measure(product_new, C_ij, weight, counterweight)
            //        - error_measure(product_old, C_ij, weight, counterweight);

            for(uint8_t k=0; k<numTests; ++k) {
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



template <typename index_t>
struct coo {
    coo(index_t x, index_t y) : x_{x}, y_{y} {}

    index_t x_;
    index_t y_;
};

template <typename bit_vector_t, typename index_t>
vector<coo<index_t>> computeProductCOO(
							const vector<bit_vector_t> &Ab,
                            const vector<bit_vector_t> &Bb,
                            const index_t height,
                            const index_t width)
{
    vector<coo<index_t>> C;

    #pragma omp parallel for ordered schedule(static,1)
    for(index_t i=0; i < height; ++i) {
        bit_vector_t row = Ab[i];
        vector<coo<index_t>> Ci;
        for(index_t j=0; j < width; ++j) {
            if(row & Bb[j])
                Ci.emplace_back(i,j);
        }
        #pragma omp ordered
        C.insert(C.end(), Ci.begin(), Ci.end());
    }
    return C;
}

#endif
