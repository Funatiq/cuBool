#ifndef UPDATES_CUH
#define UPDATES_CUH

#include "helper/cuda_helpers.cuh"
#include "helper/rngpu.hpp"

// uint32_t vector masks --------------------------------------------------------
__inline__ __device__ __host__
uint32_t get_flip_mask_many(const uint8_t factorDim, fast_kiss_state32_t state, const uint32_t rand_depth) {
    uint32_t bit_flip_mask = FULLMASK >> (32-factorDim);
    #pragma unroll
    for(int i = 0; i < rand_depth; ++i) {
        bit_flip_mask &= fast_kiss32(state);
    }
    // bit_flip_mask &= FULLMASK >> (32-factorDim);
    return bit_flip_mask;
}

__inline__ __device__ __host__
uint32_t get_flip_mask_all(const uint8_t factorDim) {
    return FULLMASK >> (32-factorDim);
}

__inline__ __device__ __host__
uint32_t get_flip_mask_one(const uint8_t factorDim, fast_kiss_state32_t state) {
    const uint32_t lane = fast_kiss32(state) % factorDim;
    return 1 << lane;
}

__inline__ __device__ __host__
uint32_t get_flip_mask(const uint8_t factorDim, fast_kiss_state32_t state,
                       const float flipManyChance,
                       const uint32_t flipManyDepth) {
    const float random_many = fast_kiss32(state) / (float) UINT32_MAX;

    return random_many < flipManyChance ? get_flip_mask_many(factorDim, state, flipManyDepth)
                                        : get_flip_mask_one(factorDim, state);
    // return random_many < flipManyChance ? get_flip_mask_all() : get_flip_mask_one(state);
}

// float updates ---------------------------------------------------------------
__inline__ __device__
float get_float_update_many(fast_kiss_state32_t state) {
    return 0.1f;
}

__inline__ __device__
float get_float_update_one(const uint8_t factorDim, fast_kiss_state32_t state) {
    const uint32_t lane = fast_kiss32(state) % factorDim;
    return threadIdx.x % warpSize == lane ? 0.1f : 0.0f;
}

__inline__ __device__
float get_float_update(const uint8_t factorDim, fast_kiss_state32_t state, const float flipManyChance) {
    const float random_many = fast_kiss32(state) / (float) UINT32_MAX;
    float update = random_many < flipManyChance ? get_float_update_many(state) : get_float_update_one(factorDim, state);

    const float random_factor = fast_kiss32(state) / (float) UINT32_MAX;
    update = random_factor < 0.5f ? 5*update : update;
    // update = 10*update;

    const float random_sign = fast_kiss32(state) / (float) UINT32_MAX;
    return random_sign < 0.5f ? update : -1.0*update;
}

// Metropolis–Hastings algorithm
template<typename error_t>
__inline__ __device__ __host__
bool metro(fast_kiss_state32_t state, const error_t error, const float temperature, const int error_max = 1) {
    if(error <= 0)
        return true;
    if(temperature <= 0)
        return false;
    const float randomNumber = fast_kiss32(state) / (float) UINT32_MAX;
    // const float metro = fminf(1.0f, expf((float) - error / error_max / temperature));
    const float metro = expf((float) - error / error_max / temperature);
    return randomNumber < metro;
}

// error measures ---------------------------------------------------------------
template<typename error_t>
__inline__ __device__ __host__
error_t error_measure0(const int test, const int truth, const error_t weigth) {
    return (truth == 1) ? weigth * (test ^ truth) : (test ^ truth);
}

template<typename error_t>
__inline__ __device__ __host__
error_t error_measure(const int test, const int truth, const error_t weigth_1, const error_t weigth_0) {
    return (truth == 1) ? weigth_1 * (test ^ truth) : weigth_0 * (test ^ truth);
}

// template<typename error_t>
// __inline__ __device__ __host__
// error_t error_measure(const int test, const int truth, const error_t weigth_1, const error_t weigth_0) {
//     return (truth == 1) ? -1 * weigth_1 * test : weigth_0 * test;
// }

template<typename error_t>
__inline__ __device__ __host__
error_t error_measurew(const int test, const int truth, const error_t weigth_0) {
    return (truth == 1) ? -1 * test : weigth_0*0.2f * test;
}

// __inline__ __device__ __host__
// int error_measure(const int test, const int truth, const int inverse_density = 0) {
//     return test ^ truth;
// }

__inline__ __device__ __host__
int error_measure(const int test, const int truth, const int inverse_density) {
    return (truth == 1) ? 1 * (test ^ truth) : (test ^ truth);
}

// __inline__ __device__ __host__
// int error_measure3(const int test, const int truth, const int inverse_density) {
//     return (truth == 0) ? inverse_density * (test ^ truth) : (test ^ truth);
// }

#endif