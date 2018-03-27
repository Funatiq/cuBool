#ifndef CUBIN_FINAL
#define CUBIN_FINAL

template<typename bit_vector_t>
void computeError(const bit_vector_t *d_Ab, const bit_vector_t *d_Bb, const bit_vector_t *d_Cb, 
                  const int height, const int width, const int padded_width,
                  int *&d_distance_C0_C, int &distance_C0_C);

template<typename bit_vector_t>
void checkDistance(const bit_vector_t *d_Ab, const bit_vector_t *d_Bb, const bit_vector_t *d_C0b,
                   const int height, const int width, const int padded_width);

__global__
void computeFullError(const uint32_t *Ab, const uint32_t *Bb, const uint32_t *Cb, 
                      const int height, const int width, const int padded_width,
                      int *distance_test);

#endif
