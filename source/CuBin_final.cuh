#ifndef CUBIN_FINAL
#define CUBIN_FINAL

template<typename bit_vector_t>
void computeStartError(bit_vector_t *d_Ab, bit_vector_t *d_Bb, bit_vector_t *d_Cb, 
                        int width, int height,
                        int *&d_distance_C0_C_start, int &distance_C0_C_start);

template<typename bit_vector_t>
void checkDistance(bit_vector_t *d_Ab, bit_vector_t *d_Bb, bit_vector_t *d_C0b, int height, int width);

// template<typename bit_vector_t, typename element_t = uint32_t>
// void aftertestGPU(bit_vector_t *d_Ab, bit_vector_t *d_Bb, bit_vector_t *d_C0b, 
//                   int width, int height);

// template<typename bit_vector_t>
// void CPUcomputation(bit_vector_t *Ab, bit_vector_t *Bb, bit_vector_t *C0, 
//                     int width, int height, 
//                     int startDistance, uint32_t seed, int updateStep,
//                     float threshold, int linesAtOnce);

// template<typename bit_vector_t>
// void CPUvectorMatrixMultCompareRow( bit_vector_t *Ab, bit_vector_t *Bb, 
//                                     bit_vector_t *C0b, int width, int height, int startrow,
//                                     int *hDistance, fast_kiss_state32_t *state, int rowsAtOnce);

// template<typename bit_vector_t>
// void CPUvectorMatrixMultCompareCol( bit_vector_t *Ab, bit_vector_t *Bb, bit_vector_t *C0b, 
//                                     int width, int height, int startcol,
//                                     int *hDistance, fast_kiss_state32_t *state, int rowsAtOnce);

// template<typename bit_vector_t, typename element_t = uint32_t>
// void aftertestCPU(bit_vector_t *Ab, bit_vector_t *Bb,
//                   bit_vector_t *d_Ab, bit_vector_t *d_Bb,
//                   bit_vector_t *C0b, 
//                   int width, int height);

// template<typename bit_vector_t>
// __global__ void
// vectorMatrixMultCompareRow( bit_vector_t *A, bit_vector_t *B, bit_vector_t *C, 
//                             int width, int height, 
//                             int startrow, int *global_error,
//                             uint32_t seed, float temperature);

// template<typename bit_vector_t>
// __global__ void
// vectorMatrixMultCompareCol(bit_vector_t *A, bit_vector_t *B, bit_vector_t *C, 
//                             int width, int height, 
//                             int startcol, int *global_error,
//                             uint32_t seed, float temperature);

__global__ void computeFullError(   uint32_t *A, uint32_t *B, uint32_t *C, 
                                    int width, int height, int *distance_test);

// template<typename bit_vector_t, typename element_t>
// __global__ void matrixMultiply( bit_vector_t *A, bit_vector_t *B, element_t *C, 
//                                 int width, int height);

// template<typename element_t>
// __global__ void matrixMultiplyInt(  element_t * A0, element_t * B0, element_t * C0, 
//                                     int m, int k, int n);


#endif
