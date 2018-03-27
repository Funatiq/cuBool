#ifndef COMPUTE_ERROR_CUH
#define COMPUTE_ERROR_CUH

#include "cuda_helpers.cuh"


// Start error kernel
__global__
void computeFullError(const uint32_t *Ab, const uint32_t *Bb, const uint32_t *Cb, 
                      const int height, const int width,const int padded_width,
                      int *distance_test) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    //__shared__ volatile int shared_distance[THREADSPERBLOCK];
    //shared_distance[threadIdx.x] = 0;
    __shared__ int reductionArray[32];
    
    int error_thread = 0;
    if (j < width) {
        for (int i = 0; i < height; i++) {
            int lineSum = (Ab[i] & Bb[j]) ? 1 : 0;
            int intId = i / 32 * padded_width + j;
            int intLane = i % 32;
            // int intId = (j * height + i) / 32;
            // int intLane = (j * height + i) % 32;
            int truthEntry = (Cb[intId] >> (32 - intLane - 1)) & 1; 
            error_thread += lineSum ^ truthEntry;
        }
    }
    __syncthreads();
    
    int error_block = blockReduceSum(error_thread, reductionArray);
    // Thread with threadIdx.x==0 now has total error of block

   if (threadIdx.x == 0) {
        atomicAdd(distance_test, error_block);
   }
}

template<typename bit_vector_t>
void computeError(const bit_vector_t *d_Ab, const bit_vector_t *d_Bb, const bit_vector_t *d_Cb, 
                  const int height, const int width, const int padded_width,
                  int *&d_distance_C0_C, int &distance_C0_C)
{

    cudaMalloc((void **) &d_distance_C0_C, sizeof(int));                                                       CUERR
    cudaMemset(d_distance_C0_C, 0, sizeof(int));                                                       CUERR
    // cudaMemcpy(d_distance_C0_C, &distance_C0_C, sizeof(int), cudaMemcpyHostToDevice);                     CUERR

    computeFullError <<< SDIV(width, THREADSPERBLOCK), THREADSPERBLOCK >>>
                        (d_Ab, d_Bb, d_Cb, height, width, padded_width, d_distance_C0_C);                    CUERR
    
    cudaMemcpy(&distance_C0_C, d_distance_C0_C, sizeof(int), cudaMemcpyDeviceToHost);     CUERR
}

// Only for debugging
template<typename bit_vector_t>
void checkDistance(const bit_vector_t *d_Ab, const bit_vector_t *d_Bb, const bit_vector_t *d_C0b,
                   const int height, const int width, const int padded_width)
{
    int distance_test;
    int *d_distance_test;
    // distance_test = 0;

    computeError(d_Ab, d_Bb, d_C0b, height, width, padded_width, d_distance_test, distance_test);

    cudaFree(d_distance_test); CUERR

    printf("Real Error: \t%f\n",
           distance_test / ((double) height * width));
}

#endif
