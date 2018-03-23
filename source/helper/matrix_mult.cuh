#ifndef MATRIX_MULT_CUH
#define MATRIX_MULT_CUH

// Each thread one entry of a row
template<typename bit_vector_t, typename element_t>
__global__ void matrixMultiply( bit_vector_t *A, bit_vector_t *B, element_t *C, 
                                int width, int height) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < width)
        for (int i = 0; i < height; i++)
            C[i * width + tid] = (A[i] & B[tid]) > 0 ? 1 : 0;
}

// https://gist.github.com/wh5a/4313739
template<typename element_t>
__global__ void matrixMultiplyInt(  element_t * A0, element_t * B0, element_t * C0, 
                                    int m, int k, int n) {
    __shared__ element_t ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ element_t ds_B[TILE_WIDTH][TILE_WIDTH];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    element_t sum = 0;

    for(int t = 0; t < (n - 1) / TILE_WIDTH + 1; t++) {
        if(row < m && t * TILE_WIDTH + tx < n)
            ds_A[ty][tx] = (A0[row * n + t * TILE_WIDTH + tx]);
        else
            ds_A[ty][tx] = 0.0;
        if(t * TILE_WIDTH + ty < n && col < k)
            ds_B[ty][tx] = (B0[(t * TILE_WIDTH + ty) * k + col]);
        else
            ds_B[ty][tx] = 0.0;
        __syncthreads();
        for(int i = 0; i < TILE_WIDTH; i++){
            sum += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    }
    if(row < m && col < k)
        C0[col + row * k] = min(1, sum);
}

#endif