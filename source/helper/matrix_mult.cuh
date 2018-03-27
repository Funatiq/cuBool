#ifndef MATRIX_MULT_CUH
#define MATRIX_MULT_CUH

#define TILE_WIDTH 32

// Each thread one entry of a row
template<typename bit_vector_t, typename element_t>
__global__ void matrixMultiply( bit_vector_t *A, bit_vector_t *B, element_t *C, 
                                int height, int width) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < width)
        for (int i = 0; i < height; i++)
            C[i * width + j] = (A[i] & B[j]) > 0 ? 1 : 0;
}

// https://gist.github.com/wh5a/4313739
template<typename element_t>
__global__ void matrixMultiplyInt(  element_t * A0, element_t * B0, element_t * C0, 
                                    int m, int n, int k) {
    __shared__ element_t ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ element_t ds_B[TILE_WIDTH][TILE_WIDTH];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    element_t sum = 0;

    for(int t = 0; t < (k - 1) / TILE_WIDTH + 1; t++) {
        if(row < m && t * TILE_WIDTH + tx < k)
            ds_A[ty][tx] = (A0[row * k + t * TILE_WIDTH + tx]);
        else
            ds_A[ty][tx] = 0.0;
        if(t * TILE_WIDTH + ty < k && col < n)
            ds_B[ty][tx] = (B0[(t * TILE_WIDTH + ty) * n + col]);
        else
            ds_B[ty][tx] = 0.0;
        __syncthreads();
        for(int i = 0; i < TILE_WIDTH; i++){
            sum += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    }
    if(row < m && col < n)
        C0[col + row * n] = min(1, sum);
}


// Used for debugging and checking for correctness, not optimized
template<typename bit_vector_t, typename element_t = uint32_t>
void aftertestGPU(  bit_vector_t *d_Ab, bit_vector_t *d_Bb, bit_vector_t *d_C0b, 
                    int height, int width, int padded_width)
{
    TIMERSTART(aftertestGPU)

    element_t *A = (element_t*) malloc(sizeof(element_t) * DIM_PARAM * height);
    element_t *B = (element_t*) malloc(sizeof(element_t) * width * DIM_PARAM);
    element_t *C0 = (element_t*) malloc(sizeof(element_t) * height * width);

    bit_vector_t *Ab = (bit_vector_t*) malloc(sizeof(bit_vector_t) * height);
    bit_vector_t *Bb = (bit_vector_t*) malloc(sizeof(bit_vector_t) * width);
    int padded_height_32 = SDIV(height, 32);
    int sizeCb = padded_height_32 * width;
    // int sizeCb = ((long long) (height * width) / 32.0 + 1);
    bit_vector_t *C0b = (bit_vector_t*) malloc(sizeof(bit_vector_t) * sizeCb);
    
    element_t *d_A, *d_B;
    cudaMalloc((void**)&d_A, sizeof(element_t) * DIM_PARAM * height);                                           CUERR
    cudaMalloc((void**)&d_B, sizeof(element_t) * width * DIM_PARAM);                                            CUERR
    
    cudaMemcpy(Ab, d_Ab, sizeof(bit_vector_t) * height, cudaMemcpyDeviceToHost);                                CUERR
    cudaMemcpy(Bb, d_Bb, sizeof(bit_vector_t) * width, cudaMemcpyDeviceToHost);                                 CUERR
    cudaMemcpy2D(C0b, sizeof(bit_vector_t) * width,
                 d_C0b, sizeof(bit_vector_t) * padded_width,
                 sizeof(bit_vector_t) * width,
                 padded_height_32,
                 cudaMemcpyDeviceToHost);                              CUERR

    uint32_t counterDensity = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < DIM_PARAM; j++){
            A[i * DIM_PARAM + j] = (Ab[i] >> 32 - j - 1) & 1;
            if (A[i*DIM_PARAM + j]) counterDensity++;
        }
    }
    float densityA = counterDensity / (double) (height*DIM_PARAM);
    
    counterDensity = 0;
    for(int j = 0; j < width; j++) {
        for(int i = 0; i < DIM_PARAM; i++) {
            B[i * width + j] = (Bb[j] >> 32 - i - 1) & 1;
            if (B[i * width + j]) counterDensity++;
        }
    }
    float densityB = counterDensity / (double) (DIM_PARAM * width);

    counterDensity = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // int intId = (i * height + j) / 32;
            // int intLane = (i * height + j) % 32;
            int intId = i / 32 * width + j;
            int intLane = i % 32;           
            C0[i * width + j] = (C0b[intId] >> 32 - intLane - 1) & 1;
            if (C0[i * width + j]) counterDensity++;
        }
    }
    
    cudaMemcpy(d_A, A, sizeof(element_t)*height*DIM_PARAM, cudaMemcpyHostToDevice);                           CUERR
    cudaMemcpy(d_B, B, sizeof(element_t)*width*DIM_PARAM, cudaMemcpyHostToDevice);                            CUERR   
    
    // Doing a check two times: once with A,B and once with Ab,Bb just to make sure
    // First check
    element_t *C_test_GPU = (element_t *) malloc(sizeof(element_t) * height * width);
    element_t *d_C_test_GPU;
    cudaMalloc((void **) &d_C_test_GPU, sizeof(element_t) * height * width);                             CUERR
    
    matrixMultiply <<< SDIV(width, THREADSPERBLOCK), THREADSPERBLOCK >>> 
                        (d_Ab, d_Bb, d_C_test_GPU, height, width);                                      CUERR
                        
    cudaMemcpy(C_test_GPU, d_C_test_GPU, sizeof(element_t) * height * width, 
                    cudaMemcpyDeviceToHost);                                                            CUERR
    
    int error_test_GPU = 0;
    for (int i = 0; i < height * width; i++)
        error_test_GPU += (((C0[i] - C_test_GPU[i]) * (C0[i] - C_test_GPU[i])));

    // Second check with regular matrix multiplication
    dim3 dimGrid( SDIV(width, TILE_WIDTH), SDIV(height, TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiplyInt <<< dimGrid, dimBlock >>> 
                            (d_A, d_B, d_C_test_GPU, height, width, DIM_PARAM);                         CUERR

    cudaMemcpy(C_test_GPU, d_C_test_GPU, sizeof(element_t) * height * width, cudaMemcpyDeviceToHost);         CUERR
    int error_test_GPU_2 = 0;
    for (int i = 0; i < height * width; i++)
        error_test_GPU_2 += (((C0[i] - C_test_GPU[i]) * (C0[i] - C_test_GPU[i])));
    
    TIMERSTOP(aftertestGPU)
    printf("Aftertest error between C0 and C on GPU (bitwise):\t%f, %i wrong entires\n",
           (double) error_test_GPU / (height*width), error_test_GPU);
    printf("Aftertest error between C0 and C on GPU (float):  \t%f, %i wrong entires\n",
           (double) error_test_GPU_2 / (height*width), error_test_GPU_2);
    printf("Density A: %f, Density B: %f\n", densityA, densityB);
    
}


// Used for debugging and checking, not optimized
template<typename bit_vector_t, typename element_t = uint32_t>
void aftertestCPU(  bit_vector_t *Ab, bit_vector_t *Bb, bit_vector_t *d_Ab, bit_vector_t *d_Bb, bit_vector_t *C0b, 
                    int height, int width) {    
    TIMERSTART(aftertestCPU)
    element_t *A = (element_t*)malloc(sizeof(element_t) * DIM_PARAM * height);
    element_t* B = (element_t*)malloc(sizeof(element_t) * width * DIM_PARAM);
    element_t* C0 = (element_t*)malloc(sizeof(element_t) * height * width);
    element_t *C_test_CPU = (element_t *) malloc(sizeof(element_t) * height * width);

    element_t *d_A, *d_B;
    element_t *d_C_test_CPU;
    cudaMalloc((void**)&d_A, sizeof(element_t) * DIM_PARAM * height);                         CUERR
    cudaMalloc((void**)&d_B, sizeof(element_t)*width * DIM_PARAM);                            CUERR
    cudaMalloc((void**) &d_C_test_CPU, sizeof(element_t) * height * width);              CUERR

    
    for(int i=0; i<height;i++)
        for(int j=0;j<DIM_PARAM;j++)
            A[i*DIM_PARAM + j] = (Ab[i] >> 32 - j - 1) & 1;

    for(int i=0;i<width;i++)
        for(int j=0;j<DIM_PARAM;j++)
            B[j*width+i] = (Bb[i] >> 32 - j - 1) & 1;
        
    int intId;
    int intLane;
    for(int i=0; i<width; i++){
        for(int j=0;j<height;j++){
             intId = (i*height + j) / 32;
             intLane = (i*height + j) % 32;
             C0[j*width + i] = (C0b[intId] >> 32 - intLane - 1) & 1;
        }
    }

    
    cudaMemcpy(d_A, A, sizeof(element_t) * height * DIM_PARAM, cudaMemcpyHostToDevice);  CUERR
    cudaMemcpy(d_B, B, sizeof(element_t) * width * DIM_PARAM, cudaMemcpyHostToDevice);   CUERR
    cudaMemcpy(d_Ab, Ab, sizeof(bit_vector_t) * height, cudaMemcpyHostToDevice);            CUERR
    cudaMemcpy(d_Bb, Bb, sizeof(bit_vector_t) * width, cudaMemcpyHostToDevice);             CUERR
    
    // Doing a check two times: once with A,B and once with Ab,Bb just to make sure
    
    matrixMultiply <<< width / THREADSPERBLOCK + 1, THREADSPERBLOCK >>> 
                        (d_Ab, d_Bb, d_C_test_CPU, height, width);                      CUERR
    
    cudaMemcpy(C_test_CPU, d_C_test_CPU, sizeof(element_t) * height * width, 
                    cudaMemcpyDeviceToHost);                                            CUERR
                    
    int distance_test_CPU = 0;
    for (int i = 0; i < height * width; i++)
        distance_test_CPU += ((C0[i] - C_test_CPU[i]) * (C0[i] - C_test_CPU[i]));
    
    dim3 dimGrid((width-1)/TILE_WIDTH+1, (height-1)/TILE_WIDTH+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiplyInt <<< dimGrid, dimBlock >>> 
                (d_A, d_B, d_C_test_CPU, height, width, DIM_PARAM);                     CUERR
    cudaMemcpy(C_test_CPU, d_C_test_CPU, sizeof(element_t) * height * width, 
                    cudaMemcpyDeviceToHost);                                            CUERR
    int distance_test_CPU_2 = 0;
    for (int i = 0; i < height * width; i++) 
        distance_test_CPU_2 += ((C0[i] - C_test_CPU[i]) * (C0[i] - C_test_CPU[i]));
    
    TIMERSTOP(aftertestCPU)
    printf("Aftertest error between C0 and C on CPU (bitwise): %i\n", distance_test_CPU);
    printf("Aftertest error between C0 and C on CPU (float): %i\n", distance_test_CPU_2);
}

#endif

