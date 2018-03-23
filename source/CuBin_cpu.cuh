#ifndef CUBIN_CPU_CUH
#define CUBIN_CPU_CUH

#include "helper/matrix_mult.cuh"

// CPU computation
template<typename bit_vector_t>
void CPUcomputation(bit_vector_t *Ab, bit_vector_t *Bb, bit_vector_t *C0, 
                    int width, int height, 
                    int startDistance, uint32_t seed, int updateStep,
                    float threshold, int linesAtOnce) {
                        
    int *hDistance = &startDistance;
    fast_kiss_state32_t state;
    state = get_initial_fast_kiss_state32(seed);
    int toBeChanged;
    int iterations = 0;
    int iterationsNoImp = 0;
    int maxIterationsNoImp = (std::max(width,height) / linesAtOnce + 1) * 1000; 
    int error_before = startDistance;

    TIMERSTART(CPUcomputation)
    printf("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("- - - - Starting CPU opimization, showing error every %i steps - - - - -\n", updateStep);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    while (*hDistance > threshold && iterations < CPUITERATIONS && iterationsNoImp < maxIterationsNoImp) {

        iterations++;
        if (iterations % updateStep == 0)
            printf("Current Distance: %i\n", *hDistance);
        
        // Change row
        toBeChanged = ((uint32_t) fast_kiss32(&state)) % height;
        CPUvectorMatrixMultCompareRow(Ab, Bb, C0, width, height, toBeChanged, hDistance, &state, linesAtOnce);
        
        // Change col
        toBeChanged = ((uint32_t) fast_kiss32(&state)) % width;
        CPUvectorMatrixMultCompareCol(Ab, Bb, C0, width, height, toBeChanged, hDistance, &state, linesAtOnce);
        
        if (error_before - *hDistance == 0) {
            iterationsNoImp++;
        } else {
            iterationsNoImp = 0;
        }
        error_before = *hDistance;
        
    }
    printf("- - - - - - - - -\n");
    printf("End Distance on CPU: %i, Number of Iterations: %i,  Error remaining: %f\n", 
                *hDistance, iterations, *hDistance / (double) (height * width));
    TIMERSTOP(CPUcomputation)
}

template<typename bit_vector_t>
void CPUvectorMatrixMultCompareRow( bit_vector_t *Ab, bit_vector_t *Bb, 
                                    bit_vector_t *C0b, int width, int height, int startrow,
                                    int *hDistance, fast_kiss_state32_t *state, int rowsAtOnce) {
    int rowToBeChanged = startrow;
    int error;
    int cTruthEntry;
    int cEntryOld;
    int cEntryNew;
    bit_vector_t currentRow;
    bit_vector_t currentRow_changed;
    uint32_t randomNumber;

    // Change multiple lines from A
    for (int l = 0; l < rowsAtOnce; l++) {
        rowToBeChanged = (startrow + l) % height;
        currentRow = Ab[rowToBeChanged];
        currentRow_changed = currentRow;

        randomNumber = fast_kiss32(state);
        for (int i = 0; i < DIM_PARAM; i++)
            currentRow_changed ^= (randomNumber >> i) & 11 ?(0 << 21 - 1 - i) : (1 << 32 - 1 - i);
        
        error = 0;

        #pragma omp parallel for private(cEntryOld, cEntryNew, cTruthEntry) reduction(+: error)
        for (int tid = 0; tid < width; tid++) {
            bit_vector_t currentCol = Bb[tid];
            int intId = (tid * height + rowToBeChanged) / 32;
            int intLane = (tid * height + rowToBeChanged) % 32;
            cTruthEntry = (C0b[intId] >> 32 - intLane - 1) & 1;

            cEntryOld = (currentRow         & currentCol) > 0 ? 1 : 0;
            cEntryNew = (currentRow_changed & currentCol) > 0 ? 1 : 0;
            error += ((cEntryNew - cTruthEntry) * (cEntryNew - cTruthEntry)) -
                     ((cEntryOld - cTruthEntry) * (cEntryOld - cTruthEntry));
        }

        if (error < 0) {
            Ab[rowToBeChanged] = currentRow_changed;
            *hDistance = *hDistance + error;
        } 

    }
}

template<typename bit_vector_t>
void CPUvectorMatrixMultCompareCol( bit_vector_t *Ab, bit_vector_t *Bb, bit_vector_t *C0b, 
                                    int width, int height, int startcol,
                                    int *hDistance, fast_kiss_state32_t *state, int rowsAtOnce) {
    int colToBeChanged = startcol;
    int error;
    int cTruthEntry;
    int cEntryOld;
    int cEntryNew;
    bit_vector_t currentCol;
    bit_vector_t currentCol_changed;
    uint32_t randomNumber;

    // Change multiple cols from B
    for (int l = 0; l < rowsAtOnce; l++) {
        colToBeChanged = (colToBeChanged + l) % width;
        currentCol = Bb[colToBeChanged];
        currentCol_changed = currentCol;

        randomNumber = fast_kiss32(state);
        for (int i = 0; i < DIM_PARAM; i++)
            currentCol_changed ^= (randomNumber >> i) & 11 ? (0 << 32 - 1 - i) : (1 << 32 - 1 - i);
        
        error = 0;
        #pragma omp parallel for private(cEntryOld, cEntryNew, cTruthEntry) reduction(+: error)
        for (int tid = 0; tid < height; tid++) {
            bit_vector_t currentRow = Ab[tid]; 
            int intId = (colToBeChanged * height + tid) / 32;
            int intLane = (colToBeChanged * height + tid) % 32;
            cTruthEntry = (C0b[intId] >> 32 - intLane - 1) & 1;

            cEntryOld = (currentCol         & currentRow) > 0 ? 1 : 0;
            cEntryNew = (currentCol_changed & currentRow) > 0 ? 1 : 0;
            error += ((cEntryNew - cTruthEntry) * (cEntryNew - cTruthEntry)) - 
                     ((cEntryOld - cTruthEntry) * (cEntryOld - cTruthEntry));
        }

        if (error < 0) {
            Bb[colToBeChanged] = currentCol_changed;
            *hDistance = *hDistance + error;
        }              
    }
}

// Used for debugging and checking, not optimized
template<typename bit_vector_t, typename element_t = uint32_t>
void aftertestCPU(  bit_vector_t *Ab, bit_vector_t *Bb, bit_vector_t *d_Ab, bit_vector_t *d_Bb, bit_vector_t *C0b, 
                    int width, int height) {    
    TIMERSTART(aftertestCPU)
    element_t *A, *B, *C0;
    element_t *d_A, *d_B;
    element_t *C_test_CPU;
    element_t *d_C_test_CPU;
    A = (element_t*)malloc(sizeof(element_t) * DIM_PARAM * height);
    B = (element_t*)malloc(sizeof(element_t) * width * DIM_PARAM);
    C0 = (element_t*)malloc(sizeof(element_t) * width * height);
    C_test_CPU = (element_t *) malloc(sizeof(element_t) * width * height);                CUERR
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
                        (d_Ab, d_Bb, d_C_test_CPU, width, height);                      CUERR
    
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