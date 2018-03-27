#ifndef CUBIN_CPU_CUH
#define CUBIN_CPU_CUH

#include "helper/matrix_mult.cuh"

// CPU computation
template<typename bit_vector_t>
void CPUcomputation(bit_vector_t *Ab, bit_vector_t *Bb, bit_vector_t *C0, 
                    int height, int width, 
                    int startDistance, uint32_t seed, int updateStep,
                    float threshold, int linesAtOnce) {
                        
    int *hDistance = &startDistance;
    fast_kiss_state32_t state;
    state = get_initial_fast_kiss_state32(seed);
    int toBeChanged;
    int iterations = 0;
    int iterationsNoImp = 0;
    int maxIterationsNoImp = (std::max(height, width) / linesAtOnce + 1) * 1000; 
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
        CPUvectorMatrixMultCompareRow(Ab, Bb, C0, height, width, toBeChanged, hDistance, &state, linesAtOnce);
        
        // Change col
        toBeChanged = ((uint32_t) fast_kiss32(&state)) % width;
        CPUvectorMatrixMultCompareCol(Ab, Bb, C0, height, width, toBeChanged, hDistance, &state, linesAtOnce);
        
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
                                    bit_vector_t *C0b, int height, int width, int startrow,
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
                                    int height, int width, int startcol,
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



#endif