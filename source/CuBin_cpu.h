#ifndef CUBIN_CPU_H
#define CUBIN_CPU_H

#include <vector>
#include <iostream>

void computeDistanceCPU(const vector<uint32_t> &Ab,
                        const vector<uint32_t> &Bb,
                        const vector<uint32_t> &Cb,
                        const int height,
                        const int width)
{
    int error = 0;

    #pragma omp parallel for reduction(+:error)
    for(int j=0; j < width; ++j) {
        uint32_t col = Bb[j];
        for(int i=0; i < height; ++i) {
            const int lineSum = (Ab[i] & col) ? 1 : 0;

            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int truthEntry = (Cb[vecId] >> vecLane) & 1;

            const int error_ij = lineSum ^ truthEntry;
            // if(error_ij != 0 && error_ij != 1)
                // std::cout << "bad error: i=" << i << " j=" << j << " error_ij=" << error_ij << std::endl;
            error += error_ij;
        }
    }

    std::cout << "cpu error: " << error << std::endl;
}

struct coo {
    coo(uint32_t x, uint32_t y) : x_{x}, y_{y} {}

    uint32_t x_;
    uint32_t y_;
};

vector<coo> computeProduct(const vector<uint32_t> &Ab,
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

#endif
