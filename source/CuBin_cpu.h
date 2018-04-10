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
            const int truthEntry = (Cb[vecId] >> (32 - 1 - vecLane)) & 1;

            const int error_ij = lineSum ^ truthEntry;
            // if(error_ij != 0 && error_ij != 1)
            	// std::cout << "bad error: i=" << i << " j=" << j << " error_ij=" << error_ij << std::endl;
            error += error_ij;
        }
    }

    std::cout << "cpu error: " << error << std::endl;
}

#endif
