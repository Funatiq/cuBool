#include <iostream>
#include <bitset>


#include "helper/config.h"
#define DIM_PARAM 4
#include "helper/rngpu.hpp"
#include "helper/io_and_allocation.hpp"

using my_bit_vector_t = uint32_t; // only tested uint32_t

using std::cout;
using std::endl;

int main(int argc, char const *argv[])
{
    my_bit_vector_t *A0b, *B0b, *C0b;
	int width = 5;
	int height = 5;
	float density;
	generate_random_matrix(height, width, 2, A0b, B0b, C0b, density);

	for(int i = 0; i < height; ++i) {
		cout << bitset<DIM_PARAM>(A0b[i] >> 32 - DIM_PARAM) << endl;
	}
	cout << endl;

	for(int k = 0; k < DIM_PARAM; ++k) {
		for(int j = 0; j < width; ++j) {
			// cout << bitset<DIM_PARAM>(B0b[j] >> 32 - DIM_PARAM) << endl;
			cout << ((B0b[j] >> k) & 1);
		}
		cout << endl;
	}
	cout << endl;

	// for(int j = 0; j < width; ++j) {
	// 	cout << bitset<5>(C0b[j] >> 32 - height) << endl;
	// }
	// cout << endl;

	for(int i = 0; i < height; ++i) {
		for(int j = 0; j < width; ++j) {
			cout << ((C0b[j] >> 32 - i - 1) & 1);
		}
		cout << endl;
	}


	return 0;
}