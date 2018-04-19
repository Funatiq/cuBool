#ifndef IO_AND_ALLOCATION
#define IO_AND_ALLOCATION

#include <string>
#include <sstream>
#include <fstream>
#include <bitset>
#include <ctime>

#include "config.h"
#include "rngpu.hpp"

// safe division
#ifndef SDIV
#define SDIV(x,y)(((x)+(y)-1)/(y))
// #define SDIV(x,y)(((x)-1)/(y)+1)
#endif

using namespace std;

void generate_random_matrix(const int height, const int width, const uint8_t factorDim, const int num_kiss, 
                            vector<uint32_t> &Ab, vector<uint32_t> &Bb, vector<uint32_t> &C0b,
                            float &density)
{
    uint32_t bit_vector_mask = uint32_t(~0) >> (32-factorDim);

    Ab.clear();
    Ab.resize(height, bit_vector_mask);
    Bb.clear();
    Bb.resize(width, bit_vector_mask);

    uint32_t seed = 42;
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);

    for(int i=0; i < height; ++i) {
        // Ab[i] = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            Ab[i] &= fast_kiss32(state);
    }
    for(int j=0; j < width; ++j) {
        // Bb[j] = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            Bb[j] &= fast_kiss32(state);
    }

    // Malloc for C0b
    int padded_height_32 = SDIV(height, 32);
    int sizeCb = padded_height_32 * width;
    // int sizeC = SDIV(height * width, 32);

    C0b.clear();
    C0b.resize(sizeCb, 0);
    
    // Create C
    int nonzeroelements = 0;

    for(int j=0; j < width; ++j) {
        for(int i=0; i < height; ++i) {
            if(Ab[i] & Bb[j]) {
                // int index = j*height+i;
                int vecId = i / 32 * width + j;
                int vecLane = i % 32;

                C0b[vecId] |= 1 << vecLane;

                ++nonzeroelements;
            }
        }
    }
    
    density = (float) nonzeroelements / (height * width);
       
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("MATRIX CREATION COMPLETE\n");
    printf("Height: %i\nWidth: %i\nNon-zero elements: %i\nDensity: %f\n",
           height, width, nonzeroelements, density);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

void generate_random_matrix(const int height, const int width, const uint8_t factorDim, const int num_kiss, 
                            vector<float> &A, vector<float> &B, vector<uint32_t> &C0b,
                            float &density)
{
    uint32_t bit_vector_mask = uint32_t(~0) >> (32-factorDim);
    
    A.clear();
    A.resize(height * factorDim, 0);
    B.clear();
    B.resize(width * factorDim, 0);

    uint32_t seed = 42;
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);

    for(int i=0; i < height; ++i) {
        uint32_t mask = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            mask &= fast_kiss32(state);
        for(int k=0; k < factorDim; ++k)
            A[i * factorDim + k] = (mask >> k) & 1 ? 1 : 0;
    }
    for(int j=0; j < width; ++j) {
        uint32_t mask = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            mask &= fast_kiss32(state);
        for(int k=0; k < factorDim; ++k)
            B[j * factorDim + k] = (mask >> k) & 1 ? 1 : 0;
    }

    // float threshold = 1.0f;
    // for(int kiss = 0; kiss < num_kiss; ++kiss)
    //     threshold /= 2.0f;

    // for(int i=0; i < height * factorDim; ++i) {
    //     float random = (float) fast_kiss32(state) / UINT32_MAX;
    //     A[i] = random < threshold ? 1.0f : 0.0f;
    // }
    // for(int j=0; j < width * factorDim; ++j) {
    //     float random = (float) fast_kiss32(state) / UINT32_MAX;
    //     B[i] = random < threshold ? 1.0f : 0.0f;
    // }

    // Malloc for C0b
    int padded_height_32 = SDIV(height, 32);
    int sizeCb = padded_height_32 * width;
    // int sizeC = SDIV(height * width, 32);

    C0b.clear();
    C0b.resize(sizeCb, 0);
    
    // Create C
    int nonzeroelements = 0;

    for(int j=0; j < width; ++j) {
        for(int i=0; i < height; ++i) {
            for (int k=0; k < factorDim; ++k) {
                if((A[i * factorDim + k] > 0.5f) && (B[j * factorDim + k] > 0.5f)) {
                    // int index = j*height+i;
                    int vecId = i / 32 * width + j;
                    int vecLane = i % 32;

                    C0b[vecId] |= 1 << vecLane;

                    ++nonzeroelements;
                    break;
                }
            }
        }
    }
    
    density = (float) nonzeroelements / (height * width);
       
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("MATRIX CREATION COMPLETE\n");
    printf("Height: %i\nWidth: %i\nNon-zero elements: %i\nDensity: %f\n",
           height, width, nonzeroelements, density);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

void readInputFileData(const string filename,
                       vector<uint32_t> &C0b,
                       int &height, int &width, 
                       float &density)
{
    ifstream is {filename};

    if(!is.good()) throw std::runtime_error{"File " + filename +
                                            " could not be opened!"};

    std::uint64_t ones = 0;
    is >> height >> width >> ones;

    int padded_height_32 = SDIV(height, 32);
    int sizeCb = padded_height_32 * width;

    C0b.clear();
    C0b.resize(sizeCb,0);

    int nonzeroelements = 0;
    for(; ones > 0; --ones) {
        std::uint64_t r, c;
        is >> r >> c;
        int vecId = r / 32 * width + c;
        int vecLane = r % 32;
        C0b[vecId] |= 1 << vecLane;
        nonzeroelements++;
    }
    
    density = (double) nonzeroelements / (height * width);

    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("READING OF .DATA FILE COMPLETE\n");
    printf("Read height: %i\nRead width: %i\nNon-zero elements: %i\nDensity: %f\n",
           height, width, nonzeroelements, density);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
bool endsWith(const string& s, const string& suffix) {
    return s.rfind(suffix) == (s.size()-suffix.size());
}

// Initialization of A and B
void initializeFactors( vector<uint32_t> &Ab, vector<uint32_t> &Bb, 
                        const int height, const int width,
                        const uint8_t factorDim,
                        const float density,
                        fast_kiss_state32_t state)
{
    Ab.clear();
    Ab.resize(height, 0);
    Bb.clear();
    Bb.resize(width, 0);

    // for (int i = 0; i < height; i++) {
    //     Ab[i] = fast_kiss32(state) >> (32-factorDim);
    // }

    // for (int j = 0; j < width; j++) {
    //     Bb[j] = fast_kiss32(state) >> (32-factorDim);
    // }
    
    printf("Initialization of A and B complete\n");
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// Initialization of A and B
void initializeFactors2( vector<uint32_t> &Ab, vector<uint32_t> &Bb, 
                        const int height, const int width,
                        const uint8_t factorDim,
                        const float density,
                        fast_kiss_state32_t state)
{
    Ab.clear();
    Ab.resize(height, 0);
    Bb.clear();
    Bb.resize(width, 0);

    // Initialize A and B
    bool threshold;
    for (int i = 0; i < height; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            switch(INITIALIZATIONMODE) {
                case 1: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (sqrt(1 - pow(1 - density, 1 / (double) factorDim)));
                                        break;
                case 2: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (density / (double) 100);
                                        break;
                case 3: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < density;
                                        break;
            }
            if (threshold) Ab[i] |= 1 << j;
        }
    }

    for (int i = 0; i < width; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            switch(INITIALIZATIONMODE) {
                case 1: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (sqrt(1 - pow(1 - density, 1 / (double) factorDim)));
                                        break;
                case 2: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (density / (double) 100);
                                        break;
                case 3: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < density;
                                        break;
            }
            if (threshold) Bb[i] |= 1 << j;
        }
    }
    
    printf("Initialization of A and B complete\n");
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// Initialization of A and B
void initializeFactors( vector<float> &A, vector<float> &B, 
                        const int height, const int width,
                        const uint8_t factorDim,
                        const float density,
                        fast_kiss_state32_t state)
{
    A.clear();
    A.resize(height * factorDim, 0);
    B.clear();
    B.resize(width * factorDim, 0);

    // Initialize A and B
    for (int i = 0; i < height; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            A[i * factorDim + j] = (float) fast_kiss32(state) / UINT32_MAX;
        }
    }

    for (int i = 0; i < width; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            B[i * factorDim + j] = (float) fast_kiss32(state) / UINT32_MAX;
        }
    }
    
    printf("Initialization of A and B complete\n");
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// Initialization of A and B
void initializeFactors2( vector<float> &A, vector<float> &B,
                         const int height, const int width,
                         const uint8_t factorDim,
                         const float density,
                         fast_kiss_state32_t state)
{
    A.clear();
    A.resize(height * factorDim, 0);
    B.clear();
    B.resize(width * factorDim, 0);

    // Initialize A and B
    bool threshold;
    for (int i = 0; i < height; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            switch(INITIALIZATIONMODE) {
                case 1: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (sqrt(1 - pow(1 - density, 1 / (double) factorDim)));
                                        break;
                case 2: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (density / (double) 100);
                                        break;
                case 3: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < density;
                                        break;
            }
            A[i * factorDim + j] = threshold;
        }
    }

    for (int i = 0; i < width; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            switch(INITIALIZATIONMODE) {
                case 1: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (sqrt(1 - pow(1 - density, 1 / (double) factorDim)));
                                        break;
                case 2: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < (density / (double) 100);
                                        break;
                case 3: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                        < density;
                                        break;
            }
            B[i * factorDim + j] = threshold;
        }
    }
    
    printf("Initialization of A and B complete\n");
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// Write result matrix in file
void writeToFiles(const string filename,
                  const vector<uint32_t> &Ab,
                  const vector<uint32_t> &Bb,
                  const int height,
                  const int width,
                  const uint8_t factorDim)
{
    time_t now = time(0);
    // char* dt = ctime(&now);
    tm *ltm = localtime(&now);

    stringstream date;
    date << 1+ltm->tm_mon << '-' << ltm->tm_mday << '_' << ltm->tm_hour << ':' << ltm->tm_min << ':' << ltm->tm_sec;
    
    stringstream filename_A;
    filename_A << filename << "_factor_A_" << date.str() << ".data";
    stringstream filename_B;
    filename_B << filename << "_factor_B_" << date.rdbuf() << ".data";

    int nonzeroelements = 0;
    for (int i = 0; i < height; i++){
        bitset<32> row(Ab[i]);
        nonzeroelements += row.count();
    }
    
    ofstream os_A(filename_A.str());
    if (os_A.good()){
        os_A << height << " " << (int)factorDim << " " << nonzeroelements << "\n";
        for (int i = 0; i < height; i++){
            // bitset<32> row(Ab[i] >> (32 - factorDim));
            // os_A << row << "\n";
            for(int k=0; k < factorDim; ++k)
                os_A << ((Ab[i] >> k) & 1 ? 1 : 0);
            os_A << "\n";
        }
        os_A.close();
    } else {
        cerr << "File " << filename_A.str() << " could not be openend!" << endl;
    }
    
    nonzeroelements = 0;
    for (int j = 0; j < width; j++){
        bitset<32> col(Bb[j]);
        nonzeroelements += col.count();
    }

    ofstream os_B(filename_B.str());
    if(os_B.good()){
        os_B  << width << " " << (int)factorDim << " " << nonzeroelements << "\n";
        for (int j = 0; j < width; j++){
            // bitset<32> col(Bb[j] >> (32 - factorDim));
            // os_B << col << "\n";
            for(int k=0; k < factorDim; ++k)
                os_B << ((Bb[j] >> k) & 1 ? 1 : 0);
            os_B << "\n";
        }
        os_B.close();
    } else {
        cerr << "File " << filename_B.str() << " could not be openend!" << endl;
    }
    
    cout << "Writing to files \"" << filename_A.rdbuf() << "\" and \"" << filename_B.rdbuf() << "\" complete" << endl;
}

#endif
