#ifndef IO_AND_ALLOCATION
#define IO_AND_ALLOCATION

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <bitset>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath> // log2

#include "config.h"
#include "helper/rngpu.hpp"

// safe division
#ifndef SDIV
#define SDIV(x,y)(((x)+(y)-1)/(y))
// #define SDIV(x,y)(((x)-1)/(y)+1)
#endif

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;


float getInitChance(float density, uint8_t factorDim) {
    float threshold;

    switch(INITIALIZATIONMODE) {
        case 1:
            threshold = (sqrt(1 - pow(1 - density, float(1) / factorDim)));
            break;
        case 2:
            threshold = (density / 100);
            break;
        case 3:
            threshold = (density);
            break;
        default:
            threshold = 0;
            break;
    }
    return threshold;
}

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
    
    density = float(nonzeroelements) / height / width;
       
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
    size_t padded_height_32 = SDIV(height, 32);
    size_t sizeCb = padded_height_32 * width;
    // size_t sizeC = SDIV(height * width, 32);

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
    
    density = float(nonzeroelements) / height / width;
       
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
    std::ifstream is {filename};

    if(!is.good()) throw std::runtime_error{"File " + filename +
                                            " could not be opened!"};

    std::uint64_t ones = 0;
    is >> height >> width >> ones;

    int padded_height_32 = SDIV(height, 32);
    int sizeCb = padded_height_32 * width;

    C0b.clear();
    C0b.resize(sizeCb,0);

    vector<uint32_t> row_permutation(height);
    std::iota(row_permutation.begin(), row_permutation.end(), 0);
    std::shuffle(row_permutation.begin(), row_permutation.end(), std::mt19937{std::random_device{}()});

    int nonzeroelements = 0;
    for(; ones > 0; --ones) {
        std::uint64_t r, c;
        is >> r >> c;
        // r = row_permutation[r];
        int vecId = r / 32 * width + c;
        int vecLane = r % 32;
        C0b[vecId] |= 1 << vecLane;
        nonzeroelements++;
    }
    
    density = float(nonzeroelements) / height / width;

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

    float threshold = getInitChance(density, factorDim);

    const int rand_depth = -log2(threshold)+1;
    // const int rand_depth = 5;

    cout << "Init threshold: " << threshold << endl;
    cout << "Init rand depth: " << rand_depth << " -> " << pow(2, -rand_depth) << endl;

    if(rand_depth < 15) {
        const uint32_t factorMask = UINT32_MAX >> (32-factorDim);

        int counter = 0;
        for (int i = 0; i < height; i++) {
            Ab[i] = factorMask;
            for(int d = 0; d < rand_depth; ++d) {
                Ab[i] &= fast_kiss32(state);
            }
            if(Ab[i]) ++counter;
        }
        cout << "# Ai != 0: " << counter << endl;

        counter = 0;
        for (int j = 0; j < width; j++) {
            Bb[j] = factorMask;
            for(int d = 0; d < rand_depth; ++d) {
                Bb[j] &= fast_kiss32(state);
            }
            if(Bb[j]) ++counter;
        }
        cout << "# Bj != 0: " << counter << endl;
    }
    
    cout << "Initialization of A and B complete\n";
    for(int i=0; i<38; ++i)
        cout << "- ";
    cout << endl;
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

    uint32_t threshold = UINT32_MAX * getInitChance(density, factorDim);

    cout << "Init threshold: " << float(threshold)/UINT32_MAX << endl;

    // Initialize A and B
    int counter = 0;
    for (int i = 0; i < height; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            if (fast_kiss32(state) < threshold)
                Ab[i] |= 1 << j;
        }
        if(Ab[i]) ++counter;
    }
    cout << "# Ai != 0: " << counter << endl;

    counter = 0;
    for (int i = 0; i < width; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            if (fast_kiss32(state) < threshold)
                Bb[i] |= 1 << j;
        }
        if(Bb[i]) ++counter;
    }
    cout << "# Bj != 0: " << counter << endl;

    
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

    uint32_t threshold = UINT32_MAX * getInitChance(density, factorDim);

    // Initialize A and B
    for (int i = 0; i < height; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            A[i * factorDim + j] = fast_kiss32(state) < threshold;
        }
    }

    for (int i = 0; i < width; i++) {
        #pragma unroll
        for (int j = 0; j < factorDim; j++) {
            B[i * factorDim + j] = fast_kiss32(state) < threshold;
        }
    }
    
    printf("Initialization of A and B complete\n");
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// Write result factors to file
void writeFactorsToFiles(const string& filename,
                         const vector<uint32_t>& Ab,
                         const vector<uint32_t>& Bb,
                         const uint8_t factorDim)
{
    using std::stringstream;
    using std::bitset;
    using std::ofstream;

    time_t now = time(0);
    // char* dt = ctime(&now);
    tm *ltm = localtime(&now);

    stringstream date;
    date << 1+ltm->tm_mon << '-' << ltm->tm_mday << '_' << ltm->tm_hour << ':' << ltm->tm_min << ':' << ltm->tm_sec;
    
    stringstream filename_A;
    filename_A << filename << "_factor_A_" << date.str() << ".data";
    stringstream filename_B;
    filename_B << filename << "_factor_B_" << date.rdbuf() << ".data";

    size_t height = Ab.size();

    int nonzeroelements = 0;
    for (int i = 0; i < height; i++){
        bitset<32> row(Ab[i]);
        nonzeroelements += row.count();
    }
    
    ofstream os_A(filename_A.str());
    if (os_A.good()){
        os_A << height << " " << int(factorDim) << " " << nonzeroelements << "\n";
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
    
    size_t width = Bb.size();

    nonzeroelements = 0;
    for (int j = 0; j < width; j++){
        bitset<32> col(Bb[j]);
        nonzeroelements += col.count();
    }

    ofstream os_B(filename_B.str());
    if(os_B.good()){
        os_B  << width << " " << int(factorDim) << " " << nonzeroelements << "\n";
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

template<typename distance_t>
void writeDistancesToFile(const string& filename,
                          const vector<distance_t>& distances)
{
    using std::stringstream;
    using std::bitset;
    using std::ofstream;

    time_t now = time(0);
    // char* dt = ctime(&now);
    tm *ltm = localtime(&now);

    stringstream date;
    date << 1+ltm->tm_mon << '-' << ltm->tm_mday << '_' << ltm->tm_hour << ':' << ltm->tm_min << ':' << ltm->tm_sec;

    stringstream filename_d;
    filename_d << filename << "_distances_" << date.str() << ".txt";

    ofstream os(filename_d.str());
    if (os.good()){
        for (int i = 0; i < distances.size(); i++){
            if(i>0) os << "\n";
            os << distances[i];
        }
        os.close();
    } else {
        cerr << "File " << filename_d.str() << " could not be openend!" << endl;
    }

    cout << "Writing to files \"" << filename_d.rdbuf() << "\" complete" << endl;
}


#endif
