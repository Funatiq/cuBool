#ifndef IO_AND_ALLOCATION
#define IO_AND_ALLOCATION

#include <string>
#include <sstream>
#include <fstream>
#include <bitset>

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
    uint32_t bit_vector_mask = uint32_t(~0) << (32-factorDim);

    Ab.clear();
    Ab.resize(height, bit_vector_mask);
    Bb.clear();
    Bb.resize(width, bit_vector_mask);

    uint32_t seed = 42;
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);

    for(int i=0; i < height; ++i) {
        // Ab[i] = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            Ab[i] &= fast_kiss32(&state);
    }
    for(int j=0; j < width; ++j) {
        // Bb[j] = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            Bb[j] &= fast_kiss32(&state);
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
                int intId = i / 32 * width + j;
                int intLane = i % 32;

                C0b[intId] |= 1 << (32 - 1 - intLane);

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
    uint32_t bit_vector_mask = uint32_t(~0) << (32-factorDim);
    
    A.clear();
    A.resize(height * factorDim, 0);
    B.clear();
    B.resize(width * factorDim, 0);

    uint32_t seed = 42;
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);

    for(int i=0; i < height; ++i) {
        uint32_t mask = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            mask &= fast_kiss32(&state);
        for(int k=0; k < factorDim; ++k)
            A[i * factorDim + k] = (mask >> 32 - 1 - k) & 1 ? 1 : 0;
    }
    for(int j=0; j < width; ++j) {
        uint32_t mask = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            mask &= fast_kiss32(&state);
        for(int k=0; k < factorDim; ++k)
            B[j * factorDim + k] = (mask >> 32 - 1 - k) & 1 ? 1 : 0;
    }

    // float threshold = 1.0f;
    // for(int kiss = 0; kiss < num_kiss; ++kiss)
    //     threshold /= 2.0f;

    // for(int i=0; i < height * factorDim; ++i) {
    //     float random = (float) fast_kiss32(&state) / UINT32_MAX;
    //     A[i] = random < threshold ? 1.0f : 0.0f;
    // }
    // for(int j=0; j < width * factorDim; ++j) {
    //     float random = (float) fast_kiss32(&state) / UINT32_MAX;
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
                    int intId = i / 32 * width + j;
                    int intLane = i % 32;

                    C0b[intId] |= 1 << (32 - 1 - intLane);

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
    ifstream infile;
    string linestring;
    string field;

    // First line: #height,#width,#non-zero-elements
    infile.open(filename);
    getline(infile, linestring);
    stringstream sep(linestring);
    getline(sep, field, ',');
    height = stoi(field, nullptr);
    getline(sep, field, ','); 
    width = stoi(field, nullptr);
    
    // Malloc for C0b
    int padded_height_32 = SDIV(height, 32);
    int sizeCb = padded_height_32 * width;

    // C0b = (uint32_t *) malloc(sizeof(uint32_t) * sizeCb);
    C0b.clear();
    C0b.resize(sizeCb,0);

    // Read rest of file
    int nonzeroelements = 0;
    while (getline(infile, linestring)) {
        stringstream sep1(linestring);
        string fieldtemp;
        getline(sep1, fieldtemp, ',');
        int x = stoi(fieldtemp, nullptr);
        getline(sep1, fieldtemp, ',');
        int y = stoi(fieldtemp, nullptr);
        int intId = x / 32 * width + y;
        int intLane = x % 32;
        C0b[intId] |= 1 << 32 - intLane - 1;
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

// Write result matrix in file
void writeToFiles(const vector<uint32_t> &Ab, const vector<uint32_t> &Bb, const int height, const int width, const uint8_t factorDim)
{
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%X", timeinfo);
    string str = buffer;
    
    string a = string("A_") + buffer + string(".data");
    string b = string("B_") + buffer + string(".data");
    
    ofstream myfile(a);
    if (myfile.is_open()){
        myfile << height << " " << factorDim << "\n";
        for (int i = 0; i < height; i++){
            // bitset<32> row(Ab[i] >> (32 - factorDim));
            // myfile << row << "\n";
            for(int k=0; k < factorDim; ++k)
                myfile << ((Ab[i] >> 32 - 1 - k) & 1 ? 1 : 0);
            myfile << "\n";
        }
        myfile.close();
    }
    
    ofstream myfile2(b);
    if(myfile2.is_open()){
        myfile2 << factorDim << " " << width << "\n";
        for (int j = 0; j<factorDim; j++){
            // bitset<32> col(Bb[j] >> (32 - factorDim));
            // myfile2 << col << "\n";
            for(int k=0; k < factorDim; ++k)
                myfile2 << ((Bb[j] >> 32 - 1 - k) & 1 ? 1 : 0);
            myfile2 << "\n";
        }
        myfile2.close();
    }   
    cout << "Writing to files \"" << a << "\" and \"" << b << "\" complete" << endl;
}

// Initialization of A and B
void initializeFactors( vector<uint32_t> &Ab, vector<uint32_t> &Bb, 
                        const int height, const int width,
                        const uint8_t factorDim,
                        const float density, fast_kiss_state32_t *state,
                        int padded_height = 0, int padded_width = 0)
{
    if (padded_height == 0)
        padded_height = height;
    if (padded_width == 0)
        padded_width = width;

    Ab.clear();
    Ab.resize(padded_height, 0);
    Bb.clear();
    Bb.resize(padded_width, 0);

    // Initialize A and B and copy to device
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
            Ab[i] |= threshold ? 1 << (32 - j - 1) : 0 ;
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
            Bb[i] |= threshold ? 1 << (32 - j - 1) : 0 ;
        }
    }
    
    printf("Initialization of A and B complete\n");
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// Initialization of A and B
void initializeFactors( vector<float> &A, vector<float> &B, 
                        const int height, const int width,
                        const uint8_t factorDim,
                        const float density, fast_kiss_state32_t *state,
                        int padded_height = 0, int padded_width = 0)
{
    if (padded_height == 0)
        padded_height = height;
    if (padded_width == 0)
        padded_width = width;

    A.clear();
    A.resize(padded_height * factorDim, 0);
    B.clear();
    B.resize(padded_width * factorDim, 0);

    // Initialize A and B and copy to device
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
                        const float density, fast_kiss_state32_t *state,
                        int padded_height = 0, int padded_width = 0)
{
    if (padded_height == 0)
        padded_height = height;
    if (padded_width == 0)
        padded_width = width;

    A.clear();
    A.resize(padded_height * factorDim, 0);
    B.clear();
    B.resize(padded_width * factorDim, 0);

    // Initialize A and B and copy to device
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
#endif