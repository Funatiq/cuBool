#ifndef IO_AND_ALLOCATION
#define IO_AND_ALLOCATION

#include <string>
#include <sstream>
#include <fstream>

#include "config.h"
#include "rngpu.hpp"

#ifndef SDIV
// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))
// #define SDIV(x,y)(((x)-1)/(y)+1)
#endif

using namespace std;

template<typename bit_vector_t>
void generate_random_matrix(int height, int width, int num_kiss, bit_vector_t * &Ab, bit_vector_t * &Bb, bit_vector_t * &C0b, float &density) {

    Ab = (bit_vector_t *) malloc(sizeof(bit_vector_t) * height);
    Bb = (bit_vector_t *) malloc(sizeof(bit_vector_t) * width);

    uint32_t seed = 42;
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);

    bit_vector_t bit_vector_mask = bit_vector_t(~0) << (32-DIM_PARAM);

    for(int i=0; i < height; ++i) {
        Ab[i] = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            Ab[i] &= fast_kiss32(&state);
    }
    for(int j=0; j < width; ++j) {
        Bb[j] = bit_vector_mask;
        for(int kiss = 0; kiss < num_kiss; ++kiss)
            Bb[j] &= fast_kiss32(&state);
    }

    // Malloc for C0b
    int padded_height_32 = SDIV(height, 32);
    int sizeCb = padded_height_32 * width;
    // int sizeC = SDIV(height * width, 32);
    C0b = (bit_vector_t *) malloc(sizeof(bit_vector_t) * sizeCb);
    
    // Set all entries 0
    for (int i = 0; i < sizeCb; i++)
        C0b[i] = 0;

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


    // Create C
    // for(int j=0; j < width; ++j) {
    //     for(int i=0; i < height; i+=sizeof(bit_vector_t)) {
    //         bit_vector_t cj = 0;
    //         for(int b=0; b < sizeof(bit_vector_t); ++b) {
    //             cj <<= 1;
    //             if(i+b < height) {
    //                 if(A[i+b] & B[j]) {
    //                     cj |= 1;
    //                     ++nonzeroelements;
    //                 }
    //             }
    //         }
    //         (*C0b)[] = cj
    //     }
    // }
    
    density = (float) nonzeroelements / (height * width);
       
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    printf("MATRIX CREATION COMPLETE\n");
    printf("Height: %i\nWidth: %i\nNon-zero elements: %i\nDensity: %f\n",
           height, width, nonzeroelements, density);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

template<typename bit_vector_t>
void readInputFileData( bit_vector_t *&C0b,
                    int &height, int &width, 
                    float &density, string filename) {
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
    int padded_height_32 = SDIV(height,32);
    int sizeCb = padded_height_32 * width;
    C0b = (bit_vector_t *) malloc(sizeof(bit_vector_t) * sizeCb);
    
    // Set all entries 0
    for (int i = 0; i < sizeCb; i++)
        C0b[i] = 0;

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
template<typename bit_vector_t, typename element_t = uint8_t>
void writeToFiles(bit_vector_t* Ab, bit_vector_t* Bb, int height, int width)
{
    element_t *A = (element_t*) malloc(sizeof(element_t) * DIM_PARAM * height);
    element_t *B = (element_t*) malloc(sizeof(element_t) * width * DIM_PARAM);

    for(int i = 0; i < height; i++)
        for(int j = 0; j < DIM_PARAM; j++)
            A[i * DIM_PARAM + j] = (Ab[i] >> 32 - j - 1) & 1;
    
    for(int i = 0; i < width; i++) 
        for(int j = 0; j < DIM_PARAM; j++) 
            B[j * width + i] = (Bb[i] >> 32 - j - 1) & 1;
    
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
        myfile << height << "," << DIM_PARAM << "\n";
        for (int i = 0; i < height; i++){
            for (int j = 0; j < DIM_PARAM; j++){
                myfile << A[i * DIM_PARAM + j] << ((j != DIM_PARAM - 1) ? "," : "");
            }
            myfile << "\n";
        }
        myfile.close();
    }
    
    ofstream myfile2(b);
    if(myfile2.is_open()){
        myfile2 << DIM_PARAM << "," << width << "\n";
        for (int i = 0; i<DIM_PARAM; i++){
            for (int j = 0; j < width; j++){
                myfile2 << B[i * width + j] << ((j != width - 1) ? "," : "");
            }
            myfile2 << "\n";
        }
        myfile2.close();
    }   
    cout << "Writing to files \"" << a << "\" and \"" << b << "\" complete" << endl;
}


// Initialization of A and B
template<typename bit_vector_t>
void initializeFactors( bit_vector_t *&Ab, bit_vector_t *&Bb, 
                        int height, int width,
                        int padded_height, int padded_width, 
                        float density, fast_kiss_state32_t *state) {

    Ab = (bit_vector_t *) malloc(sizeof(bit_vector_t) * padded_height);
    Bb = (bit_vector_t *) malloc(sizeof(bit_vector_t) * padded_width);

    // Initialize A and B and copy to device
    bool threshold;
    for (int i = 0; i < padded_height; i++) {
        Ab[i] = 0;
        if (i < height) {
            #pragma unroll
            for (int j = 0; j < DIM_PARAM; j++) {
                switch(INITIALIZATIONMODE) {
                    case 1: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                            < (sqrt(1 - pow(1 - density, 1 / (double) DIM_PARAM)));
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
    }

    for (int i = 0; i < padded_width; i++) {
        Bb[i] = 0;
        if (i < width) {
            #pragma unroll
            for (int j = 0; j < DIM_PARAM; j++) {
                switch(INITIALIZATIONMODE) {
                    case 1: threshold = (fast_kiss32(state) / (double) UINT32_MAX) 
                                            < (sqrt(1 - pow(1 - density, 1 / (double) DIM_PARAM)));
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
    }
    
    printf("Initialization of A and B complete\n");
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

#endif