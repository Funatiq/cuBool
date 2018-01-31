#include <fstream>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>     // std::cout
#include <algorithm>    // std::unique, std::distance
#include <vector>       // std::vector
using namespace std;
int main(int argc, char** argv){
    ifstream infileTSV(argv[1]);
    ofstream outputCSV(argv[2]);
    
    vector<uint32_t> x_values;
    vector<uint32_t> y_values;
    int x_counter = 0;
    int y_counter = 0;
    
    for (string row; getline(infileTSV, row, '\n'); ) {
        x_counter = 0;
        istringstream ss(row);
        for (string field; getline(ss, field, '\t'); ) {
            if (stoi(field) == 1) {
                x_values.push_back(x_counter);
                y_values.push_back(y_counter);
            }
            x_counter++;
        }
        y_counter++;
    }
    
    int width = x_counter;
    int height = y_counter;
    
    outputCSV << height << ',' << width << '\n';
    for ( int i = 0; i < x_values.size(); i++)
        outputCSV << y_values[i] << ',' << x_values[i] << '\n';
    
    infileTSV.close();
    outputCSV.close();
    
    return 0;

}