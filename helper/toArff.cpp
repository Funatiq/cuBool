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
    ifstream infileRatings(argv[1]);
    string linestring;
    string field;
    
    ofstream outputCSV(argv[2]);
    std::vector<int> userIds;
    std::vector<int> movieIds;
    
    int height;
    int width;
    getline(infileRatings, linestring);
    stringstream sep(linestring);
    getline(sep, field, ',');
    (height) = stoi(field, nullptr);
    getline(sep, field, '\n');
    (width) = stoi(field, nullptr);

    while (getline(infileRatings, linestring)) {
        stringstream sep1(linestring);
        string userId;
        string movieId;
        getline(sep1, userId, ',');
        getline(sep1, movieId, '\n');
        
        userIds.push_back(stoi(userId, nullptr));
        movieIds.push_back(stoi(movieId, nullptr));
    }
    
    cout << "VEctor size:" << userIds.size() << ", " << movieIds.size();
    
    int vectorCounter = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if((userIds[vectorCounter] == y) && (movieIds[vectorCounter] == x)){
                outputCSV << 1;
                vectorCounter++;
            } else {
                outputCSV << 0;
            }
            
            if(x == width - 1) {
                outputCSV << '\n';
            } else {
                outputCSV << ',';
            }
        }
    }
    
}