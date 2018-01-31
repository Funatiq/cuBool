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
    ifstream infileMovie("movies10m.dat");
    ifstream infileRatings("ratings10m.dat");
    string linestring;
    string field;
    
    int movieIds[65134];
    for (unsigned int i=0;i<65134;i++)
        movieIds[i] = 0;
    
    string userId;
    int nonzeroentries = 0;
    std::vector<int> userVector;
    while (getline(infileRatings, linestring)) {
        stringstream sep1(linestring);
        string fieldtemp;
        getline(sep1, userId, ':');
        userVector.push_back(stoi(userId));
        getline(sep1, fieldtemp, ':');
        getline(sep1, fieldtemp, ':');
        movieIds[stoi(fieldtemp, nullptr)] = 1;
        nonzeroentries++;
    }
    std::vector<int>::iterator it;
    it = std::unique(userVector.begin(), userVector.end());
    userVector.resize(std::distance(userVector.begin(),it));
    int totalNumberUsers = userVector.size();

    
    int movieCounter = 0;
    for(int i=0;i<65134;i++){
        if(movieIds[i] == 1) {
            movieIds[i] = movieCounter;
            movieCounter++;
        }
    }
    cout << "Actual Numbers of movies: " << movieCounter << ", actual number users: " << totalNumberUsers  << endl;
    infileRatings.close();
    
    ifstream infileRatings2("ratings10m.dat");
    ofstream outputRatings("ratings10m_cleaned.dat");
    outputRatings << totalNumberUsers << "," << movieCounter << "\n";
    int userCounter = 0;
    int userTemp = 0;
    while (getline(infileRatings2, linestring)) {
        stringstream sep1(linestring);
        string userId;
        string movieId;
        string movieRating;
        getline(sep1, userId, ':');
        getline(sep1, movieId, ':');
        getline(sep1, movieId, ':');
        getline(sep1, movieRating, ':');
        getline(sep1, movieRating, ':');
        //cout << movieRating << endl;
        
        if (stoi(userId, nullptr) != userTemp)
            userCounter++;
        
        userTemp = stoi(userId, nullptr);
 
        if (stoi(movieRating, nullptr) > 2.5) {
            outputRatings << userCounter - 1 << "," << movieIds[stoi(movieId, nullptr)] << "\n";
        } else {
            nonzeroentries--;
        }
    }
}