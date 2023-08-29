#pragma once 

#include <iostream> // For std::cout and others 
#include <fstream> // For reading files
#include <string> // For strings
#include <vector> // For vectors
#include <sstream> // For read/writing strings


#include <sys/stat.h> // For Function fExist
#include <unistd.h> // For Function fExist


using namespace std;

bool fExist(const string& name);

void readFile(string filename, vector<vector<string>> &data);
char*** readFile(string filename, char*** data);

void rmSpace(string &str);

void printData(vector<vector<string>> data);

void printData(char*** data);