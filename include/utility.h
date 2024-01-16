#pragma once

#include <iostream> // For std::cout
#include <fstream> // For reading files
#include <string> // For strings
#include <sstream> // For read/writing strings
#include <math.h> // For sqrt() pow()
#include <vector> // For vectors

#include "params.h"


///////////////////////////////////////////////////////////////////////////////
/// Misc Utility
///////////////////////////////////////////////////////////////////////////////
double getMax(std::vector<double> v);
int getMax(std::vector<int> v);

bool match(std::vector<double> A, std::vector<double> B);
bool match(char * A, std::string B);

///////////////////////////////////////////////////////////////////////////////
/// Files
///////////////////////////////////////////////////////////////////////////////
void rmSpace(std::string &str);

///////////////////////////////////////////////////////////////////////////////
/// Print
///////////////////////////////////////////////////////////////////////////////
// template <typename T> void print(std::vector<T> data);
// template <typename T> void print(std::string head, std::vector<T> data);
// template <typename T> void print(std::vector<std::vector<T>> data);
// template <typename T> void print(std::string head, std::vector<std::vector<T>> data);

// Print Doubles
void print(std::string head, double data);
void print(std::vector<double> data);
void print(std::string head, std::vector<double> data);
void print(std::vector<std::vector<double>> data);
void print(std::string head, std::vector<std::vector<double>> data);


// Print to file
void write(std::string filename, std::vector<double> data);
void writeTo(std::string filename, std::vector<double> data);
void writeLine(std::string filename, std::vector<double> data);
void writeLineTo(std::string filename, std::vector<double> data);

// print WB
void printWB(
  std::vector<std::vector<double>> W, 
  std::vector<std::vector<double>> B, 
  std::vector<unsigned int> nNodes
);

///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
///////////////////////////////////////////////////////////////////////////////
int sumVectR(std::vector<bool> A);

// Add Vectors 
void addVec(std::vector<double> &A, std::vector<double> B);
std::vector<double> addVecR(std::vector<double> A, std::vector<double> B);

// Subtract Vectors
void subVec(std::vector<double> &A, std::vector<double> B);
std::vector<double> subVecR(std::vector<double> A, std::vector<double> B);

// Multiply Vectors Element-wise
void multVec(std::vector<double> &A, std::vector<double> B);
std::vector<double> multVecR(std::vector<double> &A, std::vector<double> B);

// Multiply Scaler to Vector
void multScal(std::vector<double> &A, double scaler);
std::vector<double> multScalR(std::vector<double> A, double scaler);

// Append two vectors
void vecAppend(std::vector<unsigned int> &A, std::vector<unsigned int> B);

// Seach in Vector for value
bool inVec(std::vector<unsigned int> v, unsigned int val);

///////////////////////////////////////////////////////////
// Misc Vector Math
///////////////////////////////////////////////////////////
double AvgSSRS(std::vector<double> A);

///////////////////////////////////////////////////////////////////////////////
/// Matrix Functions
///////////////////////////////////////////////////////////////////////////////
// The average sum of the square root squared
double AvgAbsSum(std::vector<std::vector<double>> A);
// Multiply Matrices
std::vector<double> multMat(
  std::vector<double> A, 
  unsigned int Ar, 
  unsigned int Ac, 
  std::vector<double> B,
  unsigned int Br,
  unsigned int Bc
);

// Element-wise multiplication of Matrices
void ewMM(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> B);
std::vector<std::vector<double>> ewMMR(
  std::vector<std::vector<double>> A,
  std::vector<std::vector<double>> B
);

// Element-wise division of Matrices
void ewDM(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> B);
std::vector<std::vector<double>> ewDMR(
  std::vector<std::vector<double>> A,
  std::vector<std::vector<double>> B
);
// Square Matrix
void sqMat(std::vector<std::vector<double>> &A);
std::vector<std::vector<double>> sqMatR(std::vector<std::vector<double>> A);

// Square Matrix
void sqrtMat(std::vector<std::vector<double>> &A);
std::vector<std::vector<double>> sqrtMatR(std::vector<std::vector<double>> A);

// Add Matrices
void addMat(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> B);
std::vector<std::vector<double>> addMatR(
  std::vector<std::vector<double>> A, 
  std::vector<std::vector<double>> B
);
// Subtract Matrices
void subMat(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> B);
std::vector<std::vector<double>> subMatR(
  std::vector<std::vector<double>> A, 
  std::vector<std::vector<double>> B
);

// Multiply Scaler to Matrix
void multScal(std::vector<std::vector<double>> &A, double scaler);
std::vector<std::vector<double>> multScalR(std::vector<std::vector<double>> A, double scaler);

// Add Scaler to Matrix
void addScal(std::vector<std::vector<double>> &A, double scaler);
std::vector<std::vector<double>> addScalR(std::vector<std::vector<double>> A, double scaler);

///////////////////////////////////////////////////////////////////////////////
/// Read Data
///////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<std::string>> readFile(std::string filename);

// Get data with ID at Position
void getDataID(
  std::string filename, 
  std::vector<std::vector<double>> &data,
  std::vector<std::vector<double>> &observations,
  unsigned int IDpos = 0,
  unsigned int skipRow = 1,
  unsigned int skipColumn = 0,
  unsigned int skipColPattern = 0
);

///////////////////////////////////////////////////////////////////////////////
/// Get Activation Matrix 
///////////////////////////////////////////////////////////////////////////////
// bool * for_all_nodes (size of number of layers) 
// actFun ** actfun
// for each layer
//  if for all nodes
//    apply activation function(layer, sizeoflayer)
//  else
//    num
//    for each node
//      apply activation function(&layer[ith], 1)

///////////////////////////////////////////////////////////////////////////////
/// Allocating Memory
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Input Readers
///////////////////////////////////////////////////////////////////////////////