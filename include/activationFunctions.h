#pragma once

#include <math.h> // For sqrt() pow()
#include <vector> // For vectors
#include "params.h"

///////////////////////////////////////////////////////////////////////////////
// Activation Functions
///////////////////////////////////////////////////////////////////////////////
double ReLu(double x);
double dReLu(double x);


std::vector<double> ReLu(std::vector<double> layer, int stop, int meta = 0);
std::vector<double> dReLu(std::vector<double> layer, int stop, int meta = 0);

///////////////////////////////////////////////////////////////////////////////
// Layer Functions
///////////////////////////////////////////////////////////////////////////////
std::vector<double> softMax(std::vector<double> layer, int stop, int meta = 0);

std::vector<double> dSoftMax(std::vector<double> layer, int stop, int obs = 0);

std::vector<double> argMax(std::vector<double> layer, int stop, int meta = 0);