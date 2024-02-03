#pragma once

#include <math.h> // For sqrt() pow()
#include <vector> // For vectors
#include "params.h"

///////////////////////////////////////////////////////////////////////////////
// Activation Functions
///////////////////////////////////////////////////////////////////////////////
// double elu(double x);
// double delu(double x);

// std::vector<double> elu(std::vector<double> layer, int stop, int meta = 0);
// std::vector<double> delu(std::vector<double> layer, int stop, int meta = 0);

// double sigmoid(double x);
// double dsigmoid(double x);

// std::vector<double> sigmoid(std::vector<double> layer, int stop, int meta = 0);
// std::vector<double> dsigmoid(std::vector<double> layer, int stop, int meta = 0);

double relu(double x);
double drelu(double x);

std::vector<double> relu(std::vector<double> layer, int stop, int meta = 0);
std::vector<double> drelu(std::vector<double> layer, int stop, int meta = 0);


///////////////////////////////////////////////////////////////////////////////
// Layer Functions
///////////////////////////////////////////////////////////////////////////////
std::vector<double> softmax(std::vector<double> layer, int stop, int meta = 0);

std::vector<double> dsoftmax(std::vector<double> layer, int stop, int obs = 0);

std::vector<double> argmax(std::vector<double> layer, int stop, int meta = 0);