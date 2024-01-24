#pragma once

#include <math.h> // For sqrt() pow()
#include <vector> // For vectors
#include "params.h"

///////////////////////////////////////////////////////////////////////////////
// Activation Functions
///////////////////////////////////////////////////////////////////////////////
// DTYPE elu(DTYPE x);
// DTYPE delu(DTYPE x);

// std::vector<DTYPE> elu(std::vector<DTYPE> layer, int stop, int meta = 0);
// std::vector<DTYPE> delu(std::vector<DTYPE> layer, int stop, int meta = 0);

// DTYPE sigmoid(DTYPE x);
// DTYPE dsigmoid(DTYPE x);

// std::vector<DTYPE> sigmoid(std::vector<DTYPE> layer, int stop, int meta = 0);
// std::vector<DTYPE> dsigmoid(std::vector<DTYPE> layer, int stop, int meta = 0);

DTYPE relu(DTYPE x);
DTYPE drelu(DTYPE x);

std::vector<DTYPE> relu(std::vector<DTYPE> layer, int stop, int meta = 0);
std::vector<DTYPE> drelu(std::vector<DTYPE> layer, int stop, int meta = 0);


///////////////////////////////////////////////////////////////////////////////
// Layer Functions
///////////////////////////////////////////////////////////////////////////////
std::vector<DTYPE> softmax(std::vector<DTYPE> layer, int stop, int meta = 0);

std::vector<DTYPE> dsoftmax(std::vector<DTYPE> layer, int stop, int obs = 0);

std::vector<DTYPE> argmax(std::vector<DTYPE> layer, int stop, int meta = 0);