#pragma once
#include "params.h"

///////////////////////////////////////////////////////////////////////////////
// Activation Functions
///////////////////////////////////////////////////////////////////////////////
#define AMBIT_TYPE unsigned int
/// ReLu
DTYPE relu(DTYPE x);
DTYPE drelu(DTYPE x);
std::vector<DTYPE> relu(std::vector<DTYPE> layer, AMBIT_TYPE ambit);
std::vector<DTYPE> drelu(std::vector<DTYPE> layer, AMBIT_TYPE ambit);
/// Sigmoid
DTYPE sigmoid(DTYPE x);
DTYPE dsigmoid(DTYPE x);
std::vector<DTYPE> sigmoid(std::vector<DTYPE> layer, AMBIT_TYPE ambit);
std::vector<DTYPE> dsigmoid(std::vector<DTYPE> layer, AMBIT_TYPE ambit);
/// Softmax
std::vector<DTYPE> softmax(std::vector<DTYPE> layer, AMBIT_TYPE obs);
std::vector<DTYPE> dsoftmax(std::vector<DTYPE> layer, unsigned int obs);

/// Argmax
std::vector<DTYPE> argmax(std::vector<DTYPE> layer, AMBIT_TYPE ambit);
std::vector<DTYPE> dargmax(std::vector<DTYPE> layer, AMBIT_TYPE ambit);

///////////////////////////////////////////////////////////////////////////////
// Loss Functions
///////////////////////////////////////////////////////////////////////////////
DTYPE crossentropy(DTYPE x);
DTYPE dcrossentropy(DTYPE x);
std::vector<DTYPE> crossentropy(
  std::vector<DTYPE> layer,
  unsigned int obs
);
std::vector<DTYPE> dcrossentropy(
  std::vector<DTYPE> layer,
  unsigned int obs
);

