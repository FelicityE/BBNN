#pragma once
#include "params.h"

///////////////////////////////////////////////////////////////////////////////
// Activation Functions
///////////////////////////////////////////////////////////////////////////////
/// ReLu
DTYPE relu(DTYPE x);
DTYPE drelu(DTYPE x);
std::vector<DTYPE> relu(std::vector<DTYPE> layer, std::vector<DTYPE> meta);
std::vector<DTYPE> drelu(std::vector<DTYPE> layer, std::vector<DTYPE> meta);
/// Sigmoid
DTYPE sigmoid(DTYPE x);
DTYPE dsigmoid(DTYPE x);
std::vector<DTYPE> sigmoid(std::vector<DTYPE> layer, std::vector<DTYPE> meta);
std::vector<DTYPE> dsigmoid(std::vector<DTYPE> layer, std::vector<DTYPE> meta);
/// Softmax
std::vector<DTYPE> softmax(std::vector<DTYPE> layer, std::vector<DTYPE> meta);
std::vector<DTYPE> dsoftmax(std::vector<DTYPE> layer, std::vector<DTYPE> meta);

/// Argmax
std::vector<DTYPE> argmax(std::vector<DTYPE> layer, std::vector<DTYPE> ambit);
std::vector<DTYPE> dargmax(std::vector<DTYPE> layer, std::vector<DTYPE> ambit);

///////////////////////////////////////////////////////////////////////////////
// Loss Functions
///////////////////////////////////////////////////////////////////////////////
DTYPE crossentropy(DTYPE x);
DTYPE dcrossentropy(DTYPE x);
std::vector<DTYPE> crossentropy(
  std::vector<DTYPE> layer,
  int obs
);
std::vector<DTYPE> dcrossentropy(
  std::vector<DTYPE> layer,
  int obs
);

