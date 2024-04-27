#pragma once
#include "params.h"

///////////////////////////////////////////////////////////////////////////////
// Activation Functions
///////////////////////////////////////////////////////////////////////////////
#define OBS_TYPE unsigned int
///////////////////////////////////////////////////////////////////////////////
/// ReLu
DTYPE relu(DTYPE x);
DTYPE drelu(DTYPE x);
std::vector<DTYPE> relu(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> drelu(std::vector<DTYPE> layer, OBS_TYPE obs);

/// eLu
DTYPE elu(DTYPE x);
DTYPE delu(DTYPE x);
std::vector<DTYPE> elu(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> delu(std::vector<DTYPE> layer, OBS_TYPE obs);

// LeakyReLU
DTYPE leakyrelu(DTYPE x);
DTYPE dleakyrelu(DTYPE x);
std::vector<DTYPE> leakyrelu(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> dleakyrelu(std::vector<DTYPE> layer, OBS_TYPE obs);

/// GeLu
DTYPE gelu(DTYPE x);
DTYPE dgelu(DTYPE x);
std::vector<DTYPE> gelu(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> dgelu(std::vector<DTYPE> layer, OBS_TYPE obs);

/// Swish
DTYPE swish(DTYPE x);
DTYPE dswish(DTYPE x);
std::vector<DTYPE> swish(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> dswish(std::vector<DTYPE> layer, OBS_TYPE obs);

///////////////////////////////////////////////////////////////////////////////
/// Sigmoid
DTYPE sigmoid(DTYPE x);
DTYPE dsigmoid(DTYPE x);
std::vector<DTYPE> sigmoid(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> dsigmoid(std::vector<DTYPE> layer, OBS_TYPE obs);

/// Bipolar Sigmoid
DTYPE bisigmoid(DTYPE x);
DTYPE dbisigmoid(DTYPE x);
std::vector<DTYPE> bisigmoid(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> dbisigmoid(std::vector<DTYPE> layer, OBS_TYPE obs);

/// Tanh
DTYPE tanh_(DTYPE x);
DTYPE dtanh(DTYPE x);
std::vector<DTYPE> tanh_(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> dtanh(std::vector<DTYPE> layer, OBS_TYPE obs);

///////////////////////////////////////////////////////////////////////////////
/// Gaussian
DTYPE gaussian(DTYPE x);
DTYPE dgaussian(DTYPE x);
std::vector<DTYPE> gaussian(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> dgaussian(std::vector<DTYPE> layer, OBS_TYPE obs);

///////////////////////////////////////////////////////////////////////////////
/// Softmax
std::vector<DTYPE> softmax(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> dsoftmax(std::vector<DTYPE> layer, OBS_TYPE obs);

/// Argmax
std::vector<DTYPE> argmax(std::vector<DTYPE> layer, OBS_TYPE obs);
std::vector<DTYPE> dargmax(std::vector<DTYPE> layer, OBS_TYPE obs);

///////////////////////////////////////////////////////////////////////////////
// Loss Functions
///////////////////////////////////////////////////////////////////////////////
DTYPE crossentropy(DTYPE x);
DTYPE dcrossentropy(DTYPE x);
std::vector<DTYPE> crossentropy(
  std::vector<DTYPE> layer,
  OBS_TYPE obs
);
std::vector<DTYPE> dcrossentropy(
  std::vector<DTYPE> layer,
  OBS_TYPE obs
);

