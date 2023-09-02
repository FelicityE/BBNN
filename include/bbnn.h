#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

#include "data.h"

typedef DTYPE (*lossFunc)(DTYPE,DTYPE);
typedef DTYPE (*activationFunc)(DTYPE);


using namespace std;

///////////////////////////////////////////////////////////////////////////////
/// Activation Functions
///////////////////////////////////////////////////////////////////////////////
DTYPE sigmoid(DTYPE activation);
DTYPE sigmoidPrime(DTYPE activation);

///////////////////////////////////////////////////////////////////////////////
/// Error Functions
///////////////////////////////////////////////////////////////////////////////
DTYPE halfSquaredError(DTYPE x, DTYPE y);
DTYPE halfSquaredErrorPrime(DTYPE x, DTYPE y);

///////////////////////////////////////////////////////////////////////////////
/// Matrix Functions
///////////////////////////////////////////////////////////////////////////////
void matrixMultiply(
  DTYPE * A, 
  unsigned int aRow, 
  unsigned int aCol, 
  DTYPE * B, 
  unsigned int bRow, 
  unsigned int bCol, 
  DTYPE * C
);
void matrixAdd(
  DTYPE * A, 
  unsigned int aRow, 
  unsigned int aCol, 
  DTYPE * B, 
  unsigned int bRow, 
  unsigned int bCol, 
  DTYPE * C
);
void matrixSubtract(
  DTYPE * A, 
  unsigned int aRow, 
  unsigned int aCol, 
  DTYPE * B, 
  unsigned int bRow, 
  unsigned int bCol, 
  DTYPE * C
);

///////////////////////////////////////////////////////////////////////////////
/// ANN Functions
///////////////////////////////////////////////////////////////////////////////
void updateLayer(
  unsigned int inCount, 
  unsigned int outCount, 
  DTYPE * Lin, 
  DTYPE * Lout, 
  DTYPE * W, 
  DTYPE * B, 
  activationFunc ActivationFunction
);
void runForward(
  DTYPE ** layers, 
  unsigned int numLayers, 
  unsigned int * layerSizes,
  DTYPE ** weights, 
  DTYPE ** bias, 
  activationFunc * ActivationFunction
);
void backProbGradDecent(
  DTYPE ** layers, 
  unsigned int numLayers, 
  unsigned int * layerSizes,
  DTYPE * expectedOutcome,
  DTYPE ** weights, 
  DTYPE ** bias, 
  lossFunc LossFuncPrime, 
  activationFunc * ActivationFunctionPrime, 
  DTYPE learningRate
);