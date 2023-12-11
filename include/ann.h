#pragma once

#include "activationFunctions.h"
#include "errorFunctions.h"
#include "params.h"
#include "utility.h"

///////////////////////////////////////////////////////////////////////////////
/// Init Functions
///////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<double>> initWeights(
  unsigned int nLayers, 
  std::vector<unsigned int> nNodes, 
  bool setRand = false
);
std::vector<std::vector<double>> initBias(
  unsigned int nLayers, 
  std::vector<unsigned int> nNodes, 
  bool setRand = false
);
// Init vector to Zero
std::vector<std::vector<double>> initZero(
  unsigned int outter, 
  std::vector<unsigned int> inner
);
std::vector<std::vector<double>> initZero(std::vector<std::vector<double>> copy);

// Get Class Number
unsigned int classNumber(std::vector<double> lastlayer);
std::vector<unsigned int> classNumber(std::vector<std::vector<double>> predicL);

// multiply Vector (dB*A), used for backprop
std::vector<double> get_dW(std::vector<double> dB, std::vector<double> A);
std::vector<double> get_inner_dB(
  std::vector<double> delta, 
  std::vector<double> W, 
  std::vector<double> fprimeL
);

///////////////////////////////////////////////////////////////////////////////
/// ANN Functions
///////////////////////////////////////////////////////////////////////////////
// Forward
// std::vector<double> runForward(
//   std::vector<double> features, 
//   unsigned int nLayers, 
//   std::vector<unsigned int> nNodes, 
//   std::vector<std::vector<double>> weights,
//   std::vector<std::vector<double>> bias
// );
void runForward(
  // std::vector<double> features, 
  // unsigned int nLayers, 
  std::vector<unsigned int> nNodes, 
  std::vector<std::vector<double>> weights,
  std::vector<std::vector<double>> bias,
  std::vector<std::vector<double>> &layers, 
  std::vector<std::vector<double>> &Acts,
  activationFunction ** actFun,
  int * actType
);

// Backprop
void runBackprop(
  // unsigned int nLayers,
  // std::vector<unsigned int> nNodes,
  std::vector<std::vector<double>> weights,
  std::vector<std::vector<double>> bias, 
  std::vector<std::vector<double>> layers, 
  std::vector<std::vector<double>> Acts,
  std::vector<double> obs,
  std::vector<std::vector<double>> &dWeights,
  std::vector<std::vector<double>> &dBias,
  lossFunction dLossFun,
  activationFunction ** dActFun,
  int * actType
);


///////////////////////////////////////////////////////////////////////////////
/// Training Functions
///////////////////////////////////////////////////////////////////////////////
void getTrainingSet(
  std::vector<std::vector<double>> samples, 
  std::vector<std::vector<double>> &train, 
  std::vector<std::vector<double>> &test,
  double ratio = 70 // Ratio of training to test set, default 70% training - 30% test
);
void getTrainingSet(
  std::vector<std::vector<double>> samples,
  std::vector<std::vector<double>> obs, 
  std::vector<std::vector<double>> &train, 
  std::vector<std::vector<double>> &obsTrain,
  std::vector<std::vector<double>> &test,
  std::vector<std::vector<double>> &obsTest, 
  double ratio = 70 // Ratio of training to test set, default 70% training - 30% test
);

double trainSNN(
  std::vector<std::vector<double>> sample,
  std::vector<std::vector<double>> obs,
  std::vector<unsigned int> nNodes,
  std::vector<std::vector<double>> &weights,
  std::vector<std::vector<double>> &bias,
  activationFunction ** actFun,
  activationFunction ** dActFun,
  int * actType,
  lossFunction lossFun,
  unsigned int stop = 1000,
  bool Adam = true,
  std::vector<double> abbe = {0.01,0.9,0.999,10e-8},
  double gamma = 0.00001 // convergance threshold
);

// Testing Functions

void testSNN(
  std::vector<std::vector<double>> samples,
  std::vector<std::vector<double>> obs,
  std::vector<unsigned int> nNodes,
  std::vector<std::vector<double>> weights,
  std::vector<std::vector<double>> bias,
  activationFunction ** actFun,
  int * actType,
  std::vector<std::vector<double>> &outVal,
  std::vector<bool> &results
);