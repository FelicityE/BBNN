#pragma once

#include "vector_math.h"

///////////////////////////////////////////////////////////////////////////////
/// ANN Utility
///////////////////////////////////////////////////////////////////////////////
void print(struct Ann ann);
void print(std::vector<std::vector<DTYPE>> W, std::vector<DTYPE> B);
void applyAct(
  std::vector<DTYPE> &layer, 
  std::vector<unsigned int> aIDs,
  unsigned int obs
);
void applyDAct(
  std::vector<DTYPE> &layer, 
  std::vector<unsigned int> aIDs,
  unsigned int obs
);
std::vector<DTYPE> applyDActR(
  std::vector<DTYPE> &layer, 
  std::vector<unsigned int> aIDs,
  unsigned int obs
);
///////////////////////////////////////////////////////////////////////////////
/// Initializers
///////////////////////////////////////////////////////////////////////////////
void initWeights(
  std::vector<std::vector<DTYPE>> &weights, 
  std::vector<unsigned int> nNodes
);
void initWeights(struct Ann &ann);
Ann initANN(
  unsigned int nFeat, 
  unsigned int nClasses, 
  unsigned int nLayers
);
Ann initANN(
	unsigned int nFeat, 
	unsigned int nClasses, 
	unsigned int nLayers,
	std::vector<unsigned int> nNodes
);
Ann initANN(struct ANN_Ambit ann_, struct Data train);

void getDataSets(
  struct Data &train, 
  struct Data &test, 
  struct Data data
);


///////////////////////////////////////////////////////////////////////////////
/// Setters
///////////////////////////////////////////////////////////////////////////////
/// ANN 
// Get Activation ID Position
unsigned int getAIDP(
  std::vector<unsigned int> nNodes, 
  unsigned int layerN, 
  unsigned int nodeN = 0
);

void setActID(
  std::vector<unsigned int> &actIDs,
  std::vector<unsigned int> nNodes,
  unsigned int ID,
  unsigned int layerN,
  unsigned int nodeN
);

// setActID(..., start<inclusive>, end<exclusive>, ...)
void setActID(
  struct Ann &ann,
  unsigned int ID,
  unsigned int layerNStrt = 0,
  unsigned int layerNEnd = UINT_MAX,
  unsigned int nodeNStrt = 0,
  unsigned int nodeNEnd = UINT_MAX
);

///////////////////////////////////////////////////////////////////////////////
/// Training and Testing
///////////////////////////////////////////////////////////////////////////////
void forward(
  struct Ann ann,
  std::vector<DTYPE> &layer,
  std::vector<DTYPE> &act
);

void backProp(
  struct Ann ann,
  unsigned int obs,
  std::vector<DTYPE> layer,
  std::vector<DTYPE> act,
  std::vector<std::vector<DTYPE>> &dW,
  std::vector<DTYPE> &dB
);

void trainNN(
  struct Ann &ann, 
  struct Data train,
  struct Results &result,
  struct Alpha alpha, 
  unsigned int maxIter = 1000
);

void testNN(
  struct Ann ann,
  struct Data test,
  struct Results &result
);