#pragma once

#include "utility.h"

///////////////////////////////////////////////////////////////////////////////
/// Special Utility
///////////////////////////////////////////////////////////////////////////////
void print(Ann ann);

///////////////////////////////////////////////////////////////////////////////
/// Initializers
///////////////////////////////////////////////////////////////////////////////
void initWeights(
  std::vector<std::vector<DTYPE>> &weights, 
  std::vector<unsigned int> nNodes
);
void initWeights(Ann &ann);
Ann initANN(
  unsigned int nFeatures, 
  unsigned int nClasses, 
  unsigned int nLayers
);
Ann initANN(
	unsigned int nFeatures, 
	unsigned int nClasses, 
	unsigned int nLayers,
	std::vector<unsigned int> nNodes
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
  Ann &ann,
  unsigned int ID,
  unsigned int layerNStrt = 0,
  unsigned int layerNEnd = UINT_MAX,
  unsigned int nodeNStrt = 0,
  unsigned int nodeNEnd = UINT_MAX
);
