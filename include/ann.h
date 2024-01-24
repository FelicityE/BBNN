#pragma once

#include "params.h"
#include "utility.h"

///////////////////////////////////////////////////////////////////////////////
/// Init Functions
///////////////////////////////////////////////////////////////////////////////
struct Ann{
  Ann(): nFeatures(2), nClasses(2), nLayers(2), nNodes(nLayers, nClasses){
    nNodes[0] = nFeatures;
    tNodes = sum(nNodes);
    actsList = std::vector<unsigned int> (tNodes, 1);
  }
  unsigned int nFeatures;
  unsigned int nClasses;
  unsigned int nLayers;
  std::vector<unsigned int> nNodes;
  unsigned int tNodes;
  std::vector<unsigned int> actsList;
  std::vector<std::vector<DTYPE>> weights;
  std::vector<DTYPE> bias;
};
void setnLayers(Ann &ann, unsigned int value);
void setnNodes(
  std::vector<unsigned int> &nNodes, 
  unsigned int value, 
  unsigned int layer
);
void setnNodes(
  std::vector<unsigned int> &nNodes, 
  unsigned int value, 
  unsigned int layerStart,
  unsigned int layerEnd
);

void initWeights(Ann &ann);
void initBias(Ann &ann);


struct Adam{
  Adam(): adam(false), alpha(0.01), beta1(0.9), beta2(0.999){}
  bool adam;
  double alpha;
  double beta1;
  double beta2;
};

void setAdam(bool &ambit, bool value);
void setAdam(double &ambit, double value);
