#pragma once
#include "activations.h"

///////////////////////////////////////////////////////////////////////////////
// Lists 
///////////////////////////////////////////////////////////////////////////////
static const std::vector<actF> ACTF{sigmoid, relu, softmax, argmax};
static const std::vector<actF> DACTF{dsigmoid, drelu, dsoftmax, dargmax};
static const int typeChange = 2;

static const std::vector<lossF> LOSSF{crossentropy};
static const std::vector<lossF> DLOSSF{dcrossentropy};


///////////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////////
struct Adam{
  Adam(): adam(false), alpha(0.01), beta1(0.9), beta2(0.999){}
  bool adam;
  double alpha;
  double beta1;
  double beta2;
};

struct Ann{
  unsigned int nFeatures;
  unsigned int nClasses;
  unsigned int nLayers;
  std::vector<unsigned int> nNodes;
  unsigned int tNodes;
  std::vector<unsigned int> actIDs;
  unsigned int lossID;
  std::vector<std::vector<DTYPE>> weights;
  std::vector<DTYPE> bias;
};

struct Data{
  std::vector<DTYPE> features;
  unsigned int nFeatures;
  std::vector<unsigned int> observations;
};

struct Meta{
  Meta(): maxIter(1000), ratio(70), sseed(0), wseed(42){}
  unsigned int maxIter;
  double ratio;
  unsigned int sseed;
  unsigned int wseed;
};

struct MetaRead{
  MetaRead(std::string filepath): idp(0), skipRow(1), skipCol(1){
    this->dataFilePath = filepath;
  }
  std::string dataFilePath; // Filepath to data
  unsigned int idp; // Class ID column number
  unsigned int skipRow; // Number of rows to skip
  unsigned int skipCol; // Number of columns to skip
};

