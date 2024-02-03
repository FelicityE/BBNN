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

enum ACTID{SIGMOID, RELU, SOFTMAX, ARGMAX};


///////////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////////
struct ActID_Set{
  ActID_Set():
    ID(1),
    layerStrt(0),
    layerEnd(UINT_MAX),
    nodeStrt(0),
    nodeEnd(UINT_MAX)
  {}
  ActID_Set(std::vector<unsigned int> set):
    layerEnd(UINT_MAX),
    nodeStrt(0),
    nodeEnd(UINT_MAX)
  {
    this->ID = set[0];
    this->layerStrt = set[1];
    if(set.size() > 2){this->layerEnd = set[2];}
    if(set.size() > 3){this->nodeStrt = set[3];}
    if(set.size() > 4){this->nodeEnd = set[4];}
  }
  unsigned int ID;
  unsigned int layerStrt;
  unsigned int layerEnd;
  unsigned int nodeStrt;
  unsigned int nodeEnd;
};

struct Adam{
  Adam(): adam(false), alpha(0.01), beta1(0.9), beta2(0.999){}
  bool adam;
  double alpha;
  double beta1;
  double beta2;
};

struct Ann{
  Ann():argmax(true){}
  unsigned int nFeatures;
  unsigned int nClasses;
  unsigned int nLayers;
  std::vector<unsigned int> nNodes;
  unsigned int tNodes;
  std::vector<unsigned int> actIDs;
  unsigned int lossID;
  std::vector<std::vector<DTYPE>> weights;
  std::vector<DTYPE> bias;
  bool argmax; // Set the last layer for testing to argmax

  void setArgmax(bool set){this->argmax = set; return;}
};

struct ANN_Ambit{
  ANN_Ambit():nLayers(3), hNodes(std::vector<unsigned int>(1,2)), maxIter(1000), wseed(42){}
  unsigned int nLayers;
  std::vector<unsigned int> hNodes;
  std::vector<ActID_Set> ActIDSets;
  unsigned int maxIter;
  unsigned int wseed; // weights seed
};

struct Data{
  Data(){}
  std::vector<DTYPE> features;
  unsigned int nFeatures; // numbder of features/ size of first layer
  std::vector<unsigned int> observations;
  unsigned int nClasses; // number of classes / size of last layer
  unsigned int nSamples;
};

struct Read_Ambit{
  Read_Ambit(): idp(0), skipRow(1), skipCol(1), sseed(0), ratio(70){};
  Read_Ambit(std::string filepath): idp(0), skipRow(1), skipCol(1), sseed(0), ratio(70){
    this->dataFilePath = filepath;
  }
  std::string dataFilePath; // Filepath to data
  unsigned int idp; // Class ID column number
  unsigned int skipRow; // Number of rows to skip
  unsigned int skipCol; // Number of columns to skip
  unsigned int sseed; // sample seed for selection
  double ratio;
};
