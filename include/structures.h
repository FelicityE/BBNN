#pragma once
#include "activations.h"

///////////////////////////////////////////////////////////////////////////////
// Lists 
///////////////////////////////////////////////////////////////////////////////
const std::vector<actT1> ACT1{sigmoid, relu};
const std::vector<actT1> DACT1{dsigmoid, drelu};
const std::vector<actT2> ACT2{sigmoid, relu, softmax, argmax};
const std::vector<actT2> DACT2{dsigmoid, drelu, dsoftmax, dargmax};
const int TYPECHANGE = ACT1.size();

const std::vector<lossF> LOSSF{crossentropy};
const std::vector<lossF> DLOSSF{dcrossentropy};

enum ACTID{SIGMOID, RELU, SOFTMAX, ARGMAX};


///////////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////////
struct ActID_Set{
  ActID_Set():
    ID(RELU),
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
  Ann():lLAID(ARGMAX){}
  void setLLFT(unsigned int ID){this->lLAID = ARGMAX; return;}

  unsigned int nLayers; // Number of layers

  std::vector<unsigned int> nNodes; // Number of Nodes
  std::vector<unsigned int> sNodes; // Starting Node number
  unsigned int tNodes; // Total Number of Nodes
  
  std::vector<unsigned int> actIDs; // Activation Function IDs
  unsigned int lossID; // Loss Function ID
  unsigned int lLAID; // Set the ID of the last layer for testing to argmax(3)

  std::vector<std::vector<DTYPE>> weights; // The Weight parameters
  std::vector<DTYPE> bias; // This bias parameters
  
};

struct ANN_Ambit{
  ANN_Ambit():nLayers(3), hNodes(std::vector<unsigned int>(1,2)), maxIter(1000), wseed(42){}
  unsigned int nLayers;
  std::vector<unsigned int> hNodes;
  std::vector<struct ActID_Set> ActIDSets;
  unsigned int maxIter;
  unsigned int wseed; // weights seed
};

struct Data{
  Data():sseed(0), ratio(70), acc_err(0.01){}
  unsigned int nFeat; // numbder of feat/ size of first layer
  unsigned int nClasses; // number of classes / size of last layer
  unsigned int nSamp; // number samples
  unsigned int sseed;
  double ratio;
  double acc_err; // acceptable error for regression (replace with algorithm?)
  std::vector<DTYPE> feat;
  std::vector<unsigned int> obs;
};

struct Read_Ambit{
  Read_Ambit(): idp(0), skipRow(1), skipCol(0){
    this->sseed = std::vector<unsigned int> (1,0);
    this->ratio = std::vector<double> (1,70);
  };
  Read_Ambit(std::string filepath): idp(0), skipRow(1), skipCol(0){
    this->filepath = filepath;
    this->sseed = std::vector<unsigned int> (1,0);
    this->ratio = std::vector<double> (1,70);
  }
  std::string filepath; // Filepath to data
  unsigned int idp; // Class ID column number
  unsigned int skipRow; // Number of rows to skip
  unsigned int skipCol; // Number of columns to skip
  std::vector<unsigned int> sseed; // sample seed for selection
  std::vector<double> ratio; // percent training to testing (default 70:30)
};

struct Results{
  Results():uint_ambit(0), double_ambit(0){}
  Results(
    unsigned int nSamples,
    unsigned int nClasses,
    unsigned int nFeatures
  ):uint_ambit(0), double_ambit(0){
    this->vector_bool = std::vector<bool>(nSamples, false);
    this->vector_unit = std::vector<unsigned int>(nSamples, nClasses);
    this->vector_dtype = std::vector<DTYPE>(nSamples*nFeatures, 0);
  }
  
  unsigned int uint_ambit; // number of correct predictions
  DTYPE  double_ambit; // error
  std::vector<bool> vector_bool; // bool was the sample prediction correct (or within x error for nonclass)
  std::vector<unsigned int> vector_unit; // sample prediction
  std::vector<DTYPE> vector_dtype; // sample prediction
};