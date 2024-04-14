#pragma once
#include "activations.h"

///////////////////////////////////////////////////////////////////////////////
// Lists 
///////////////////////////////////////////////////////////////////////////////
const std::vector<actT1> ACT1{relu, sigmoid};
const std::vector<actT1> DACT1{drelu, dsigmoid};
const std::vector<actT2> ACT2{relu, sigmoid, softmax, argmax};
const std::vector<actT2> DACT2{drelu, dsigmoid, dsoftmax, dargmax};
const int TYPECHANGE = ACT1.size();

const std::vector<lossF> LOSSF{crossentropy};
const std::vector<lossF> DLOSSF{dcrossentropy};

enum ACTID{RELU, SIGMOID, SOFTMAX, ARGMAX};


///////////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////////
struct ActID_Set{
  ActID_Set(unsigned int id, std::vector<unsigned int> nodePos){
    this->ID = id;
    this->nodePositions = nodePos;
  }
  unsigned int ID;
  std::vector<unsigned int> nodePositions;
};

struct Alpha{
  Alpha(): 
    alpha(0.01), 
    gamma(0.0001), 
    adam(false), 
    beta1(0.9), 
    beta2(0.999),
    epsilon(10e-8)
  {}
  double alpha;
  double gamma;
  bool adam;
  double beta1;
  double beta2;
  double epsilon;
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
  ANN_Ambit():nLayers(3), hNodes(std::vector<unsigned int>(1,2)), maxIter(1000), wseed(42){
    this->logpath = "../results/log.csv";
  }
  unsigned int nLayers;
  std::vector<unsigned int> hNodes;
  std::vector<struct ActID_Set> ActIDSets;
  unsigned int maxIter;
  unsigned int wseed; // weights seed
  std::string logpath; // Filepath to output
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
    this->analyze = false;
  };
  Read_Ambit(std::string filepath): idp(0), skipRow(1), skipCol(0){
    this->filepath = filepath;
    this->sseed = std::vector<unsigned int> (1,0);
    this->ratio = std::vector<double> (1,70);
    this->analyze = false;
  }
  std::string filepath; // Filepath to data
  bool analyze;
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
    unsigned int nClasses
  ):uint_ambit(0), double_ambit(0){
    this->observedValue = std::vector<unsigned int> (nSamples, nClasses);
    this->vector_bool = std::vector<bool>(nSamples, false);
    this->vector_uint = std::vector<unsigned int>(nSamples, nClasses);
    this->vector_dtype = std::vector<DTYPE>(nSamples*nClasses, 0);
  }
  Results(
    unsigned int nSamples,
    unsigned int nClasses,
    std::vector<unsigned int> obs 
  ):uint_ambit(0), double_ambit(0){
    this->observedValue = obs;
    this->vector_bool = std::vector<bool>(nSamples, false);
    this->vector_uint = std::vector<unsigned int>(nSamples, nClasses);
    this->vector_dtype = std::vector<DTYPE>(nSamples*nClasses, 0);
  }
  
  unsigned int uint_ambit; // number of correct predictions
  DTYPE  double_ambit; // error
  std::vector<unsigned int> observedValue;
  std::vector<bool> vector_bool; // bool was the sample prediction correct (or within x error for nonclass)
  std::vector<unsigned int> vector_uint; // sample prediction
  std::vector<DTYPE> vector_dtype; // sample prediction
};

struct Scores{
  Scores(){}
  Scores(unsigned int nClass){
    this->accuracy = 0;
    this->precision = std::vector<double>(nClass,0);
    this->recall = std::vector<double>(nClass,0);
    this->F1 = std::vector<double>(nClass,0);
  }
  double accuracy;
  std::vector<double> precision;
  std::vector<double> recall;
  std::vector<double> F1;
};