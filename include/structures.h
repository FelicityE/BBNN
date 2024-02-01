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
  Ann(){}
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
  Data(){}
  // Data(
  //   std::vector<DTYPE> features, 
  //   unsigned int nFeatures,
  //   std::vector<unsigned int> observations
  // ){
  //   if(features.size()/nFeatures != observations.size()){
  //     std::cout << 
  //     "ERROR - struct Data: Number of observations does not match number of sample sets."
  //     << features.size()/nFeatures << ":" << observations.size() << std::endl;
  //   }
  //   this->features = features;
  //   this->nFeatures = nFeatures;
  //   this->observations = observations;
  //   this->nClasses = max(observations);
  //   this->nSamples = observations.size();
  // }
  std::vector<DTYPE> features;
  unsigned int nFeatures; // numbder of features/ size of first layer
  std::vector<unsigned int> observations;
  unsigned int nClasses; // number of classes / size of last layer
  unsigned int nSamples;
};

// struct Meta{
//   Meta(): maxIter(1000), ratio(70), wseed(42){}
//   Meta(
//     unsigned int maxIter, 
//     double ratio,
//     unsigned int wseed
//   ){
//     this->maxIter = maxIter;
//     this->ratio = ratio;
//     this->wseed = wseed;
//   }
//   unsigned int maxIter;
//   double ratio;
//   unsigned int wseed;
// };

struct ANN_Ambit{
  ANN_Ambit():nLayers(3), hNodes(std::vector<unsigned int>(1,2)), maxIter(1000), wseed(42){}
  unsigned int nLayers;
  std::vector<unsigned int> hNodes;
  std::vector<std::vector<unsigned int>> setActID_inputs;
  unsigned int maxIter;
  unsigned int wseed; // weights seed
};

struct Read_Ambit{
  Read_Ambit(): idp(0), skipRow(1), skipCol(1), sseed(0), ratio(70){};
  Read_Ambit(std::string filepath): idp(0), skipRow(1), skipCol(1), sseed(0), ratio(70){
    this->dataFilePath = filepath;
  }
  // MetaRead(
  //   std::string filepath,
  //   unsigned int idp,
  //   unsigned int skipRow,
  //   unsigned int skipCol,
  //   unsigned int sseed,
  //   double ratio
  // ){
  //   this->dataFilePath = filepath;
  //   this->idp = idp;
  //   this->skipRow = skipRow;
  //   this->skipCol = skipCol;
  //   this->sseed = sseed;
  //   this->ratio = ratio;
  // }
  std::string dataFilePath; // Filepath to data
  unsigned int idp; // Class ID column number
  unsigned int skipRow; // Number of rows to skip
  unsigned int skipCol; // Number of columns to skip
  unsigned int sseed; // sample seed for selection
  double ratio;
};

// struct SetUp{
//   // Constructors
//   SetUp(){};
//   // Set Defaults
//   SetUp(std::string filepath):
//     idp(0), skipRow(1), skipCol(1),
//     maxIter(1000), ratio(70), sseed(0), wseed(42),
//     adam(false), alpha(0.01), beta1(0.9), beta2(0.999),
//     nLayers(3)
//   {this->hNodes = std::vector<unsigned int> (1,2);}
 
//   // Data infomation
//   std::string dataFilePath;
//   unsigned int idp; // Class ID column postion
//   unsigned int skipRow; // Number of rows to skip
//   unsigned int skipCol;
//   double ratio;
//   unsigned int sseed; // sample seed

//   // Adam information
//   bool adam;
//   double alpha;
//   double beta1;
//   double beta2;
  
//   // Ann Information
//   unsigned int nLayers;
//   std::vector<unsigned int> hNodes;
//   std::vector<std::vector<unsigned int>> setActID_inputs;
//   unsigned int maxIter;
//   unsigned int wseed; // weights seed
// };