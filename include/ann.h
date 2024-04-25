#pragma once

#include "vector_math.h"

///////////////////////////////////////////////////////////////////////////////
/// ANN Utility
///////////////////////////////////////////////////////////////////////////////
void print(struct Ann ann);
void print(std::vector<std::vector<DTYPE>> W, std::vector<DTYPE> B);

void printTo(
  struct Scores scores,
  std::string filename = "scores.log"
);



///////////////////////////////////////////////////////////////////////////////
/// Applying Activation Functions
///////////////////////////////////////////////////////////////////////////////
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

Ann initANN(struct ANN_Ambit &annbit, struct Data train);

///////////////////////////////////////////////////////////////////////////////
/// Setters
///////////////////////////////////////////////////////////////////////////////
void setActList(std::vector<unsigned int> &list, unsigned int value);

void setHLayers(
  struct ANN_Ambit &annbit,
  unsigned int hLayers,
  std::vector<unsigned int> nNodes
);

void set_actNodes(
  std::vector<unsigned int> &actIDs,
  unsigned int id,
  std::vector<unsigned int> nodes
);
void set_actDivide(
  std::vector<unsigned int> &actIDs,
  std::vector<unsigned int> id_list,
  unsigned int divider,
  std::vector<unsigned int> nNodes
);
void setActID(
  std::vector<unsigned int> &actIDs,
  std::vector<struct ActID_Set> sets,
  std::vector<unsigned int> nNodes,
  std::vector<unsigned int> sNodes
);

///////////////////////////////////////////////////////////////////////////////
/// Getters
///////////////////////////////////////////////////////////////////////////////
void getDataSets(
  struct Data &train, 
  struct Data &test, 
  struct Data data
);

unsigned int getNodePosition(
  std::vector<unsigned int> nNodes,
  unsigned int Layer,
  unsigned int node
);

// std::vector<unsigned int> getNodeActivations(
void getNodeActivations(
  std::vector<struct ActID_Set> &sets,
  std::vector<unsigned int> actList,
  std::vector<unsigned int> nNodes,
  unsigned int aseed
);

void getResults(
  struct Results &result,
  std::vector<DTYPE> lastAct,
  std::vector<DTYPE> lastLayer,
  unsigned int lastActID,
  unsigned int lossID,
  unsigned int sampleIndex,
  unsigned int obs
);
void getScores(
  struct Scores &score,
  std::vector<unsigned int> obs,
  std::vector<unsigned int> pre,
  std::vector<bool> cor
);
struct Scores getScores(
  struct Results result,
  unsigned int nClasses
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

unsigned int trainNN(
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

///////////////////////////////////////////////////////////////////////////////
/// Runners
///////////////////////////////////////////////////////////////////////////////
void runANN(
  struct Alpha alpha,
  struct ANN_Ambit &annbit,
  struct Data data,
  double stamp
);

void runAnalysis(
  struct Read_Ambit &readbit,
  struct ANN_Ambit &annbit,
  struct Alpha &alpha,
  struct Data data,
  bool addheader
);