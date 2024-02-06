#include "include/ann.h"

///////////////////////////////////////////////////////////////////////////////
/// Special Utility
///////////////////////////////////////////////////////////////////////////////
void print(struct Ann ann){
  std::cout << "Number of Features: " << ann.nNodes[0] << std::endl;
  std::cout << "Number of Classes: " << ann.nNodes[ann.nLayers-1] << std::endl;
  std::cout << "Number of Layers: " << ann.nLayers << std::endl;
  std::cout << "Total Number of Nodes: " << ann.tNodes << std::endl;
  std::cout << "Number of Nodes per Layer: ";
  for(unsigned int i = 0; i < ann.nNodes.size(); i++){
    std::cout << ann.nNodes[i] << ", ";
  }
  std::cout << std::endl;
  std::cout << "List of Activation IDs: ";
  unsigned int p = 0;
  for(unsigned int i = 0; i < ann.nNodes.size(); i++){
    std::cout << "L" << i << "("<< ann.nNodes[i] <<"): ";
    for(unsigned int j = 0; j < ann.nNodes[i]; j++){
      std::cout << ann.actIDs[p] << ", ";  
      p++;
    }
  }
  std::cout << std::endl;
  std::cout << "Weights("<< size(ann.weights) << "): ";
  for(unsigned int i = 0; i < ann.weights.size(); i++){
    std::cout << "\n\tL" << i << "("<< ann.weights[i].size() <<") ";
    for(unsigned int j = 0; j < ann.weights[i].size(); j++){
      std::cout << ann.weights[i][j] << ", ";
    }
  }
  std::cout << std::endl;
  std::cout << "Bias("<< ann.bias.size() << "): ";
  for(unsigned int i = 0; i < ann.bias.size(); i++){
    std::cout << ann.bias[i] << ", ";
  }
  std::cout << std::endl << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
/// Initializers
///////////////////////////////////////////////////////////////////////////////
void initWeights(
  std::vector<std::vector<DTYPE>> &weights,
	std::vector<unsigned int> nNodes
){
  for(unsigned int i = 0; i < nNodes.size() - 1; i++){
    unsigned int j = nNodes[i] * nNodes[i+1];
    weights.push_back(rng(j));
  }
  return;
}
void initWeights(struct Ann &ann){
  // Unpack
  std::vector<unsigned int> nNodes = ann.nNodes;
  // Getting weights
  std::vector<std::vector<DTYPE>> weights;
  initWeights(weights, nNodes);
  // Pack
  ann.weights = weights;
  return;
}

Ann initANN(
	unsigned int nFeat, 
	unsigned int nClasses, 
	unsigned int nLayers
){
  // Getting Default Number of Nodes Per Layer
  std::vector<unsigned int> nNodes(nLayers, nClasses);
  nNodes[0] = nFeat;
  // Getting Number of total Nodes
  unsigned int tNodes = sum(nNodes);
  // Getting Default Activation List 
  std::vector<unsigned int> actIDs(tNodes, 1);
  for(unsigned int i = tNodes-1; i > tNodes-nNodes[nNodes.size()-1]-1; i--){
    actIDs[i] = 2; 
  }
  // Initializing Weights
  std::vector<std::vector<DTYPE>> weights;
  initWeights(weights, nNodes);
  // Initializing Bias
  std::vector<DTYPE> bias(tNodes-nFeat, 0);
  // Packing
  struct Ann ann;
  ann.nLayers = nLayers;
  ann.nNodes = nNodes;
  ann.tNodes = tNodes;
  ann.actIDs = actIDs;
  ann.lossID = 0;
  ann.weights = weights;
  ann.bias = bias;
  
  return ann;
}
Ann initANN(
	unsigned int nFeat, 
	unsigned int nClasses, 
	unsigned int nLayers,
	std::vector<unsigned int> nNodes
){
  struct Ann ann; 
  if(nNodes.size() != nLayers){
    errPrint("ERROR - initANN: nNodes size does not match nLayers.");
    std::cout << nNodes.size() << ":" << nLayers << std::endl;
    exit(1);
    return ann;
  }
  // Getting Default Number of Nodes Per Layer
  nNodes[0] = nFeat;
  nNodes[nNodes.size()-1] = nClasses;
  // Getting Number of total Nodes
  unsigned int tNodes = sum(nNodes);
  // Getting Default Activation List 
  std::vector<unsigned int> actIDs(tNodes, 1);
  for(unsigned int i = tNodes-1; i > tNodes-nNodes[nNodes.size()-1]-1; i--){
    actIDs[i] = 2; 
  }
  // Initializing Weights
  std::vector<std::vector<DTYPE>> weights;
  initWeights(weights, nNodes);
  // Initializing Bias
  std::vector<DTYPE> bias(tNodes-nFeat, 0);
  // Packing
  ann.nLayers = nLayers;
  ann.nNodes = nNodes;
  ann.tNodes = tNodes;
  ann.actIDs = actIDs;
  ann.lossID = 0;
  ann.weights = weights;
  ann.bias = bias;
  
  return ann;
}

Ann initANN(struct ANN_Ambit ann_, struct Data train){
  unsigned int nFeat = train.nFeat;
  unsigned int nClasses = train.nClasses;
  unsigned int nLayers = ann_.nLayers;
  
  std::vector<unsigned int> nNodes;
  nNodes.push_back(nFeat);
  for(unsigned int i = 0; i < ann_.hNodes.size(); i++){
    nNodes.push_back(ann_.hNodes[i]);
  }
  nNodes.push_back(nClasses);

  if(nNodes.size() != nLayers){

    errPrint(
      "ERROR - initANN(ANN_Ambit, Data): nNodes.size does not match the number of layers"
    );
    std::cout << nNodes.size() << ":" << nLayers << std::endl;
    exit(1);
  }

  srand(ann_.wseed);
  struct Ann ann = initANN(
    nFeat,
    nClasses,
    nLayers,
    nNodes
  );

  for(unsigned int i = 0; i < ann_.ActIDSets.size(); i++){
    setActID(
      ann,
      ann_.ActIDSets[i].ID,
      ann_.ActIDSets[i].layerStrt,
      ann_.ActIDSets[i].layerEnd,
      ann_.ActIDSets[i].nodeStrt,
      ann_.ActIDSets[i].nodeEnd
    );
  }
  return ann;
}

void getDataSets(
  struct Data &train, 
  struct Data &test, 
  struct Data data
){
  // Unpack
  unsigned int nFeat = data.nFeat;
  double ratio = data.ratio;
  unsigned int nSamp = data.nSamp;
  unsigned int sseed = data.sseed;

  // Get Training Set
  srand(sseed);
  unsigned int nSamp_Train = (ratio / 100)*nSamp;
  std::vector<unsigned int> trainSet = rng_unq(nSamp_Train, 0, nSamp);
  std::cout << "Training Set: " << std::endl;
  print(trainSet);

  // Sort Features and Observations
  std::vector<DTYPE> train_feat;
  std::vector<unsigned int> train_obs;

  std::vector<DTYPE> test_feat;
  std::vector<unsigned int> test_obs;
  
  for(unsigned int i = 0; i < nSamp; i++){
    // if(inVec(i,trainSet)){
    if(std::find(trainSet.begin(), trainSet.end(), i) != trainSet.end()){
      for(unsigned int j = nFeat*i; j < nFeat*i+nFeat; j++){
        train_feat.push_back(data.feat[j]);
      }
      train_obs.push_back(data.obs[i]);
      if(train_feat.size()/nFeat != train_obs.size()){
        errPrint("ERROR - getDataSet: Features do not match observations.");
        std::cout << train_feat.size()/nFeat << ":" << train_obs.size() << std::endl;
      }
    }else{
      for(unsigned int j = nFeat*i; j < nFeat*i+nFeat; j++){
        test_feat.push_back(data.feat[j]);
      }
      test_obs.push_back(data.obs[i]);
      if(test_feat.size()/nFeat != test_obs.size()){
        errPrint("ERROR - getDataSet: Features do not match observations.");
        std::cout << test_feat.size()/nFeat << ":" << test_obs.size() << std::endl;
      }
    }
  }

  // Pack
  train.nFeat = nFeat;
  train.nClasses = data.nClasses;
  train.nSamp = nSamp_Train;
  train.sseed = sseed;
  train.ratio = ratio;
  train.feat = train_feat;
  train.obs = train_obs;

  test.nFeat = nFeat;
  test.nClasses = data.nClasses;
  test.nSamp = nSamp - nSamp_Train;
  test.sseed = sseed;
  test.ratio = 100 - ratio;
  test.feat = test_feat;
  test.obs = test_obs;
};

///////////////////////////////////////////////////////////////////////////////
/// Setters
///////////////////////////////////////////////////////////////////////////////
/// ANN
// Get Activation ID Postion
unsigned int getAIDP(
  std::vector<unsigned int> nNodes, 
  unsigned int layerN, 
  unsigned int nodeN /*0*/
){
  std::vector<unsigned int> pre_nNodes = {nNodes.begin(), nNodes.begin()+layerN};
  unsigned int position = sum(pre_nNodes)+nodeN;
  return position;
}

void setActID(
  std::vector<unsigned int> &actIDs,
  std::vector<unsigned int> nNodes,
  unsigned int ID,
  unsigned int layerN,
  unsigned int nodeN
){
  // Get position
  unsigned int position = getAIDP(nNodes, layerN, nodeN);
  // Set actIDs
  actIDs[position] = ID;
  return;
}

void setActID(
  struct Ann &ann,
  unsigned int ID,
  unsigned int layerNStrt,
  unsigned int layerNEnd /*UINT_MAX*/,
  unsigned int nodeNStrt /*0*/,
  unsigned int nodeNEnd /*UINT_MAX*/
){
  // Unpack
  std::vector<unsigned int> actIDs = ann.actIDs;
  std::vector<unsigned int> nNodes = ann.nNodes;
  // Get Limits
  if(layerNEnd > nNodes.size()){layerNEnd = nNodes.size();}
  // Get new ActIDs
  for(unsigned int i = layerNStrt; i < layerNEnd; i++){
    if(nodeNEnd > nNodes[i]){nodeNEnd = nNodes[i];}
    for(unsigned int j = nodeNStrt; j < nodeNEnd; j++){
      setActID(actIDs, nNodes, ID, i, j);
    }
  }
  // Pack
  ann.actIDs = actIDs;
  
  return;
}
