#include "include/ann.h"

void setAdam(bool &ambit, bool value){ambit = value; return;}
void setAdam(double &ambit, double value){ambit = value; return;}

void setnLayers(Ann &ann, unsigned int value){
  ann.nLayers = value;
  for(unsigned int i = ann.nNodes.size(); i < ann.nLayers; i++){
    ann.nNodes.push_back(ann.nNodes[ann.nNodes.size()-1]);
  }
  return;
}
void setnNodes(
  std::vector<unsigned int> &nNodes, 
  unsigned int value, 
  unsigned int layer
){nNodes[layer] = value; return;}
void setnNodes(
  std::vector<unsigned int> &nNodes, 
  unsigned int value, 
  unsigned int layerStart,
  unsigned int layerEnd
){
  for(unsigned int i = layerStart; i < layerEnd; i++){
    setnNodes(nNodes, value, i);   
  }
  return;
}

void initWeights(Ann &ann){
  for(unsigned int i = 0; i < ann.nNodes.size()-1; i++){
    std::vector<DTYPE> temp;
    for(unsigned int j = 0; j < (ann.nNodes[i]*ann.nNodes[i+1]); j++){
      temp.push_back((DTYPE)rand()/RAND_MAX);
    }
    ann.weights.push_back(temp);
  }
  return;
}

void initBias(Ann &ann){
  for(unsigned int i = 1; i < ann.nNodes.size(); i++){
    for(unsigned int j = 0; j < ann.nNodes[i].size(); j++){
      ann.bias.push_back(0);
    }
  }
  return;
}