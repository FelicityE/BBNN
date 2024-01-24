#include "include/activations.h"
///////////////////////////////////////////////////////////////////////////////
// Activation Functions
///////////////////////////////////////////////////////////////////////////////
/// ReLu
DTYPE relu(DTYPE x){
  if(x <= 0){return 0;}
  return x;
}
DTYPE drelu(DTYPE x){
  if(x <= 0){return 0;}
  return 1;
}
std::vector<DTYPE> relu(std::vector<DTYPE> layer, std::vector<DTYPE> ambit){
  std::vector<DTYPE> activation;
  for(unsigned int i = 0; i < layer.size(); i++){
    activation.push_back(relu(layer[i]));
  }
  return activation;
}
std::vector<DTYPE> drelu(std::vector<DTYPE> layer, std::vector<DTYPE> ambit){
  std::vector<DTYPE> dA;
  for(unsigned int i = 0; i < layer.size(); i++){
    dA.push_back(drelu(layer[i]));
  }
  return dA;
}

/// Sigmoid
DTYPE sigmoid(DTYPE x){
  return 1/(1+exp(-x));
}
DTYPE dsigmoid(DTYPE x){
  return sigmoid(x)*(1-sigmoid(x));
}
std::vector<DTYPE> sigmoid(std::vector<DTYPE> layer, std::vector<DTYPE> ambit){
  std::vector<DTYPE> temp;
  for(unsigned int i = 0; i < layer.size(); i++){
    temp.push_back(sigmoid(layer[i]));
  }
  return temp;
}
std::vector<DTYPE> dsigmoid(std::vector<DTYPE> layer, std::vector<DTYPE> ambit){
  std::vector<DTYPE> temp;
  for(unsigned int i = 0; i < layer.size(); i++){
    temp.push_back(dsigmoid(layer[i]));
  }
  return temp;
}

/// Softmax
std::vector<DTYPE> softmax(std::vector<DTYPE> layer, std::vector<DTYPE> ambit){
  std::vector<DTYPE> predictiveProbability;
  DTYPE denom = 0; 
  for(unsigned int i = 0; i < layer.size(); i++){
    denom += exp(layer[i]);
  }
  for(unsigned int i = 0; i < layer.size(); i++){
    predictiveProbability.push_back(exp(layer[i])/denom);
  }
  return predictiveProbability;
}
std::vector<DTYPE> dsoftmax(std::vector<DTYPE> layer, std::vector<DTYPE> ambit){
  // dp_i/da_j e.g. dpda[0] = dp0/da0, dpda[1] = dp0/da1, dpda[2] = dp1/da0 etc.
  std::vector<std::vector<DTYPE>> dpda;

  DTYPE denom = 0;
  for(unsigned int i = 0; i < layer.size(); i++){
    denom += exp(layer[i]);
  }
  for(unsigned int i = 0; i < layer.size(); i++){
    DTYPE numer = exp(layer[i]);
    std::vector<DTYPE> obs;
    for(unsigned int j = 0; j < layer.size(); j++){
      if(j == i){
        obs.push_back(((numer*denom)-(exp(layer[j])*numer))/(denom*denom));
      }else{
        obs.push_back(-(exp(layer[j])*numer)/(denom*denom));
      }
    }
    dpda.push_back(obs);
  }
  int obsClass = int(ambit[0]);
  return dpda[obsClass]; 
}

/// Argmax
std::vector<DTYPE> argmax(std::vector<DTYPE> layer, std::vector<DTYPE> ambit){
  DTYPE max = layer[0];
  unsigned int index = 0;

  for(unsigned int i = 1; i < layer.size(); i++){
    if(layer[i] > max){
      max = layer[i];
      index = i;
    }
  }

  std::vector<DTYPE> out;
  for(unsigned int i = 0; i < layer.size(); i++){
    if(i == index){
      out.push_back(1);
    }else{
      out.push_back(0);
    }
  }

  return out;
}

///////////////////////////////////////////////////////////////////////////////
// Loss Functions
///////////////////////////////////////////////////////////////////////////////
DTYPE crossentropy(DTYPE x){
  return -log(x);
}
DTYPE dcrossentropy(DTYPE x){
  return -1.0/x;
}
std::vector<DTYPE> crossentropy(
  std::vector<DTYPE> layer,
  int obs
){
  std::vector<DTYPE> cross;
  for(unsigned int i = 0; i < layer.size(); i++){
    cross.push_back(crossentropy(layer[i]));
  }
  return cross;
}
std::vector<DTYPE> dcrossentropy(
  std::vector<DTYPE> layer,
  int obs
){
  std::vector<DTYPE> dcross;
  for(unsigned int i = 0; i < layer.size(); i++){
    dcross.push_back(dcrossentropy(layer[i]));
  }
  return dcross;
}