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
std::vector<DTYPE> relu(std::vector<DTYPE> layer, AMBIT_TYPE ambit){
  std::vector<DTYPE> activation(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    activation[i] = relu(layer[i]);
  }
  return activation;
}
std::vector<DTYPE> drelu(std::vector<DTYPE> layer, AMBIT_TYPE ambit){
  std::vector<DTYPE> dA(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    dA[i] = drelu(layer[i]);
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
std::vector<DTYPE> sigmoid(std::vector<DTYPE> layer, AMBIT_TYPE ambit){
  std::vector<DTYPE> temp(layer.size(), 0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = sigmoid(layer[i]);
  }
  return temp;
}
std::vector<DTYPE> dsigmoid(std::vector<DTYPE> layer, AMBIT_TYPE ambit){
  std::vector<DTYPE> temp(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = dsigmoid(layer[i]);
  }
  return temp;
}

/// Softmax
std::vector<DTYPE> softmax(std::vector<DTYPE> layer, AMBIT_TYPE ambit){
  std::vector<DTYPE> predictiveProbability(layer.size(),0);
  DTYPE denom = 0; 
  for(unsigned int i = 0; i < layer.size(); i++){
    denom += exp(layer[i]);
  }
  for(unsigned int i = 0; i < layer.size(); i++){
    predictiveProbability[i]= exp(layer[i])/denom;
  }
  return predictiveProbability;
}
std::vector<DTYPE> dsoftmax(std::vector<DTYPE> layer, AMBIT_TYPE obs){
  // dp_i/da_j e.g. dpda[0] = dp0/da0, dpda[1] = dp0/da1, dpda[2] = dp1/da0 etc.
  std::vector<std::vector<DTYPE>> dpda(layer.size());

  DTYPE denom = 0;
  for(unsigned int i = 0; i < layer.size(); i++){
    denom += exp(layer[i]);
  }
  for(unsigned int i = 0; i < layer.size(); i++){
    DTYPE numer = exp(layer[i]);
    dpda[i] = std::vector<DTYPE>(layer.size(),0);
    for(unsigned int j = 0; j < layer.size(); j++){
      if(j == i){
        dpda[i][j] = ((numer*denom)-(exp(layer[j])*numer))/(denom*denom);
      }else{
        dpda[i][j] = -(exp(layer[j])*numer)/(denom*denom);
      }
    }
  }
  return dpda[obs]; 
}

/// Argmax
std::vector<DTYPE> argmax(std::vector<DTYPE> layer, AMBIT_TYPE ambit){
  DTYPE max = layer[0];
  unsigned int index = 0;

  for(unsigned int i = 1; i < layer.size(); i++){
    if(layer[i] > max){
      max = layer[i];
      index = i;
    }
  }
  // return std::vector<DTYPE>(layer.size(), (DTYPE)index);

  std::vector<DTYPE> out(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    if(i == index){
      out[i] = 1;
      return out;
    }
  }
  return out;
}
std::vector<DTYPE> dargmax(std::vector<DTYPE> layer, AMBIT_TYPE ambit){
  std::vector<DTYPE> temp(layer.size(), 0.0);
  return temp;
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
  unsigned int obs
){
  std::vector<DTYPE> cross;
  for(unsigned int i = 0; i < layer.size(); i++){
    cross.push_back(crossentropy(layer[i]));
  }
  return cross;
}
std::vector<DTYPE> dcrossentropy(
  std::vector<DTYPE> layer,
  unsigned int obs
){
  std::vector<DTYPE> dcross;
  for(unsigned int i = 0; i < layer.size(); i++){
    dcross.push_back(dcrossentropy(layer[i]));
  }
  return dcross;
}

