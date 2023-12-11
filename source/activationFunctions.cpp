#include "include/activationFunctions.h"

///////////////////////////////////////////////////////////////////////////////
// Add new activation functions below
// Be sure to add function definition to the header file
///////////////////////////////////////////////////////////////////////////////
/*
  inline std::vector<double> activationFunctionName(
    std::vector<double> layer, 
    std::vector<double> numberOfNodesInLayer,
    int metaData // any extra required data
  ){

  }
*/

///////////////////////////////////////////////////////////////////////////////
// Activation Functions
///////////////////////////////////////////////////////////////////////////////
double ReLu(double x){
  if(x <= 0){return 0;}
  return x;
}
double dReLu(double x){
  if(x <= 0){return 0;}
  return 1;
}

std::vector<double> ReLu(std::vector<double> layer, int stop, int meta){
  std::vector<double> activation;
  for(unsigned int i = 0; i < stop; i++){
    activation.push_back(ReLu(layer[i]));
  }
  return activation;
}
std::vector<double> dReLu(std::vector<double> layer, int stop, int meta){
  std::vector<double> dA;
  for(unsigned int i = 0; i < stop; i++){
    dA.push_back(dReLu(layer[i]));
  }
  return dA;
}

///////////////////////////////////////////////////////////////////////////////
// Layer Functions
///////////////////////////////////////////////////////////////////////////////
std::vector<double> softMax(std::vector<double> layer, int stop, int meta){
  std::vector<double> predictiveProbability;
  double denom = 0; 
  for(unsigned int i = 0; i < stop; i++){
    denom += exp(layer[i]);
  }
  for(unsigned int i = 0; i < stop; i++){
    predictiveProbability.push_back(exp(layer[i])/denom);
  }
  return predictiveProbability;
}

std::vector<double> dSoftMax(std::vector<double> layer, int stop, int obs){
  // dp_i/da_j e.g. dpda[0] = dp0/da0, dpda[1] = dp0/da1, dpda[2] = dp1/da0 etc.

  std::vector<std::vector<double>> dpda;

  double denom = 0;
  for(unsigned int i = 0; i < stop; i++){
    denom += exp(layer[i]);
  }

  for(unsigned int i = 0; i < stop; i++){
    double numer = exp(layer[i]);
    std::vector<double> obs;
    for(unsigned int j = 0; j < stop; j++){
      if(j == i){
        obs.push_back(((numer*denom)-(exp(layer[j])*numer))/(denom*denom));
      }else{
        obs.push_back(-(exp(layer[j])*numer)/(denom*denom));
      }
    }
    dpda.push_back(obs);
  }
  

  return dpda[obs];
}

std::vector<double> argMax(std::vector<double> layer, int stop, int meta){
  double max = layer[0];
  unsigned int index = 0;

  for(unsigned int i = 1; i < stop; i++){
    if(layer[i] > max){
      max = layer[i];
      index = i;
    }
  }

  std::vector<double> out;
  for(unsigned int i = 0; i < stop; i++){
    if(i == index){
      out.push_back(1);
    }else{
      out.push_back(0);
    }
  }

  return out;
}