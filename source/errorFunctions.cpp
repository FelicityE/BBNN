#include "include/errorFunctions.h"

///////////////////////////////////////////////////////////////////////////////
// Add new loss functions below
// Be sure to add function definition in header file
///////////////////////////////////////////////////////////////////////////////
/*
  inline std::vector<double> lossFunctionName(
    std::vector<double> lastLayer, 
    std::vector<double> observedValues
  ){

  }
*/

double crossEntropy(double predictiveProbability){
  return -log(predictiveProbability);
}
double dCrossEntropy(double predictiveProbability){
  return -1.0/predictiveProbability;
}
std::vector<double> crossEntropy(
  std::vector<double> layer,
  std::vector<double> obs
){
  std::vector<double> cross;
  for(unsigned int i = 0; i < layer.size(); i++){
    cross.push_back(crossEntropy(layer[i]));
  }
  return cross;
}
std::vector<double> dCrossEntropy(
  std::vector<double> layer,
  std::vector<double> obs
){
  std::vector<double> dcross;
  for(unsigned int i = 0; i < layer.size(); i++){
    dcross.push_back(dCrossEntropy(layer[i]));
  }

  BUGT3(print("dLoss", dcross));
  return dcross;
}

