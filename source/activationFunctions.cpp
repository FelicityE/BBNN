#include "include/activationFunctions.h"
#include "include/utility.h"

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
  for(unsigned int i = 0; i < layer.size(); i++){
    activation.push_back(ReLu(layer[i]));
  }
  return activation;
}
std::vector<double> dReLu(std::vector<double> layer, int stop, int meta){
  BUGT1(
    std::cout << "\nRunning dReLu" << std::endl;
    // std::cout << "Layer Size: " << layer.size() << std::endl;
  )
  std::vector<double> dA;
  for(unsigned int i = 0; i < layer.size(); i++){
    dA.push_back(dReLu(layer[i]));
  }

  BUGT1(
    print("dA", dA);
    std::cout << "Returning\n" << std::endl;
  )
  return dA;
}

///////////////////////////////////////////////////////////////////////////////
// Layer Functions
///////////////////////////////////////////////////////////////////////////////
std::vector<double> softMax(std::vector<double> layer, int stop, int meta){
  BUGT1(
    std::cout << "\nRunning SoftMax" << std::endl;
    std::cout << "Layer Size: " << layer.size() << std::endl;
  )
  std::vector<double> predictiveProbability;
  double denom = 0; 
  for(unsigned int i = 0; i < layer.size(); i++){
    denom += exp(layer[i]);
  }
  for(unsigned int i = 0; i < layer.size(); i++){
    predictiveProbability.push_back(exp(layer[i])/denom);
  }
  return predictiveProbability;
}

std::vector<double> dSoftMax(std::vector<double> layer, int stop, int g_obs){
  // dp_i/da_j e.g. dpda[0] = dp0/da0, dpda[1] = dp0/da1, dpda[2] = dp1/da0 etc.
  BUGT1(
    std::cout << "\nRunning dSoftMax" << std::endl;
    std::cout << "Layer Size: " << layer.size() << std::endl;
  )
  std::vector<std::vector<double>> dpda;

  double denom = 0;
  for(unsigned int i = 0; i < layer.size(); i++){
    denom += exp(layer[i]);
  }
  BUGT1(
    print("denominator",denom);
    std::cout << "For each Node" << std::endl;
  )
  for(unsigned int i = 0; i < layer.size(); i++){
    double numer = exp(layer[i]);
    BUGT1(
      std::cout << "\tNode: " << i << std::endl;
      print("\tnumerator", numer);
      std::cout << "For each Node" << std::endl;
    )
    std::vector<double> obs;
    for(unsigned int j = 0; j < layer.size(); j++){
      BUGT1(
        std::cout << "\t\tNode: " << j << std::endl;
      )
      if(j == i){
        obs.push_back(((numer*denom)-(exp(layer[j])*numer))/(denom*denom));
      }else{
        obs.push_back(-(exp(layer[j])*numer)/(denom*denom));
      }
    }
    dpda.push_back(obs);
    BUGT1(
      print("dpda", dpda[dpda.size()-1]);
    )
  }
  
  BUGT1(
    std::cout << "g_obs" << g_obs << std::endl;
    print("dpda",dpda[g_obs]);
    std::cout << "Returning\n" << std::endl;
  )
  return dpda[g_obs];
}

std::vector<double> argMax(std::vector<double> layer, int stop, int meta){
  double max = layer[0];
  unsigned int index = 0;

  for(unsigned int i = 1; i < layer.size(); i++){
    if(layer[i] > max){
      max = layer[i];
      index = i;
    }
  }

  std::vector<double> out;
  for(unsigned int i = 0; i < layer.size(); i++){
    if(i == index){
      out.push_back(1);
    }else{
      out.push_back(0);
    }
  }

  return out;
}