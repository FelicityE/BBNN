#include "include/ann.h"

int main(int numInputs, char * inputs[]){
  // Create Defaults;
  // struct SetUp setup(inputs[1]);
  struct Adam adam;
  struct ANN_Ambit ann_;
  struct Read_Ambit read_;

  // Set changes
  if(getSetup(adam, ann_, read_, numInputs, inputs)){return 1;}

  // Get trainging and testing data
  struct Data train;
  struct Data test;
  initDataSets(train, test, read_);

  // Build ANN
  struct Ann ann = initANN(ann_, train);

  
  return 0;
}