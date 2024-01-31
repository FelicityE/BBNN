#include "include/ann.h"

int main(int numInputs, char * inputs[]){
  // Create Defaults;
  struct MetaRead read(inputs[1]);
  struct Meta meta;
  // Set changes
  setup(read, meta, numInputs, inputs);
  // Get trainging and testing data
  struct Data train;
  struct Data test;
  // build ANN
  struct Ann ann;

  
  return 0;
}