#include "include/ann.h"

int main(int numInputs, char * inputs[]){
  // Create Defaults;
  struct Adam adam;
  struct ANN_Ambit ann_;
  struct Read_Ambit read_(inputs[1]);

  // Set changes
  if(getSetup(adam, ann_, read_, numInputs, inputs)){return 1;}

  // Get trainging and testing data
  struct Data data; // Full dataset
  if(getData(data, read_)){return 1;}
  BUG(
    std::cout << "Getting Data" << std::endl;
    print(data);
  )

  // Future: Separate training from testing and use batches for different ANN's
  struct Data train; // training set
  struct Data test; // test set
  getDataSets(train, test, data);
  BUG(
    std::cout << "\nTraining Set" << std::endl;
    print(train);
    std::cout << "\nTesting Set" << std::endl;
    print(test);
  )
  

  // // Build ANN
  struct Ann ann = initANN(ann_, train);
  std::cout << "\nGetting ANN" << std::endl;
  print(ann);

  // Train
  trainNN(ann, train, adam, ann_.maxIter);
  // Test
  struct Results result;
  testNN(ann, test, result);
  print(result);

  return 0;
}