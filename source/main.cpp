#include "include/ann.h"

int main(int numInputs, char * inputs[]){
  // Create Defaults;
  struct Alpha alpha;
  struct ANN_Ambit ann_;
  struct Read_Ambit read_(inputs[1]);

  // Set changes
  if(getSetup(alpha, ann_, read_, numInputs, inputs)){return 1;}

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
  // BUG(
    std::cout << "\nTraining Set" << std::endl;
    print(train);
    std::cout << "\nTesting Set" << std::endl;
    print(test);
  // )
  

  // // Build ANN
  struct Ann ann = initANN(ann_, train);
  std::cout << "\nGetting ANN" << std::endl;
  print(ann);

  // Train
  struct Results train_results(train.nSamp, train.nClasses, train.nFeat);
  trainNN(ann, train, train_results, alpha, ann_.maxIter);
  
  std::cout << "\nUpdated Weights and Bias" << std::endl;
  print(ann.weights, ann.bias);
  std::cout << "\nTraining Results" << std::endl;
  print(train_results);
  
  // Test
  struct Results test_results(test.nSamp, test.nClasses, test.nFeat);
  testNN(ann, test, test_results);
  
  std::cout << "\nTesting Results" << std::endl;
  print(test_results);

  double trainPC = (double)train_results.uint_ambit/train.nSamp*100;
  double testPC = (double)test_results.uint_ambit/test.nSamp*100;
  double totalPC = ((double)train_results.uint_ambit+(double)test_results.uint_ambit)/data.nSamp*100;
  double totalErr = test_results.double_ambit + train_results.double_ambit;
  fprintf(
    stderr, "%u, %f, %f, %u, %u, %f, %f, %f, %f, %f, %f\n", 
    ann_.maxIter, alpha.alpha, data.ratio, 
    data.sseed, ann_.wseed, 
    testPC, trainPC, totalPC,
    test_results.double_ambit, train_results.double_ambit, totalErr
  );
  return 0;
}