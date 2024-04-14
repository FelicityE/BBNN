#include "include/ann.h"

int main(int numInputs, char * inputs[]){
  // Create Defaults;
  struct Alpha alpha;
  struct ANN_Ambit annbit;
  struct Read_Ambit readbit(inputs[1]);

  // Set changes
  if(getSetup(alpha, annbit, readbit, numInputs, inputs)){return 1;}

  // Get trainging and testing data
  struct Data data; // Full dataset
  if(getData(data, readbit)){return 1;}
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

  // Get Initial Trial Stamp (stamp should change for each trial)
  std::time_t stamp = std::time(0);
  // Add header if needed
  std::string header = buildHeader(data.nClasses);
  addHeader(annbit.logpath, header);
  
  // Print Meta Data (within loop)
  printTo(annbit, readbit, alpha, data, stamp);

  struct Scores trainScores;
  struct Scores testScores;
  runANN(
    alpha,
    annbit,
    data,
    train,
    test,
    trainScores,
    testScores,
    stamp
  );

  // unsigned int tNodes = sum(annbit.hNodes)+data.nFeat+data.nClasses;
  std::string metapath = "../results/meta.csv";
  

  

  return 0;
}