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

  // Add header to output log if needed
  std::string header = buildHeader(data.nClasses);
  bool addheader = addHeader(annbit.logpath, header);
  
  if(readbit.analyze){
    runAnalysis(readbit, annbit, alpha, data, addheader);
  }else{
    if(addheader){
      std::string header = buildHeader(annbit.hNodes.size(), ACT1.size());
      addHeader(annbit.logpath, header, true);
      addheader = false;
    }

    std::vector<unsigned int> actout(annbit.hNodes.size()*ACT1.size(), 0);
    if(readbit.diversify){
      actout = getActIDs(annbit, data.nClasses);
    }

    // Get Identifier
    double stamp = omp_get_wtime();
    // Print Meta Data
    printTo(annbit, readbit, alpha, data, stamp);
    // Run ANN
    runANN(
      alpha,
      annbit,
      data,
      stamp
    );

    printTo(annbit.logpath, actout);
  }
  
  
  
  
  return 0;
}