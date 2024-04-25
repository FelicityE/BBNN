#include "include/ann.h"

int main(int numInputs, char * inputs[]){
  // Create Defaults;
  struct Alpha alpha;
  struct ANN_Ambit annbit;
  struct Read_Ambit readbit(inputs[1]);

  // Set changes
  if(getSetup(alpha, annbit, readbit, numInputs, inputs)){return 1;}

  // Setting Act List
  setActList(readbit.actList, annbit.actDefault);
  BUG(print(readbit.actList, "Actlist");)

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

    unsigned int nHL = annbit.hNodes.size();
    unsigned int nActs = readbit.actList.size();
    std::vector<struct ActID_Set> temp = annbit.ActIDSets;
    if(readbit.diversify){
      annbit.ActIDSets = std::vector<struct ActID_Set>(nActs);
      getNodeActivations(
        annbit.ActIDSets,
        readbit.actList, 
        annbit.hNodes, 
        readbit.aseed
      );
      for(unsigned int i = 0; i < temp.size(); i++){
        annbit.ActIDSets.push_back(temp[i]);
      }
    }
    
    

    // Get Identifier
    double stamp = omp_get_wtime();
    // print(stamp, "Stamp", false);
    fprintf(stdout, "stamp: %f ", stamp);
    // Print Meta Data
    printTo(annbit, readbit, alpha, data, stamp);
    // Run ANN
    runANN(
      alpha,
      annbit,
      data,
      stamp
    );
    BUG(print(annbit.actCnts, "Main - Activation Counts");)
    printTo(annbit.logpath, annbit.actCnts, annbit.hNodes);

    double e_time = omp_get_wtime();
    fprintf(stderr, "%f, %f\n", stamp, e_time-stamp); 
  }
  return 0;
}