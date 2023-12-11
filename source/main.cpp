#include "include/ann.h"
#include "include/utility.h"
// std::vector<double> percentCorrect;

int main(int numInputs, char * inputs[]){
  if(numInputs < 2){
    std::cout << "ERROR - main input: missing data filepath." << std::endl;
    return 1;
  }
  for(unsigned int i = 0; i < numInputs; i++){
    std::cout << inputs[i] << " ";
  }
  std::cout << std::endl;

  // Inputs Filename is always after ./main
  std::string dataFilePath = inputs[1];
  unsigned int IDposition = 0; // Class ID column number
  unsigned int nLayers = 3;  // Number of layers including input and output layer
  std::vector<unsigned int> nHiddenNodes{2}; // number of nodes for each hidden layer
  unsigned int skipCol = 0; // Number of columns to skip at the begining of the dataset
  unsigned int skipColPat = 0; // Number of columns to skip in a pattern (everyother, every third, etc.)

  bool Adam = true;
  double alpha = 0.01; // The learning rate
  double beta1 = 0.9; // The first moment rate of decay
  double beta2 = 0.999; // The second moment rate of decay

  unsigned int maxIter = 1000; // Max number of iterations
  double ratio = 70; // Ratio of training to testing set
  int sample_seed = 0; // sample selection seed 
  int weights_seed = 42; // Initial weights seed

  if(numInputs > 2){
    for(unsigned int i = 2; i < numInputs; i++){
      if(match(inputs[i],"-Adam")){
        Adam = false;
      }else if(match(inputs[i], "alpha")){
        alpha = std::stod(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "beta")){
        beta1 = std::stod(inputs[i+1]);
        beta2 = std::stod(inputs[i+2]);
        i += 2;
      }
      
      else if(match(inputs[i], "Layers")){
        nLayers = std::stoi(inputs[i+1]);
        i++;
        nHiddenNodes[0] = std::stoi(inputs[i+1]);
        for(unsigned int j = 1; j < nLayers-2; j++){
          nHiddenNodes.push_back(std::stoi(inputs[i+1]));
        }
        i++;
      }else if(match(inputs[i], "hNodes")){
        nLayers = std::stoi(inputs[i+1])+2;
        std::cout << "\nNumber of Layers: " << nLayers << std::endl;
        i++;
        nHiddenNodes[0] = std::stoi(inputs[i+1]);
        i++;
        for(unsigned int j = 1; j < nLayers-2; j++){
          nHiddenNodes.push_back(std::stoi(inputs[i+1]));
          i++;
        }
        // print("Hidden Nodes", nHiddenNodes);
        std::cout << std::endl;
      }else if(match(inputs[i], "ID_column")){
        IDposition = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "skip_column")){
        skipCol = std::stoi(inputs[i+1]);
        i++;
      }
      
      else if(match(inputs[i],"maxIter")){
        maxIter = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "ratio")){
        ratio = std::stod(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "sseed")){
        sample_seed = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "wseed")){
        weights_seed = std::stoi(inputs[i+1]);
        i++;
      }

      // else if(match(inputs[i], "actFuns"))
      
      else{
        std::cout << "ERROR - main input: input["<< i <<"], " << inputs[i] <<  ", not found." << std::endl;
      }
    }
  }

  if(ERRPRINT){
    //  fprintf(stderr, "#####################################\n");
   fprintf(stderr, "%u, %f, %f, %u, %u, ", maxIter, alpha, ratio, sample_seed, weights_seed); 
  }
  srand(sample_seed);

  // Data Read In
  std::vector<std::vector<double>> samples;
  std::vector<std::vector<double>> obs;
  getDataID(dataFilePath, samples, obs, IDposition, 1, skipCol);
  

  std::vector<std::vector<double>> train;
  std::vector<std::vector<double>> obsTrain;
  std::vector<std::vector<double>> test;
  std::vector<std::vector<double>> obsTest;
  getTrainingSet(samples, obs, train, obsTrain, test, obsTest, ratio);

  std::cout << "\n\nTraining Size: " << train.size() << std::endl;
  // std::cout << "Observed Trainging Size: " << obsTrain.size() << std::endl;
  std::cout << "Test Size: " << test.size() << std::endl;
  // std::cout << "Observed Test Size: " << obsTest.size() << std::endl;

  // Set/Get Adam, if not set default
  std::vector<double> abbe = {
    alpha, // alpha: Learning rate, step-size
    beta1, // beta 1: Exponential decay rate 1
    beta2, // beta 2: Exponential decay rate 2
    10e-8 // epsilon: Idk what this is but don't change it!
  };
  
  std::vector<std::vector<double>> weights;
  std::vector<std::vector<double>> bias;

  // Get number of Nodes for each layer
  unsigned int nFeatures = samples[0].size();
  std::cout << "Number of features: " << nFeatures << std::endl;
  
  unsigned int nOutputs = obs[0].size();
  std::cout << "Number of nOutputs: " << nOutputs << std::endl;

  std::vector<unsigned int> nNodes{nFeatures};
  vecAppend(nNodes, nHiddenNodes);
  nNodes.push_back(nOutputs);
  std::cout << "Number of Layers: " << nLayers << std::endl;
  std::cout << "Number of nodes per Layer: ";
  for(unsigned int i = 0; i < nNodes.size(); i++){
    std::cout << nNodes[i] << ", ";
  }
  std::cout << std::endl;


  int * actType = (int*)malloc(sizeof(int)*nLayers);
  for(unsigned int i = 0; i < nLayers; i++){
    actType[i] = 2;
  }

  BUGT1(
    std::cout << "Creating activation function arrays." << std::endl;
  )

  unsigned int nActs = nLayers-1;
  // Allocating activation functions and activation function derivatives
  activationFunction ** actFun = (activationFunction**)malloc(sizeof(activationFunction*)*nActs);
  activationFunction ** actFunTest = (activationFunction**)malloc(sizeof(activationFunction*)*nActs);
  activationFunction ** dActFun = (activationFunction**)malloc(sizeof(activationFunction*)*nActs);
  
  BUGT1(
    std::cout << "\t Malloced" << std::endl;
  )

  for(unsigned int i = 0; i < nActs-1; i++){
    if(actType[i] == 1){
      actFun[i] = (activationFunction*)malloc(sizeof(activationFunction)*nNodes[i]);
      actFunTest[i] = (activationFunction*)malloc(sizeof(activationFunction)*nNodes[i]);
      dActFun[i] = (activationFunction*)malloc(sizeof(activationFunction)*nNodes[i]);
      for(unsigned int j = 0; j < nNodes[i]; j++){
        actFun[i][j] = ReLu;
        actFunTest[i][j] = ReLu;
        dActFun[i][j] = dReLu;
      }
    }
    if(actType[i] == 2){
      actFun[i] = (activationFunction*)malloc(sizeof(activationFunction)*1);
      actFunTest[i] = (activationFunction*)malloc(sizeof(activationFunction)*1);
      dActFun[i] = (activationFunction*)malloc(sizeof(activationFunction)*1);
      actFun[i][0] = ReLu;
      actFunTest[i][0] = ReLu;
      dActFun[i][0] = dReLu;
    }
    BUGT1(
      std::cout << "\t L " << i << " finished."<< std::endl;
    )
  }

  BUGT1(
    std::cout << "\t setting final layer" << std::endl;
    std::cout << "\t nActs-1: " << nActs-1 << std::endl;
  )
  actFun[nActs-1] = (activationFunction*)malloc(sizeof(activationFunction)*1);
  actFunTest[nActs-1] = (activationFunction*)malloc(sizeof(activationFunction)*1);
  dActFun[nActs-1] = (activationFunction*)malloc(sizeof(activationFunction)*1);
  actFun[nActs-1][0] = softMax;
  actFunTest[nActs-1][0] = argMax;
  dActFun[nActs-1][0] = dSoftMax;

  BUGT1(
    std::cout << "\t Setting Loss Function" << std::endl;
  )

  lossFunction dLossFun = dCrossEntropy;
  
  BUGT1(
    std::cout << "Running Training " << std::endl;
  )
  srand(weights_seed);
  double trainPC = trainSNN(
    train,
    obsTrain,
    nNodes,
    weights,
    bias,
    actFun,
    dActFun,
    actType,
    dLossFun,
    maxIter,
    Adam,
    abbe
  );

  BUGT1(
    std::cout << "Running Tests " << std::endl;
  )

  std::vector<std::vector<double>> resultVals;
  std::vector<bool> results;
  testSNN(
    test,
    obsTest,
    nNodes,
    weights,
    bias,
    actFunTest,
    actType,
    resultVals,
    results
  );

  // if(ERRPRINT){
  //   fprintf(stderr, "%f, ", double(sumVectR(results))/results.size()*100);
  //   for(unsigned int i = 0; i < percentCorrect.size()-1; i++){
  //     fprintf(stderr, "%f, ", percentCorrect[i]);
  //   }
  //   fprintf(stderr, "%f", percentCorrect[percentCorrect.size()-1]);
  //   fprintf(stderr, "\n");
  // }
  // std::cout << std::endl;

  BUGT1(
    std::cout << "Printing " << std::endl;
  )

  std::cout << "Training Error: " << trainPC << std::endl;
  double testPC = 1-double(sumVectR(results))/results.size();
  std::cout << "Testing Error: " << testPC << std::endl;
  double totalPC = (trainPC*train.size() + testPC*test.size())/(train.size()+test.size());
  std::cout << "Total Error: " << totalPC << std::endl;
  
  
  std::cout << std::endl;
  if(ERRPRINT){
    fprintf(stderr, "%f, %f, %f\n", testPC, trainPC, totalPC); 
  }

  printWB(weights,bias,nNodes);


  // std::string wbFileName = "out/WB_L" + to_string(nLayers) + "_N";
  // for(unsigned int i = 0; i < nLayers; i++){
  //   wbFileName += to_string(nNodes[i]);
  //   if(i != nLayers-1){wbFileName += "-";}
  // }
  // wbFileName += ".txt";
  // printTo(wbFileName, weights);
  // appendToFile(wbFileName, bias);
  std::cout << "###########################################################################" << std::endl;

  return 0;
} 