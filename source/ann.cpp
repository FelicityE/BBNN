#include "include/ann.h"

///////////////////////////////////////////////////////////////////////////////
/// Init Functions
///////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<double>> initWeights(
  unsigned int nLayers, 
  std::vector<unsigned int> nNodes, 
  bool setRand
){
  std::vector<std::vector<double>> v;
  for(unsigned int i = 0; i < nLayers-1; i++){
    std::vector<double> temp;
    for(unsigned int j = 0; j < (nNodes[i]*nNodes[i+1]); j++){
      if(setRand){
        temp.push_back((double)rand()/RAND_MAX);
      }else{
        temp.push_back(0);
      }
    }
    v.push_back(temp);
  }
  return v;
}
std::vector<std::vector<double>> initBias(
  unsigned int nLayers, 
  std::vector<unsigned int> nNodes, 
  bool setRand
){
  std::vector<std::vector<double>> v;
  for(unsigned int i = 0; i < nLayers-1; i++){
    std::vector<double> temp;
    for(unsigned int j = 0; j < (nNodes[i+1]); j++){
      if(setRand){
        temp.push_back((double)rand()/RAND_MAX);
      }else{
        temp.push_back(0);
      }
    }
    v.push_back(temp);
  }
  return v;
}
// Init vector to Zero
std::vector<std::vector<double>> initZero(
  unsigned int outter, 
  std::vector<unsigned int> inner
){
  std::vector<std::vector<double>> vec;
  for(unsigned int i = 0; i < outter; i++){
    std::vector<double> temp;
    for(unsigned int j = 0; j < inner[i]; j++){
      temp.push_back(0.0);
    }
    vec.push_back(temp);
  }
  return vec;
}
std::vector<std::vector<double>> initZero(std::vector<std::vector<double>> copy){
  std::vector<std::vector<double>> vec;
  for(unsigned int i = 0; i < copy.size(); i++){
    std::vector<double> temp;
    for(unsigned int j = 0; j < copy[i].size(); j++){
      temp.push_back(0.0);
    }
    vec.push_back(temp);
  }
  return vec;
}

// Get Class Number
unsigned int classNumber(std::vector<double> lastlayer){
  for(unsigned int i = 0; i < lastlayer.size(); i++){
    if(lastlayer[i] == 1){return i;}
  }
  std::cout << "ERROR - classNumber(std::vector<double>): No class found." << std::endl;
  return -1;
}
std::vector<unsigned int> classNumber(std::vector<std::vector<double>> predicL){
  std::vector<unsigned int> classN;
  for(unsigned int i = 0; i < predicL.size(); i++){
    classN.push_back(classNumber(predicL[i]));
  }
  return classN;
}

// multiply Vector (dB*A) for backprop
std::vector<double> get_dW(std::vector<double> dB, std::vector<double> A){
  std::vector<double> dW;
  for(unsigned int i = 0; i < dB.size(); i++){
    for(unsigned int j = 0; j < A.size(); j++){
      dW.push_back(dB[i]*A[j]);
    }
  }
  return dW;
}
std::vector<double> get_inner_dB(
  std::vector<double> delta, 
  std::vector<double> W, 
  std::vector<double> fprimeL
){
  std::vector<double> dW = get_dW(delta, fprimeL);
  if(W.size() != dW.size()){
    std::cout << "ERROR - get_inner_dB: dW size is not the same as W." << std::endl;
    return dW;
  }
  for(unsigned int i = 0; i < dW.size(); i++){
    dW[i] *= W[i];
  }

  unsigned int dBsize = fprimeL.size();
  std::vector<double> dB(dBsize, 0.0);
  for(unsigned int i = 0; i < dW.size(); i++){
    dB[i%dBsize] += dW[i];
  }
  return dB;
}
///////////////////////////////////////////////////////////////////////////////
/// ANN Functions
///////////////////////////////////////////////////////////////////////////////
// Forward
// std::vector<double> runForward(
//   std::vector<double> features, 
//   unsigned int nLayers, 
//   std::vector<unsigned int> nNodes, 
//   std::vector<std::vector<double>> weights,
//   std::vector<std::vector<double>> bias
// ){
//   std::vector<std::vector<double>> layers;
//   std::vector<std::vector<double>> Acts;

//   // Set first Layer
//   layers.push_back(features);
//   Acts.push_back(features);

//   // For each hidden layer
//   for(unsigned int i = 1; i < nLayers-1; i++){
//     unsigned int l = i-1;
//     layers.push_back(
//       addVecR(
//         multMat(weights[l],nNodes[i],nNodes[l], Acts[l],nNodes[l],1), 
//         bias[l]
//       )
//     );

//     Acts.push_back(ReLu(layers[i]));
//   }

//   // Set last Layer
//   unsigned int ln = nLayers-1;
//   unsigned int lm = nLayers-2;

//   layers.push_back(
//     addVecR(
//       multMat(weights[lm],nNodes[ln],nNodes[lm], Acts[lm],nNodes[lm],1), 
//       bias[lm]
//     )
//   );
//   Acts.push_back(argMax(layers[ln]));

//   return Acts[ln];
// }
void runForward(
  // std::vector<double> features, 
  // unsigned int nLayers, 
  std::vector<unsigned int> nNodes, 
  std::vector<std::vector<double>> weights,
  std::vector<std::vector<double>> bias,
  std::vector<std::vector<double>> &layers, 
  std::vector<std::vector<double>> &Acts,
  activationFunction ** actFun,
  int * actType
){
  // Set first Layer // Should be set already
  // layers.resize(nLayers); 
  // layers[0] = features; 
  // Acts.resize(nLayers); 
  // Acts[0] = features;

  // For each hidden layer
  for(unsigned int i = 1; i < nNodes.size(); i++){
    unsigned int l = i-1;
    layers[i] = (
      addVecR(
        multMat(weights[l],nNodes[i],nNodes[l], Acts[l],nNodes[l],1), 
        bias[l]
      )
    );

    //uses a single node at a time
    if (actType[i] == 1){
      Acts[i].resize(layers[i].size());
      // for each node
      for(unsigned int k = 0; k < layers[i].size(); k++){
        std::vector<double> temp{layers[i][k]};
        std::vector<double> tempVector = (*actFun[i][k])(temp, 1, 0);
        Acts[i][k] = tempVector[0];
      }
    }

    // uses the whole layer at once and returns an array
    if (actType[i] == 2){
      Acts[i] = (*actFun[i][0])(layers[i], layers[i].size(), 1);
    }
  }

  // // Set last Layer
  // unsigned int ln = nLayers-1;
  // unsigned int lm = nLayers-2;

  // layers[i] = (
  //   addVecR(
  //     multMat(weights[lm],nNodes[ln],nNodes[lm], Acts[lm],nNodes[lm],1), 
  //     bias[lm]
  //   )
  // );
  // Acts.push_back(softMax(layers[ln]));
  
  return;
}

// Backprop
void runBackprop(
  // unsigned int nLayers,
  // std::vector<unsigned int> nNodes,
  std::vector<std::vector<double>> weights,
  std::vector<std::vector<double>> bias, 
  std::vector<std::vector<double>> layers, 
  std::vector<std::vector<double>> Acts,
  std::vector<double> obs,
  std::vector<std::vector<double>> &dWeights,
  std::vector<std::vector<double>> &dBias,
  lossFunction dLossFun,
  activationFunction ** dActFun,
  int * actType
){
  unsigned int llp = layers.size()-1; // Last Layer Position
  unsigned int lap = llp-1; // Last activation/weight matrix/bias vector position
  // unsigned int ln = nLayers-1; // number of layers from zero
  // unsigned int lm = nLayers-2; // number of weights, bias, and activation functions from zero

  /////////////////////////////////////////////////////////
  // Getting the loss and derivative of the final layer
  /////////////////////////////////////////////////////////
  // Creating initial derivative vector
  std::vector<double> dVect(layers[llp].size(),0.0);
  // If the final activation function is applied to a specific node
  if(actType[lap] == 1){
    // for each node
    for(unsigned int i = 0; i < layers[llp].size(); i++){
      // get the derivative for this node
      std::vector<double> tempVal{layers[llp][i]};
      std::vector<double> temp = (*dActFun[lap][i])(tempVal,1,0);
      dVect[i] = temp[0];
    }
  }
  // If the final activation function is applied to a layer
  if(actType[lap] == 2){
    // for each observation class
    for(unsigned int i = 0; i < obs.size(); i++){
      // check if the current class is the observed class
      if(obs[i] == 1){
        // get the layer derivative w.r.t the observed class
        dVect = (*dActFun[lap][0])(layers[llp], layers.size(), i);
        break;
      }
    }
  }

  // Multiply the derivative of the loss function
  multVec(dVect, (*dLossFun)(Acts[llp], obs));
  addVec(dBias[lap], dVect);
  addVec(dWeights[lap], get_dW(dBias[lap], Acts[lap]));

  /////////////////////////////////////////////////////////
  // Getting change for weigths and bias
  /////////////////////////////////////////////////////////
  for(unsigned int i = lap-1; i == 0; i--){
    std::vector<double> temp_dA(layers[i+1].size());
    if(actType[i] == 1){
      for(unsigned int j = 0; j < layers[i+1].size(); j++){
        std::vector<double> tempVal{layers[i+1][j]};
        std::vector<double> temp = (*dActFun[i][j])(tempVal,1, 0);
        temp_dA[j] = temp[0];
      }
    }
    if(actType[i] == 2){
      temp_dA = (*dActFun[i][0])(layers[i+1], layers.size(), 0);
    }
    addVec(dBias[i],get_inner_dB(dBias[i+1],weights[i+1],temp_dA));
    addVec(dWeights[i], get_dW(dBias[i], Acts[i]));
  }
}


///////////////////////////////////////////////////////////////////////////////
/// Training Functions
///////////////////////////////////////////////////////////////////////////////
void getTrainingSet(
  std::vector<std::vector<double>> samples, 
  std::vector<std::vector<double>> &train, 
  std::vector<std::vector<double>> &test,
  double ratio // Ratio of training to test set, default 70% training - 30% test
){
  unsigned int val0 = rand() % samples.size();
  std::vector<unsigned int> randInx{val0};
  for(unsigned int i = 1; i < int(samples.size()*(ratio/100)); i++){
    unsigned int pVal = rand() % samples.size();
    while(inVec(randInx, pVal)){pVal = rand() % samples.size();}
    randInx.push_back(pVal);
  }

  for(unsigned int i = 0; i < samples.size(); i++){
    if(inVec(randInx, i)){
      train.push_back(samples[i]);
    }else{
      test.push_back(samples[i]);
    }
  }
  return;
}
void getTrainingSet(
  std::vector<std::vector<double>> samples,
  std::vector<std::vector<double>> obs, 
  std::vector<std::vector<double>> &train, 
  std::vector<std::vector<double>> &obsTrain,
  std::vector<std::vector<double>> &test,
  std::vector<std::vector<double>> &obsTest, 
  double ratio // Ratio of training to test set, default 70% training - 30% test
){
  unsigned int val0 = rand() % samples.size();
  std::vector<unsigned int> randInx{val0};
  for(unsigned int i = 1; i < int(samples.size()*(ratio/100)); i++){
    unsigned int pVal = rand() % samples.size();
    while(inVec(randInx, pVal)){pVal = rand() % samples.size();}
    randInx.push_back(pVal);
  }

  for(unsigned int i = 0; i < samples.size(); i++){
    if(inVec(randInx, i)){
      train.push_back(samples[i]);
      obsTrain.push_back(obs[i]);
    }else{
      test.push_back(samples[i]);
      obsTest.push_back(obs[i]);
    }
  }
  return;
}

double trainSNN(
  std::vector<std::vector<double>> sample,
  std::vector<std::vector<double>> obs,
  std::vector<unsigned int> nNodes,
  std::vector<std::vector<double>> &weights,
  std::vector<std::vector<double>> &bias,
  activationFunction ** actFun,
  activationFunction ** dActFun,
  int * actType,
  lossFunction lossFun,
  unsigned int stop,
  bool Adam,
  std::vector<double> abbe,
  double gamma // convergance threshold
){
  double trainErr;
  std::vector<double> tempTrainErr;
  // Get number of layers
  unsigned int nLayers = nNodes.size();
  unsigned int epoch = 0; // Timestep
  bool converged = false; // converged
  if(weights.empty()){
    weights = initWeights(nLayers, nNodes, true);
    bias = initBias(nLayers, nNodes);
  }

  std::vector<std::vector<double>> mtW;
  std::vector<std::vector<double>> mtB;
  std::vector<std::vector<double>> vtW;
  std::vector<std::vector<double>> vtB;

  if(Adam){
    // print("\nAdam", abbe);
    mtW = initZero(weights); // 1st Moment std::vector
    mtB = initZero(bias); // 1st Moment std::vector
    vtW = initZero(weights); // 2nd Moment std::vector
    vtB = initZero(bias); // 2nd Moment std::vector
  }else{
    std::cout << "\nalpha: " << abbe[0] << std::endl;
  }

  // While not converged do:
  while(epoch < stop && !converged){
    epoch++;
    // init deltas
    std::vector<std::vector<double>> dWeights = initZero(weights);
    std::vector<std::vector<double>> dBias = initZero(bias);

    // Get Error for plotting (for each sample)
    std::vector<std::vector<double>> tempPVal;
    std::vector<bool> tempNCorrect;
    
    // For each training sample
    for(unsigned int s = 0; s < sample.size(); s++){
      // Init Layers and Activations
      std::vector<std::vector<double>> layers(nLayers);
      std::vector<std::vector<double>> Acts(nLayers);
      layers[0] = sample[s];
      Acts[0] = sample[s];
      // runForward
      runForward(
        // sample[s], 
        // nLayers, 
        nNodes, 
        weights, 
        bias, 
        layers, 
        Acts, 
        actFun, 
        actType
      );
      
      // Get Error for plotting (for each sample)
      tempPVal.push_back(argMax(layers[layers.size()-1],layers[layers.size()-1].size(),0));

      // init sample deltas
      std::vector<std::vector<double>> sdWeights = initZero(weights);
      std::vector<std::vector<double>> sdBias = initZero(bias);
      // runBackprop, save sums of delta
      runBackprop(
        // nLayers,
        // nNodes,
        weights,
        bias, 
        layers, 
        Acts,
        obs[s],
        sdWeights,
        sdBias,
        lossFun,
        dActFun,
        actType
      );

      addMat(dWeights, sdWeights);
      addMat(dBias, sdBias);
      
    }

    for(unsigned int i = 0; i < tempPVal.size(); i++){
      bool correct = match(tempPVal[i], obs[i]);
      tempNCorrect.push_back(correct);
    }

    // Get Error for plotting (for each epoch)
    trainErr = 1-double(sumVectR(tempNCorrect))/tempNCorrect.size();
    if(TRAINPRINT){tempTrainErr.push_back(trainErr);}

    // update weights and bias
    if(Adam){
      double alphat = abbe[0]*sqrt(1-pow(abbe[2],epoch))/(1-pow(abbe[1],epoch));

      // Weights
      multScal(mtW, abbe[1]);
      addMat(mtW, multScalR(dWeights, (1-abbe[1])));
      multScal(vtW, abbe[2]);
      addMat(vtW, multScalR(sqMatR(dWeights), (1-abbe[2])));
      subMat(
        weights, 
        ewDMR(
          multScalR(mtW, alphat),
          addScalR(sqrtMatR(vtW), abbe[3])
        )
      );

      // Bias
      multScal(mtB, abbe[1]);
      addMat(mtB, multScalR(dBias, (1-abbe[1])));
      multScal(vtB, abbe[2]);
      addMat(vtB, multScalR(sqMatR(dBias), (1-abbe[2])));
      subMat(
        bias,
        ewDMR(
          multScalR(mtB, alphat),
          addScalR(sqrtMatR(vtB), abbe[3])
        )
      );

    }else{
      subMat(weights, multScalR(dWeights, abbe[0]));
      subMat(bias, multScalR(dBias, abbe[0]));
    }

    // Get average change in delta
    double sumdW = AvgAbsSum(dWeights);
    double sumdB = AvgAbsSum(dBias);




    if(sumdW < gamma && sumdB < gamma){
      converged = true;
      std::cout << "Epoch: " << epoch << std::endl;
      std::cout << "Converged" << std::endl;
    }

    
  }
  if(!converged){std::cout << "Epoch: " << epoch << std::endl;}

  if(ERRPRINT){
    fprintf(stderr, "%u, ", epoch);
  }
  if(TRAINPRINT){writeLineTo("TrainingError.txt",tempTrainErr);}
  
  return trainErr;
}

void testSNN(
  std::vector<std::vector<double>> samples,
  std::vector<std::vector<double>> obs,
  std::vector<unsigned int> nNodes,
  std::vector<std::vector<double>> weights,
  std::vector<std::vector<double>> bias,
  activationFunction ** actFun,
  int * actType,
  std::vector<std::vector<double>> &outVal,
  std::vector<bool> &results
){
  unsigned int nLayers = nNodes.size();
  outVal.resize(samples.size());

  // #pragma omp parallel for
  for(unsigned int i = 0; i < samples.size(); i++){
    // Set first Layer
    std::vector<std::vector<double>> layers;
    layers.resize(nLayers);
    layers[0] = samples[i];
    
    std::vector<std::vector<double>> Acts;
    Acts.resize(nLayers);
    Acts[0] = samples[i];

    runForward(
      nNodes, 
      weights, 
      bias, 
      layers, 
      Acts, 
      actFun, 
      actType
    );

    // // For each hidden layer
    // for(unsigned int j = 1; j < nLayers; j++){
    //   unsigned int l = j-1;
    //   layers[j] = (
    //     addVecR(
    //       multMat(weights[l],nNodes[j],nNodes[l], Acts[l],nNodes[l],1), 
    //       bias[l]
    //     )
    //   );

    //   //uses a single node at a time
    //   if (actType[j] == 1){
    //     Acts[j].resize(layers[j].size());
    //     // for each node
    //     for(unsigned int k = 0; k < layers[j].size(); k++){
    //       std::vector<double> tempVector = (*actFun[j][k])(&layers[j][k], 1);
    //       Acts[j][k] = tempVector[0];
    //     }
    //   }

    //   // uses the whole layer at once and returns an array
    //   if (actType[j] == 2){
    //     Acts[j] = (*actFun[j][0])(&layers[j], layers[j].size())
    //   }
      
    // }

    // // Set last Layer
    // unsigned int ln = nLayers-1;
    // unsigned int lm = nLayers-2;

    // layers.push_back(
    //   addVecR(
    //     multMat(weights[lm],nNodes[ln],nNodes[lm], Acts[lm],nNodes[lm],1), 
    //     bias[lm]
    //   )
    // );


    // Acts.push_back(argMax(layers[ln]));
    outVal[i] = Acts[nLayers-1];
  }

  for(unsigned int i = 0; i < outVal.size(); i++){
    bool correct = match(outVal[i], obs[i]);
    results.push_back(correct);
  }

  return;
}