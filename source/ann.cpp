#include "include/ann.h"

///////////////////////////////////////////////////////////////////////////////
/// ANN Utility
///////////////////////////////////////////////////////////////////////////////
void print(struct Ann ann){
  std::cout << "Number of Features: " << ann.nNodes[0] << ", ";
  std::cout << "Number of Classes: " << ann.nNodes[ann.nLayers-1] << ", ";
  std::cout << "Number of Layers: " << ann.nLayers << ", ";
  std::cout << "Total Number of Nodes: " << ann.tNodes << ", ";
  std::cout << "Number of Nodes per Layer: ";
  for(unsigned int i = 0; i < ann.nNodes.size(); i++){
    std::cout << ann.nNodes[i] << ", ";
  }
  std::cout << std::endl;
  BUG(
    std::cout << "Starting Position of Each Layer (Summed Nodes): ";
    for(unsigned int i = 0; i < ann.sNodes.size(); i++){
      std::cout << ann.sNodes[i] << ", ";
    }
    std::cout << std::endl;
  )
  std::cout << "List of Activation IDs: ";
  unsigned int p = 0;
  for(unsigned int i = 1; i < ann.nNodes.size(); i++){
    std::cout << "L" << i << "("<< ann.nNodes[i] <<"): ";
    for(unsigned int j = 0; j < ann.nNodes[i]; j++){
      std::cout << ann.actIDs[p] << ", ";  
      p++;
    }
  }
  std::cout << std::endl;
  std::cout << "Loss ID: " << ann.lLAID << std::endl;
  std::cout << "Weights("<< getSize(ann.weights) << "): ";
  for(unsigned int i = 0; i < ann.weights.size(); i++){
    std::cout << "\n\tL" << i << "("<< ann.weights[i].size() <<") ";
    for(unsigned int j = 0; j < ann.weights[i].size(); j++){
      std::cout << ann.weights[i][j] << ", ";
    }
  }
  std::cout << std::endl;
  std::cout << "Bias("<< ann.bias.size() << "): ";
  for(unsigned int i = 0; i < ann.bias.size(); i++){
    std::cout << ann.bias[i] << ", ";
  }
  std::cout << std::endl << std::endl;
}

void print(std::vector<std::vector<DTYPE>> W, std::vector<DTYPE> B){
  std::cout << "Weights("<< getSize(W) << "): ";
  for(unsigned int i = 0; i < W.size(); i++){
    std::cout << "\n\tL" << i << "("<< W[i].size() <<") ";
    for(unsigned int j = 0; j < W[i].size(); j++){
      std::cout << W[i][j] << ", ";
    }
  }
  std::cout << std::endl;
  std::cout << "Bias("<< B.size() << "): ";
  for(unsigned int i = 0; i < B.size(); i++){
    std::cout << B[i] << ", ";
  }
  std::cout << std::endl << std::endl;
}

void printTo(struct Scores scores, std::string filename /*= "score.log"*/){
  bool addHeader = false;
  std::ifstream check(filename);
  if(!check || is_empty(check)){
    check.close();
    addHeader = true;
  }
  std::ofstream file;
  file.open(filename, std::ofstream::app);
  
  std::vector<std::string> headers{"p", "r", "f"};
  if(addHeader){
    file << "a";
    for(unsigned int i = 0; i < scores.F1.size(); i++){
      for(unsigned int j = 0; j < headers.size(); j++){
        file << ", " << headers[j] << i; 
      }
    }
  }

  file << "\n" << scores.accuracy;
  for(unsigned int i = 0; i < scores.F1.size(); i++){
    file 
      << ", " << scores.precision[i]
      << ", " << scores.recall[i]
      << ", " << scores.F1[i];
  }
  
  file.close();
}

///////////////////////////////////////////////////////////////////////////////
/// Applying Activation Functions
///////////////////////////////////////////////////////////////////////////////
void applyAct(
  std::vector<DTYPE> &layer, 
  std::vector<unsigned int> aIDs,
  unsigned int obs
){
  if(layer.size() != aIDs.size()){
    errPrint("ERROR applyAct: layer.size() != aIDs.size().", layer.size(), aIDs.size());
    return;
  }
  std::vector<DTYPE> temp = layer;
  if(same(aIDs)){
    layer = ACT2[aIDs[0]](temp, obs);
  }else{
    for(unsigned int i = 0; i < layer.size(); i++){
      if(aIDs[i] < TYPECHANGE){
        layer[i] = ACT1[aIDs[i]](temp[i]);
      }else{
        layer[i] = ACT2[aIDs[i]](temp,obs)[i];
      }
    }
  }
  return;
} 

void applyDAct(
  std::vector<DTYPE> &layer, 
  std::vector<unsigned int> aIDs,
  unsigned int obs
){
  if(layer.size() != aIDs.size()){
    errPrint("ERROR applyAct: layer.size() != aIDs.size().", layer.size(), aIDs.size());
    return;
  }
  std::vector<DTYPE> temp = layer;
  if(same(aIDs)){
    layer = DACT2[aIDs[0]](temp, obs);
  }else{
    for(unsigned int i = 0; i < layer.size(); i++){
      if(aIDs[i] < TYPECHANGE){
        layer[i] = DACT1[aIDs[i]](temp[i]);
      }else{
        layer[i] = DACT2[aIDs[i]](temp,obs)[i];
      }
    }
  }
  return;
} 

std::vector<DTYPE> applyDActR(
  std::vector<DTYPE> &layer, 
  std::vector<unsigned int> aIDs,
  unsigned int obs
){
  if(layer.size() != aIDs.size()){
    errPrint("ERROR applyAct: layer.size() != aIDs.size().", layer.size(), aIDs.size());
    return layer;
  }
  std::vector<DTYPE> temp(layer.size(),0);
  if(same(aIDs)){
    temp = DACT2[aIDs[0]](layer, obs);
  }else{
    for(unsigned int i = 0; i < layer.size(); i++){
      if(aIDs[i] < TYPECHANGE){
        temp[i] = DACT1[aIDs[i]](layer[i]);
      }else{
        temp[i] = DACT2[aIDs[i]](layer,obs)[i];
      }
    }
  }
  return temp;
} 

///////////////////////////////////////////////////////////////////////////////
/// Initializers
///////////////////////////////////////////////////////////////////////////////
void initWeights(
  std::vector<std::vector<DTYPE>> &weights,
	std::vector<unsigned int> nNodes
){
  for(unsigned int i = 0; i < nNodes.size() - 1; i++){
    unsigned int j = nNodes[i] * nNodes[i+1];
    weights.push_back(rng(j));
  }
  return;
}
void initWeights(struct Ann &ann){
  // Unpack
  std::vector<unsigned int> nNodes = ann.nNodes;
  // Getting weights
  std::vector<std::vector<DTYPE>> weights;
  initWeights(weights, nNodes);
  // Pack
  ann.weights = weights;
  return;
}

// Ann initANN(
// 	unsigned int nFeat, 
// 	unsigned int nClasses, 
// 	unsigned int nLayers
// ){
//   // Getting Default Number of Nodes Per Layer
//   std::vector<unsigned int> nNodes(nLayers, nClasses);
//   nNodes[0] = nFeat;
//   // Getting Layer Starting nodes
//   std::vector<unsigned int> sNodes(nLayers,0);
//   for(unsigned int i = 1; i < nLayers; i++){
//     sNodes[i] = sNodes[i-1]+nNodes[i-1];
//   }
//   // Getting Number of total Nodes
//   unsigned int tNodes = sum(nNodes);
//   // Getting Default Activation List 
//   std::vector<unsigned int> actIDs(tNodes, RELU);
//   for(unsigned int i = tNodes-1; i > tNodes-nNodes[nNodes.size()-1]-1; i--){
//     // print(i, "[i], 3 input");
//     actIDs[i] = SOFTMAX; 
//   }
//   // Initializing Weights
//   std::vector<std::vector<DTYPE>> weights;
//   initWeights(weights, nNodes);
//   // Initializing Bias
//   std::vector<DTYPE> bias(tNodes-nFeat, 0);
//   // Packing
//   struct Ann ann;
//   ann.nLayers = nLayers;
//   ann.nNodes = nNodes;
//   ann.sNodes = sNodes;
//   ann.tNodes = tNodes;
//   ann.actIDs = actIDs;
//   ann.lossID = 0;
//   ann.weights = weights;
//   ann.bias = bias;
  
//   return ann;
// }

Ann initANN(
	unsigned int nFeat, 
	unsigned int nClasses, 
	unsigned int nLayers,
	std::vector<unsigned int> nNodes
){
  struct Ann ann; 
  if(nNodes.size() != nLayers){
    errPrint("ERROR - initANN: nNodes size does not match nLayers.");
    std::cout << nNodes.size() << ":" << nLayers << std::endl;
    exit(1);
    return ann;
  }

  // Getting Layer Starting nodes
  std::vector<unsigned int> sNodes(nLayers,0);
  for(unsigned int i = 1; i < nLayers; i++){
    sNodes[i] = sNodes[i-1]+nNodes[i-1];
  }
  // print(sNodes, "sNodes");
  // Getting Number of total Nodes
  unsigned int tNodes = sum(nNodes);
  // print(tNodes-nFeat, "Total number of activations");

  // Getting Default Activation List 
  std::vector<unsigned int> actIDs(tNodes-nFeat, RELU);
  // Setting last layer (must be SoftMax)
  // print(sNodes[nLayers-1], "Start"); 
  int tempsize = actIDs.size();
  // print(tempsize, "Number of activations");
  for(unsigned int i = sNodes[nLayers-1]-nFeat; i < actIDs.size(); i++){
    // print(i, "[i], 4 input");
    actIDs[i] = SOFTMAX; 
  }
  // Initializing Weights
  std::vector<std::vector<DTYPE>> weights;
  initWeights(weights, nNodes);
  // Initializing Bias
  std::vector<DTYPE> bias(tNodes-nFeat, 0);
 
 // Packing
  ann.nLayers = nLayers;
  ann.nNodes = nNodes;
  ann.sNodes = sNodes;
  ann.tNodes = tNodes;
  ann.actIDs = actIDs;
  ann.lossID = 0;
  ann.weights = weights;
  ann.bias = bias;
  
  return ann;
}

Ann initANN(struct ANN_Ambit ann_, struct Data train){
  unsigned int nFeat = train.nFeat;
  unsigned int nClasses = train.nClasses;
  unsigned int nLayers = ann_.nLayers;
  
  if(ann_.hNodes.size() != nLayers-2){
    errPrint(
      "ERROR - initANN: hNodes size does not match number of hidden layers."
    );
    std::cout << ann_.hNodes.size() << ":" << nLayers-2 << std::endl;
    exit(1);
  }
  std::vector<unsigned int> nNodes(nLayers, 0);
  nNodes[0] = nFeat;
  for(unsigned int i = 0; i < ann_.hNodes.size(); i++){
    nNodes[i+1] = ann_.hNodes[i];
  }
  nNodes[nLayers-1] = nClasses;

  if(nNodes.size() != nLayers){
    errPrint(
      "ERROR - initANN(ANN_Ambit, Data): nNodes.size does not match the number of layers"
    );
    std::cout << nNodes.size() << ":" << nLayers << std::endl;
    exit(1);
  }

  srand(ann_.wseed);
  
  struct Ann ann = initANN(
    nFeat,
    nClasses,
    nLayers,
    nNodes
  );

  // Need list of node positions and what activation ID to change to
  for(unsigned int i = 0; i < ann_.ActIDSets.size(); i++){
    unsigned int ID = ann_.ActIDSets[i].ID;
    for(unsigned int j = 0; j < ann_.ActIDSets[i].nodePositions.size(); j++){
      unsigned int pos = ann_.ActIDSets[i].nodePositions[j];
      ann.actIDs[pos] = ID;
    }
  }

  // for(unsigned int i = 0; i < ann_.ActIDSets.size(); i++){
    // setActID(
    //   ann,
    //   ann_.ActIDSets[i].ID,
    //   ann_.ActIDSets[i].layerStrt,
    //   ann_.ActIDSets[i].layerEnd,
    //   ann_.ActIDSets[i].nodeStrt,
    //   ann_.ActIDSets[i].nodeEnd
    // );
  // }
  return ann;
}

void getDataSets(
  struct Data &train, 
  struct Data &test, 
  struct Data data
){
  // Unpack
  unsigned int nFeat = data.nFeat;
  double ratio = data.ratio;
  unsigned int nSamp = data.nSamp;
  unsigned int sseed = data.sseed;

  // Get Training Set
  srand(sseed);
  unsigned int nSamp_Train = (ratio / 100)*nSamp;
  std::vector<unsigned int> trainSet = rng_unq(nSamp_Train, 0, nSamp);
  BUG(
    std::cout << "Training Set: " << std::endl;
    print(trainSet);
  )

  // Sort Features and Observations
  std::vector<DTYPE> train_feat;
  std::vector<unsigned int> train_obs;

  std::vector<DTYPE> test_feat;
  std::vector<unsigned int> test_obs;
  
  for(unsigned int i = 0; i < nSamp; i++){
    // if(inVec(i,trainSet)){
    if(std::find(trainSet.begin(), trainSet.end(), i) != trainSet.end()){
      for(unsigned int j = nFeat*i; j < nFeat*i+nFeat; j++){
        train_feat.push_back(data.feat[j]);
      }
      train_obs.push_back(data.obs[i]);
      if(train_feat.size()/nFeat != train_obs.size()){
        errPrint("ERROR - getDataSet: Features do not match observations.");
        std::cout << train_feat.size()/nFeat << ":" << train_obs.size() << std::endl;
      }
    }else{
      for(unsigned int j = nFeat*i; j < nFeat*i+nFeat; j++){
        test_feat.push_back(data.feat[j]);
      }
      test_obs.push_back(data.obs[i]);
      if(test_feat.size()/nFeat != test_obs.size()){
        errPrint("ERROR - getDataSet: Features do not match observations.");
        std::cout << test_feat.size()/nFeat << ":" << test_obs.size() << std::endl;
      }
    }
  }

  // Pack
  train.nFeat = nFeat;
  train.nClasses = data.nClasses;
  train.nSamp = nSamp_Train;
  train.sseed = sseed;
  train.ratio = ratio;
  train.feat = train_feat;
  train.obs = train_obs;

  test.nFeat = nFeat;
  test.nClasses = data.nClasses;
  test.nSamp = nSamp - nSamp_Train;
  test.sseed = sseed;
  test.ratio = 100 - ratio;
  test.feat = test_feat;
  test.obs = test_obs;
};

///////////////////////////////////////////////////////////////////////////////
/// Setters
///////////////////////////////////////////////////////////////////////////////
// Get Activation ID Postion
unsigned int getAIDP(
  std::vector<unsigned int> nNodes, 
  unsigned int layerN, 
  unsigned int nodeN /*0*/
){
  // std::vector<unsigned int> pre_nNodes = {nNodes.begin(), nNodes.begin()+layerN};
  std::vector<unsigned int> pre_nNodes = subVector(nNodes, 0, layerN);
  unsigned int position = sum(pre_nNodes)+nodeN;
  return position;
}

void setActID(
  std::vector<unsigned int> &actIDs,
  std::vector<unsigned int> nNodes,
  unsigned int ID,
  unsigned int layerN,
  unsigned int nodeN
){
  // Get position
  unsigned int position = getAIDP(nNodes, layerN, nodeN);
  // Set actIDs
  actIDs[position] = ID;
  return;
}

void setActID(
  struct Ann &ann,
  unsigned int ID,
  unsigned int layerNStrt,
  unsigned int layerNEnd /*UINT_MAX*/,
  unsigned int nodeNStrt /*0*/,
  unsigned int nodeNEnd /*UINT_MAX*/
){
  // Unpack
  std::vector<unsigned int> actIDs = ann.actIDs;
  std::vector<unsigned int> nNodes = ann.nNodes;
  // Get Limits
  if(layerNEnd > nNodes.size()){layerNEnd = nNodes.size();}
  // Get new ActIDs
  for(unsigned int i = layerNStrt; i < layerNEnd; i++){
    if(nodeNEnd > nNodes[i]){nodeNEnd = nNodes[i];}
    for(unsigned int j = nodeNStrt; j < nodeNEnd; j++){
      setActID(actIDs, nNodes, ID, i, j);
    }
  }
  // Pack
  ann.actIDs = actIDs;
  
  return;
}

///////////////////////////////////////////////////////////////////////////////
/// Training and Testing
///////////////////////////////////////////////////////////////////////////////
void getResults(
  struct Results &result,
  std::vector<DTYPE> lastAct,
  std::vector<DTYPE> lastLayer,
  unsigned int lastActID,
  unsigned int lossID,
  unsigned int sampleIndex,
  unsigned int obs
){
  // Copy result set to results struct
  result.observedValue[sampleIndex] = obs;
  // Get Sample Starting Index (when listed as a layer output)
  unsigned int cli = sampleIndex*lastAct.size();
  // Set Sample Prediction to the last layer output
  set(result.vector_dtype, lastAct, cli);
  // Get the class prediction
  unsigned int prediction = (unsigned int)max(lastAct, true);
  // Set class prediction to in sample index
  result.vector_uint[sampleIndex] = prediction;
  if(prediction == obs){
    // Count the number of correct predictions
    result.uint_ambit ++;
    // Set sample index to true
    result.vector_bool[sampleIndex] = true;
  }
  result.observedValue[sampleIndex] = obs;
  std::vector<DTYPE> temp = ACT2[lastActID](lastLayer, 0);
  result.double_ambit += sum(LOSSF[lossID](temp,obs));
}

void getScores(
  struct Scores &score,
  std::vector<unsigned int> obs,
  std::vector<unsigned int> pre,
  std::vector<bool> cor
){
  std::vector<unsigned int> obsCnt(3,0);
  std::vector<unsigned int> preCnt(3,0);
  std::vector<unsigned int> corCnt(3,0);
  
  for(unsigned int i = 0; i < obs.size(); i++){
    obsCnt[obs[i]]++;
    preCnt[pre[i]]++;
    if(cor[i]){
      corCnt[obs[i]]++;
    }
    BUG(
      std::cout << "Sample: " << i
        << "  Observed: " << obs[i]
        << "  Predicted: " << pre[i]
        << "  Correct: " << cor[i]
        << "  New Observed " << obs[i] << " Count: " << obsCnt[obs[i]]
        << "  New Predicted " << pre[i] << " Count: " << preCnt[pre[i]]
        << "  New Correct " << obs[i] << " Count: " << corCnt[obs[i]]
      << std::endl;
    )
  }

  score.accuracy = (double)sum(corCnt)/(double)cor.size();
  
  for(unsigned int i = 0; i < 3; i++){
    if(obsCnt[i] == 0){
      score.recall[i] = -1;
    }else{
      score.recall[i] = (double)corCnt[i]/(double)obsCnt[i];
    }
    if(preCnt[i] == 0){
      score.precision[i] = -1;
    }else{
      score.precision[i] = (double)corCnt[i]/(double)preCnt[i];    
    }
    if(obsCnt[i] == 0 || preCnt[i] == 0){
      score.F1[i] = -1;
    }else{
      score.F1[i] = (2*score.recall[i]*score.precision[i])/(score.recall[i]+score.precision[i]);
    }
    BUG(
      std::cout << "Class: " << i
        << "  Observed: " << obsCnt[i]
        << "  Predicted: " << preCnt[i]
        << "  Correct: " << corCnt[i]
        << "  = Recall: " << score.recall[i]
        << "  Precision: " << score.precision[i]
        << "  F1: " << score.F1[i]
      << std::endl;
    )
  }
  BUG(
    std::cout << "Accuracy: " << score.accuracy << std::endl;
  )
  return;
}
struct Scores getScores(
  struct Results result,
  unsigned int nClasses
){
  std::vector<unsigned int> obs = result.observedValue;
  std::vector<unsigned int> pre = result.vector_uint;
  std::vector<bool> cor = result.vector_bool;
  
  struct Scores score(nClasses);
  getScores(
    score,
    obs,
    pre,
    cor
  );

  return score;
}

void forward(
  struct Ann ann,
  std::vector<DTYPE> &layer,
  std::vector<DTYPE> &act
){
  // unpack
  unsigned int nLayers = ann.nLayers;
  BUG(print(nLayers, "nLayers");)
  // For each layer after the feature layer/vector
  /*bias layer index*/ unsigned int bli = 0; // add the layer size at end of for loop
  for(unsigned int i = 1; i < nLayers; i++){
    BUG(print(i, "\nLayer"));
    /*activation layer index*/ unsigned int ali = ann.sNodes[i-1];
    /*activation layer size*/  unsigned int als = ann.nNodes[i-1];
    /*bias layer size*/ unsigned int bls = ann.nNodes[i];
    BUG(
      print(ali, "activation layer index");
      print(als, "activation layer size");
      print(bli, "bias layer index");
      print(bls, "bias layer size");
    )
    // get w, a_i-1, and b
    std::vector<DTYPE> w = ann.weights[i-1];
    std::vector<DTYPE> a = subVector(act, ali, als);
    std::vector<DTYPE> b = subVector(ann.bias, bli, bls);
    BUG(
      print(w, "weights");
      print(a, "activation");
      print(b, "bias");
    )
    // matrix multiply w*a
    std::vector<DTYPE> wab = dot(w, a, als);
    BUG(print(wab, "w*a"));
    // add b
    add(wab, b);
    BUG(print(wab, "w*a+b"));
    // set to layer position
    set(layer, wab, ali+als);
    BUG(print(layer, "layer-Update"));
    // Get activation IDs
    std::vector<unsigned int> aIDs = subVector(ann.actIDs, bli, bls);
    BUG(print(aIDs, "aID"));
    // apply activation function
    applyAct(wab, aIDs, 0);
    BUG(print(wab, "a1(w*a0+b)"));    
    // set to act position
    set(act, wab, ali+als);
    BUG(print(act, "act-Update"));
    // Update bli
    bli += bls;
  }
  return;
}

void backProp(
  struct Ann ann,
  unsigned int obs,
  std::vector<DTYPE> layer,
  std::vector<DTYPE> act,
  std::vector<std::vector<DTYPE>> &dW,
  std::vector<DTYPE> &dB
){
  // Get dLoss
  unsigned int lID = ann.lossID; // Loss ID
  unsigned int ll = ann.nLayers-1; // Last Layer
  unsigned int strtA = ann.sNodes[ll]; // Start of Layer
  unsigned int size = ann.nNodes[ll]; // Size of Layer;
  std::vector<DTYPE> a = subVector(act, strtA, size); // current activation vector
  std::vector<DTYPE> dLoss = DLOSSF[lID](a, obs);

  // Get Initial Delta
  unsigned int nFeat = ann.nNodes[0];
  unsigned int strtB = ann.sNodes[ll]-nFeat;
  std::vector<unsigned int> aIDs = subVector(ann.actIDs, strtB, size);
  std::vector<DTYPE> l = subVector(layer, strtA, size);
  std::vector<DTYPE> delta = dot(dLoss, applyDActR(l, aIDs, obs));
  
  // Set Last Layer dB and dW
  add(dB, delta, strtB, size);
  strtA = ann.sNodes[ll-1]; size = ann.nNodes[ll-1];
  a = subVector(act, strtA, size);
  std::vector<DTYPE> tensorDA = tensor(delta, a);
  add(dW[ll-1], tensorDA, 0, tensorDA.size());

  // For each layer (backwards)
  for(unsigned int i = ll-1; i > 0; i--){
    strtA = ann.sNodes[i]; size = ann.nNodes[i];
    strtB = ann.sNodes[i]-nFeat;

    std::vector<DTYPE> w = ann.weights[i];
    unsigned int stride = w.size()/size;
    
    l = subVector(layer, strtA, size);
    aIDs = subVector(ann.actIDs, strtB, size);
    std::vector<DTYPE> tempDelta = delta;
    delta = dot(dotT(w, tempDelta, stride),applyDActR(l, aIDs, obs));

    add(dB, delta, strtB, size);
    strtA = ann.sNodes[i-1]; size = ann.nNodes[i-1];
    a = subVector(act, strtA, size);
    tensorDA = tensor(delta, a);
    add(dW[i-1], tensorDA, 0, tensorDA.size());
  }
  BUG(
    std::cout << "Updated dW and dB." << std::endl;
    print(dW, dB);
  )

  return;
}

void trainNN(
  struct Ann &ann,
  struct Data data,
  struct Results &result, 
  struct Alpha alpha,
  unsigned int maxIter /*1000*/ 
){
  // Unpack
  unsigned int nSamp = data.nSamp;
  unsigned int nFeat = data.nFeat;
  unsigned int nClasses = data.nClasses;

  // Training Iteration Epoch
  unsigned int epoch = 0;
  bool converged = false;
  std::vector<unsigned int> wsize = getSizeVec(ann.weights);

  // For Learning Rate
  std::vector<std::vector<DTYPE>> mtW;
  std::vector<DTYPE> mtB;
  std::vector<std::vector<DTYPE>> vtW;
  std::vector<DTYPE> vtB;
  if(alpha.adam){
    print(alpha.adam, "Adam", false);
    print(alpha.alpha, "Alpha", false);
    print(alpha.beta1, "beta1", false);
    print(alpha.beta2, "beta2");
    
    mtW = zero(wsize); //1st Moment std::vector
    mtB = std::vector<DTYPE> (ann.bias.size(), 0); // 1st Moment std::vector
    vtW = zero(wsize); // 2nd Moment std::vector
    vtB = std::vector<DTYPE> (ann.bias.size(), 0); // 2nd Moment std::vector
  }else{
    print(alpha.alpha, "alpha");
  }

  while(epoch < maxIter && !converged){
    epoch++;
    // Init deltas
    std::vector<std::vector<DTYPE>> dW = zero(wsize);
    std::vector<DTYPE> dB(ann.bias.size(), 0);
    result.double_ambit = 0;
    result.uint_ambit = 0;
    result.vector_bool = std::vector<bool>(nSamp, false);
    result.vector_uint = std::vector<unsigned int>(nSamp, nClasses);
    result.vector_dtype = std::vector<DTYPE>(nSamp*nClasses, 0);

    // For each sample
    // Parallelize
    for(unsigned int i = 0; i < nSamp; i++){
      unsigned int featStride = i*nFeat;
      unsigned int obsStride = i*nClasses;

      std::vector<DTYPE> layer(ann.tNodes, 0);
      std::vector<DTYPE> act(ann.tNodes, 0);
      for(unsigned int j = 0; j < ann.nNodes[0]; j++){
        layer[j] = data.feat[featStride+j];
        act[j] = data.feat[featStride+j];
      }
      
      // Run forward propagation
      forward(
        ann,
        layer,
        act
      );

      unsigned int lli = ann.sNodes[ann.nLayers-1];
      unsigned int lls = ann.nNodes[ann.nLayers-1];
      unsigned int lastActID = ann.actIDs[ann.actIDs.size()-1];
      std::vector<DTYPE> lastAct = subVector(act,lli,lls);
      std::vector<DTYPE> lastLayer = subVector(act,lli,lls);
      getResults(result, lastAct, lastLayer, lastActID, ann.lossID, i, data.obs[i]);

      // Run Back Propagation
      backProp(
        ann,
        data.obs[i],
        layer,
        act,
        dW,
        dB
      );
    }

    // update weights and bias
    if(alpha.adam){
      double alpha_ = alpha.alpha*sqrt(1-pow(alpha.beta2,epoch))/(1-pow(alpha.beta1,epoch));

      // Weights
      dot(mtW, alpha.beta1);
      add(mtW, dotR(dW, (1-alpha.beta1)));
      dot(vtW, alpha.beta2);
      add(vtW, dotR(squareR(dW), (1-alpha.beta2)));
      subtract(
        ann.weights, 
        divideR(
          dotR(mtW, alpha_),
          addR(rootR(vtW), alpha.epsilon)
        )
      );

      // Bias
      dot(mtB, alpha.beta1);
      add(mtB, dotR(dB, (1-alpha.beta1)));
      dot(vtB, alpha.beta2);
      add(vtB, dotR(squareR(dB), (1-alpha.beta2)));
      subtract(
        ann.bias,
        divideR(
          dotR(mtB, alpha_),
          addR(rootR(vtB), alpha.epsilon)
        )
      );

    }else{
      subtract(ann.weights, dotR(dW, alpha.alpha));
      subtract(ann.bias, dotR(dB, alpha.alpha));
    }

    if((result.double_ambit < alpha.gamma) || (result.uint_ambit == nSamp)){
      converged = true;
      print(epoch, "Epoch"); print(converged, "Converged, In IF");
    }
  }

  if(!converged){
    print(epoch, "Epoch");
    fprintf(stderr, "%u, ",epoch);
  }
  return;
}


void testNN(
  struct Ann ann,
  struct Data data,
  struct Results &result
){
  BUG(std::cout << "\nTesting" << std::endl;)
  // unpack
  /*Number of Samples*/     unsigned int nSamp = data.nSamp;
  /*Total Number of Nodes*/ unsigned int tNodes = ann.tNodes;
  /*Number of Features*/    unsigned int nFeat = data.nFeat;
  /*Number of Layers*/      unsigned int nLayer = ann.nLayers;

  // Change the last Layer activation ID if necessary
  /*Last Layer Activation ID*/ unsigned int lLAID = ann.lLAID;
  /*Last Layer Index*/ unsigned int lli = ann.sNodes[nLayer-1];
  /*Last Layer Size*/ unsigned int lls = data.nClasses;
  /*Activation ID Last Layer Index*/ unsigned int alli = ann.sNodes[nLayer-1]-nFeat;
  /*Activation ID Last Layer*/ std::vector<unsigned int> aIDLL = subVector(ann.actIDs, alli, lls);
  BUG(print(aIDLL, "Activation ID Last Layer");)
  /*Original Last Layer Activation ID*/ unsigned int oAIDLL= aIDLL[0];
  bool changed = false;
  if(same(aIDLL) & (oAIDLL != lLAID)){
    BUG(std::cout << "Setting aIDLL to "<< lLAID << std::endl;)
    set(ann.actIDs, lLAID, alli, lls);
    changed = true;
  }

  // For each sample
  for(unsigned int i = 0; i < nSamp; i++){
    /*Feature layer index*/unsigned int fli = i*nFeat;
    /*Class Layer index*/ unsigned int cli = i*lls;
    
    // get features
    std::vector<DTYPE> feat = subVector(data.feat, fli, nFeat);

    // Create and Setup layer vector and activation vector
    std::vector<DTYPE> layer(tNodes, 0);
    set(layer, feat, 0, nFeat);
    std::vector<DTYPE> act(tNodes, 0);
    set(act, feat, 0, nFeat);

    // Run forward function
    forward(ann, layer, act);

    // Get Results
    std::vector<DTYPE> lastAct = subVector(act,lli,lls);
    std::vector<DTYPE> lastLayer = subVector(act,lli,lls);
    getResults(result, lastAct, lastLayer, oAIDLL, ann.lossID, i, data.obs[i]);
  }
  if(changed){
    BUG(std::cout << "Setting aIDLL to back to "<< oAIDLL << std::endl;)
    set(ann.actIDs, oAIDLL, alli, lls);
  }
  return;
}

void runANN(
  struct Alpha alpha,
  struct ANN_Ambit ann_,
  struct Data data,
  struct Data train,
  struct Data test,
  struct Scores &trainScores,
  struct Scores &testScores
){
  // Build
  struct Ann ann = initANN(ann_, train);
  BUG(std::cout << "\nGetting ANN" << std::endl;);
  print(ann);

  // Train
  struct Results train_results(train.nSamp, train.nClasses);
  trainNN(ann, train, train_results, alpha, ann_.maxIter);
  
  std::cout << "\nUpdated Weights and Bias" << std::endl;
  print(ann.weights, ann.bias);
  std::cout << "\nTraining Results" << std::endl;
  print(train_results);

  trainScores = getScores(train_results, train.nClasses);
  
  // Test
  struct Results test_results(test.nSamp, test.nClasses);
  testNN(ann, test, test_results);
  
  std::cout << "\nTesting Results" << std::endl;
  print(test_results);

  testScores = getScores(test_results, test.nClasses);
}