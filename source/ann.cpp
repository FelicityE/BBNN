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
    errPrint("ERROR applyDAct: layer.size() != aIDs.size().", layer.size(), aIDs.size());
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
    errPrint("ERROR applyDActR: layer.size() != aIDs.size().", layer.size(), aIDs.size());
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

Ann initANN(struct ANN_Ambit &annbit, struct Data train){
  unsigned int nFeat = train.nFeat;
  unsigned int nClasses = train.nClasses;
  unsigned int nLayers = annbit.nLayers;
  
  // Checking that inputs match
  if(annbit.hNodes.size() != nLayers-2){
    errPrint(
      "ERROR - initANN: hNodes size does not match number of hidden layers."
    );
    std::cout << annbit.hNodes.size() << ":" << nLayers-2 << std::endl;
    exit(1);
  }

  // Getting the Number of Nodes Per Layer
  std::vector<unsigned int> nNodes(nLayers, 0);
  nNodes[0] = nFeat;
  for(unsigned int i = 0; i < annbit.hNodes.size(); i++){
    nNodes[i+1] = annbit.hNodes[i];
  }
  nNodes[nLayers-1] = nClasses;
  // Checking Nodes and Layers match
  if(nNodes.size() != nLayers){
    errPrint(
      "ERROR - initANN(ANN_Ambit, Data): nNodes.size does not match the number of layers"
    );
    std::cout << nNodes.size() << ":" << nLayers << std::endl;
    exit(1);
  }

  // Creating ANN struct
  struct Ann ann;

  // Getting Layer Starting nodes
  std::vector<unsigned int> sNodes(nLayers,0);
  for(unsigned int i = 1; i < nLayers; i++){
    sNodes[i] = sNodes[i-1]+nNodes[i-1];
  }

  // Getting Number of total Nodes
  unsigned int tNodes = sum(nNodes);

  // Getting Default Activation List 
  std::vector<unsigned int> actIDs(tNodes-nFeat, annbit.actDefault);
  int tempsize = actIDs.size();
  for(unsigned int i = sNodes[nLayers-1]-nFeat; i < actIDs.size(); i++){
    actIDs[i] = SOFTMAX; 
  }

  // Initializing Weights
  srand(annbit.wseed);
  std::vector<std::vector<DTYPE>> weights;
  initWeights(weights, nNodes);

  // Initializing Bias
  std::vector<DTYPE> bias(tNodes-nFeat, 0);
 
  // Updating Activation IDs for all nodes
  setActID(actIDs, annbit.ActIDSets, nNodes, sNodes);
  
  // for(unsigned int i = 0; i < annbit.ActIDSets.size(); i++){
  //   unsigned int ID = annbit.ActIDSets[i].ID;
  //   if(!annbit.ActIDSets[i].layers.empty()){
  //     BUG(std::cout << "Using ActID_Set.layers" << std::endl;)
  //     for(unsigned int j = 1; j < nLayers-1; j++){
  //       BUG(
  //         print(j, "Layer");
  //         print(annbit.ActIDSets[i].layers, "Layer Vector");
  //         print(inVec(j, annbit.ActIDSets[i].layers), "j Found inVec");
  //       )
  //       if(inVec(j, annbit.ActIDSets[i].layers)){
  //         BUG(std::cout << "Layer selected" << std::endl;)
  //         for(unsigned int k = 0; k < nNodes[j]; k++){
  //           BUG(print(sNodes[j]+k-nFeat, "Node");)
  //           actIDs[sNodes[j]+k-nFeat] = ID;
  //           BUG(print(ID, "Set to ID");)
  //         }
  //       }
  //     }
  //   }else if(!annbit.ActIDSets[i].nodePositions.empty()){
  //     BUG(std::cout << "Using ActID_Set.nodePositions"  << std::endl;)
  //     for(unsigned int j = 0; j < annbit.ActIDSets[i].nodePositions.size(); j++){
  //       unsigned int pos = annbit.ActIDSets[i].nodePositions[j];
  //       actIDs[pos] = ID;
  //     }
  //   }else{
  //     BUG(errPrint("Warning - no layers or nodes were given to set act ID. Nothing is changed.");)
  //   }
  // }

  std::cout << std::endl;
  std::vector<unsigned int> actCnts(ACT1.size()*(nLayers-2), 0);
  unsigned int cnt = 0;
  for(unsigned int i = 1; i < nNodes.size()-1; i++){
    for(unsigned int j = 0; j < nNodes[i]; j++){
      actCnts[(i-1)*ACT1.size()+actIDs[cnt]] ++;
      cnt++;
    }
  }
  
  BUG(
    std::cout << "Activation Counts: ";
    for(unsigned int i = 0; i < nLayers-2; i++){
      for(unsigned int j = 0; j < ACT1.size(); j++){
        std::cout << actCnts[i*ACT1.size()+j] << " ";
      }
      std::cout << "; ";
    }
    std::cout << std::endl;
  )

 // Packing
  annbit.actCnts = actCnts;
  BUG(print(annbit.actCnts, "initANN - Activation Counts");)
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


///////////////////////////////////////////////////////////////////////////////
/// Setters
///////////////////////////////////////////////////////////////////////////////
void set_actLayers(
  std::vector<unsigned int> &actIDs,
  unsigned int id,
  std::vector<unsigned int> layers,
  std::vector<unsigned int> nNodes,
  std::vector<unsigned int> sNodes
){
  if(id > ACT1.size()){
    fprintf(
      stderr, 
      "ERROR - set_actLayers: %d is not an Activation Function Option.",
      id
    ); 
  }
  BUG(std::cout << "set_ActLayers()" << std::endl);
  unsigned int nLayers = nNodes.size();
  unsigned int nFeat = nNodes[0];
  // For each hidden layer
  for(unsigned int i = 1; i < nLayers-1; i++){
    BUG(
      print(i, "Layer");
      print(layers, "Layer Vector");
      print(inVec(i, layers), "i Found inVec");
    )
    // If setting layer i
    if(inVec(i, layers)){
      BUG(std::cout << "Layer selected" << std::endl;)
      // For each node in layer i
      for(unsigned int k = 0; k < nNodes[i]; k++){
        BUG(print(sNodes[i]+k-nFeat, "Node");)
        // Set the actID to the new ID
        actIDs[sNodes[i]+k-nFeat] = id;
        BUG(print(ID, "Set to ID");)
      }
    }
  }
  return;
}

void set_actNodes(
  std::vector<unsigned int> &actIDs,
  unsigned int id,
  std::vector<unsigned int> nodes
){
  BUG(std::cout << "set_actNodes()" << std::endl;)
  if(id > ACT1.size()){
    fprintf(
      stderr, 
      "ERROR - set_actNodes: %d is not an Activation Function Option.",
      id
    ); 
  }
  // For each node
  for(unsigned int i = 0; i < nodes.size(); i++){
    if(nodes[i] > actIDs.size()){
      fprintf(
        stderr, 
        "ERROR - set_actNodes: %d is greater than total number of nodes.",
        id
      ); 
    }
    // Set the node to the new ID
    actIDs[nodes[i]] = id;
  }
  return;
}

void set_actDivide(
  std::vector<unsigned int> &actIDs,
  std::vector<unsigned int> id_list,
  std::vector<unsigned int> nNodes
){
  BUG(std::cout << "\nset_actDivide()" << std::endl;)
  unsigned int nLayers = nNodes.size();
  unsigned int divider = id_list.size();
  unsigned int nodeCnt = 0;
  BUG(
    print(nLayers, "Number of Layers", false);
    print(divider, "Number of Activation IDs", false);
    print(nodeCnt, "Node Count");
  )
  // For each Hidden Layer
  for(unsigned int i = 1; i < nLayers-1; i++){
    // Get total number of nodes per layer
    unsigned int tNPL = nNodes[i];
    // Get the number of nodes to be set in each layer
    unsigned int sNN = floor(tNPL/divider);
    unsigned int setCnt = 0;
    unsigned int listCnt = 0;
    BUG(
      print(tNPL, "Total Nodes/Layer", false);
      print(sNN, "n Nodes to be Set");
      print(setCnt, "Set Count", false);
      print(listCnt, "ID List Count");
    )
    // For each node
    for(unsigned int j = 0; j < nNodes[i]; j++){
      if(listCnt >= id_list.size()){
        BUG(std::cout << "Break" << std::endl;)
        break;
      }
      // Set n nodes to the id in list
      actIDs[nodeCnt+j] = id_list[listCnt];
      BUG(
        print(nodeCnt+j, "Node", false);
        print(id_list[listCnt], "Act ID", false);
        print(setCnt, "Set Count");
      )
      setCnt++;
      if(setCnt >= sNN){
        listCnt++;
        setCnt = 0;
        BUG(
          print(listCnt, "List Count", false);
          print(setCnt, "Set Count");
        )
      }
    }
    nodeCnt += nNodes[i]; 
    BUG(print(nodeCnt, "Node Count");)
  }
  return;
}

void setActID(
  std::vector<unsigned int> &actIDs,
  std::vector<struct ActID_Set> sets,
  std::vector<unsigned int> nNodes,
  std::vector<unsigned int> sNodes
){
  // For each ActID_Set Update the actIDs vector
  BUG(print(sets);)
  for(unsigned int i = 0; i < sets.size(); i++){
    // If Setting A full Layer
    if(!sets[i].layers.empty()){
      set_actLayers(actIDs, sets[i].ID, sets[i].layers, nNodes, sNodes);
    }
    // Else if Setting from Nodes
    else if(!sets[i].nodePositions.empty()){
      set_actNodes(actIDs, sets[i].ID, sets[i].nodePositions);
    }else if(sets[i].divide){
      // If an ID list is given
      if(!sets[i].ID_list.empty()){
        set_actDivide(actIDs,sets[i].ID_list, nNodes);
      }else{
        set_actDivide(actIDs,count(ACT1.size()), nNodes);
      }
    }
    else{
      BUG(errPrint("Warning - setActID: Nothing is changed.");)
    }
  }
  return;
}

void setActList(std::vector<unsigned int> &list, unsigned int value){
  BUG(
    print(list, "Original Actlist");
    print(value, "Default Value");
  )
  if(inVec(value, list)){
    std::vector<unsigned int> newList(list.size()-1, 0);
    unsigned int cnt = 0;
    for(unsigned int i = 0; i < list.size(); i++){
      BUG(print(list[i], "List[i]");)
      if(list[i] != value){
        newList[cnt++] = list[i];
        BUG(std::cout << "New List Value Added" << std::endl;)
      }
    }
    list = newList;
    BUG(print(list, "New List");)
  }
  return;
}

void setHLayers(
  struct ANN_Ambit &annbit,
  unsigned int hLayers,
  std::vector<unsigned int> nNodes
){
  annbit.nLayers = hLayers+2;
  BUG(print(hLayers, "Number of Hidden Layers");)
  annbit.hNodes = std::vector<unsigned int>(hLayers, 3);
  for(unsigned int i = 0; i < nNodes.size(); i++){
    BUG(print(i, "hLayer[i]");)
    if(i > hLayers-1){return;}
    annbit.hNodes[i] = nNodes[i];
    BUG(print(nNodes[i], "Number of Nodes");)
  }
  for(unsigned int i = nNodes.size(); i < hLayers; i++){
    BUG(print(i, "hLayer[i]");)
    annbit.hNodes[i] = nNodes[nNodes.size()-1];
    BUG(print(nNodes[nNodes.size()-1], "Number of Nodes");)
  }

  return;
}

///////////////////////////////////////////////////////////////////////////////
/// Getters
///////////////////////////////////////////////////////////////////////////////
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

unsigned int getNodePosition(
  std::vector<unsigned int> nNodes,
  unsigned int Layer,
  unsigned int node
){
  unsigned int position = sum(subVecR(nNodes, Layer))+node;
  BUG(print(position, "Position");)
  return position;
}

// std::vector<unsigned int> getNodeActivations(
void getNodeActivations(
  std::vector<struct ActID_Set> &sets,
  std::vector<unsigned int> actList,
  std::vector<unsigned int> nNodes,
  unsigned int aseed
){
  srand(aseed);
  unsigned int nActs = actList.size();
  std::vector<unsigned int>  aL_Shuffle = actList;
  std::random_shuffle(aL_Shuffle.begin(), aL_Shuffle.end());
  BUG(
    print(actList, "ActList");
    print(aL_Shuffle, "Activation List Shuffled");
  )
  unsigned int nLayers = nNodes.size();

  std::vector<unsigned int> counts(nActs*nLayers, 0);
  std::vector<unsigned int> actout(TYPECHANGE*nLayers, 0);
  std::vector<unsigned int> sums(nActs, 0);
  std::vector<unsigned int> nodesR = nNodes;
  for(unsigned int i = 0; i < aL_Shuffle.size(); i++){
    for(unsigned int j = 0; j < nLayers; j++){
      unsigned int rngNum = rng(0, nodesR[j]+1);
      counts[i*nLayers+j] = rngNum;
      actout[j*aL_Shuffle[i]] = rngNum;
      sums[i] += rngNum;
      nodesR[j] -= rngNum;
      // actout[j*(nActs+1)] = nodesR[j];
    }
    BUG(print(nodesR, "Remaining Nodes");)
  }
  BUG(
    print(counts, "counts");
    print(sums, "sums");
    print(actout, "actout");
  )

  std::vector<std::vector<unsigned int>> positions(nActs);
  std::vector<unsigned int> offsets(nLayers, 0);
  for(unsigned int i = 0; i < nActs; i++){
    unsigned int count = 0; 
    unsigned int temp = 0; 
    positions[i] = std::vector<unsigned int>(sums[i]);
    for(unsigned int j = 0; j < nLayers; j++){
      for(unsigned int k = 0; k < counts[i*nLayers+j]; k++){
        positions[i][k+count] = temp + k + offsets[j];
      }
      count += counts[i*nLayers+j];
      temp += nNodes[j];
      offsets[j] += counts[i*nLayers+j];
    }
    BUG(
      print(positions[i], "Positions");
    )
  }

  // std::vector<struct ActID_Set> sets(nActs);
  for(unsigned int i = 0; i < nActs; i++){
    sets[i].ID = aL_Shuffle[i];
    sets[i].nodePositions = positions[i];
  }
  // actCounts = actout;
  // return actout;
  return;
}

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

///////////////////////////////////////////////////////////////////////////////
/// Training and Testing
///////////////////////////////////////////////////////////////////////////////
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
    BUG(
      print(i,"i");
      print(wab, "a_{i}(w*a_{i-1}+b)")
    );    
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
  BUG(print(dLoss, "dLoss");)
  std::vector<DTYPE> delta = dot(dLoss, applyDActR(l, aIDs, obs));
  BUG(print(delta, "Delta");)
  
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

unsigned int trainNN(
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
    BUG(
      print(alpha.adam, "Adam", false);
      print(alpha.alpha, "Alpha", false);
      print(alpha.beta1, "beta1", false);
      print(alpha.beta2, "beta2");
    )
    
    mtW = zero(wsize); //1st Moment std::vector
    mtB = std::vector<DTYPE> (ann.bias.size(), 0); // 1st Moment std::vector
    vtW = zero(wsize); // 2nd Moment std::vector
    vtB = std::vector<DTYPE> (ann.bias.size(), 0); // 2nd Moment std::vector
  }else{
    BUG(print(alpha.alpha, "alpha");)
  }
  
  int numThreads = omp_get_max_threads();
  if(nSamp / numThreads < 10) numThreads = nSamp / 10;
  fprintf(stderr, "%d, ", numThreads);

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

    // fprintf(stderr, "alloc, ");
    std::vector<std::vector<std::vector<DTYPE>>> threadW;
    std::vector<std::vector<DTYPE>> threadB(numThreads, std::vector<DTYPE> (ann.bias.size(),0));
    for(int i = 0; i < numThreads; i++){
        threadW.push_back(zero(wsize));
    }
    // fprintf(stderr, "loop, ");
    // For each sample
    // Parallelize
    #pragma omp parallel for num_threads(numThreads)
    for(unsigned int i = 0; i < nSamp; i++){
      int tid = omp_get_thread_num();

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
      #pragma omp critical
      {
        getResults(result, lastAct, lastLayer, lastActID, ann.lossID, i, data.obs[i]);
      }


      // Run Back Propagation
      backProp(
        ann,
        data.obs[i],
        layer,
        act,
        threadW[tid],
        threadB[tid]
      );
    }

    // fprintf(stderr, "sync, ");
    //sync deltas
    for(int i = 0; i  < numThreads; i++){
      for(int j = 0; j < wsize.size(); j++){
        add(dW[j], threadW[i][j], 0, threadW[i][j].size());
      }
      add(dB, threadB[i], 0, threadB[i].size());
    }
    
    // fprintf(stderr, "update, ");

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
      BUG(print(epoch, "Epoch"); print(converged, "Converged, In IF");)
      return epoch;
    }
  }


  return epoch;
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

///////////////////////////////////////////////////////////////////////////////
/// Runners
///////////////////////////////////////////////////////////////////////////////
void runANN(
  struct Alpha alpha,
  struct ANN_Ambit &annbit,
  struct Data data,
  double stamp
){
  // Get data separation
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

  // Build ANN
  struct Ann ann = initANN(annbit, train);
  BUG(
    std::cout << "\nGetting ANN" << std::endl;
    print(ann);
  )
  
  // std::string annPath = "../results/ann.csv";
  printTo(
    ann,
    annbit.annpath,
    stamp
  );

  // Train
  struct Results train_results(train.nSamp, train.nClasses);
  unsigned int epoch = trainNN(ann, train, train_results, alpha, annbit.maxIter);
  BUG(
    std::cout << "\nUpdated Weights and Bias" << std::endl;
    print(ann.weights, ann.bias);
    std::cout << "\nTraining Results" << std::endl;
    print(train_results);
  )
  struct Scores trainScores = getScores(train_results, train.nClasses);
  
  // Test
  struct Results test_results(test.nSamp, test.nClasses);
  testNN(ann, test, test_results);
  
  BUG(
    std::cout << "\nTesting Results" << std::endl;
    print(test_results);
  )

  struct Scores testScores = getScores(test_results, test.nClasses);

  double total = ((double)train_results.uint_ambit+(double)test_results.uint_ambit)/data.nSamp;

  printTo(
    annbit.logpath,
    epoch,
    testScores,
    trainScores,
    total
  );

  BUG(print(annbit.actCnts, "runANN - Activation Counts");)

  return;
}


void runAnalysis(
  struct Read_Ambit &readbit,
  struct ANN_Ambit &annbit,
  struct Alpha &alpha,
  struct Data data,
  bool addheader
){

  
  // Testing 
  std::vector<unsigned int> nHLayers{2, 4, 8, 16}; 
  std::vector<std::vector<unsigned int>> nNodes{
    {10, 8},
    {10, 9, 8, 7},
    {10, 10, 9, 9, 8, 8, 7, 7},
    {11, 11, 11, 10, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6},
  };
  // std::vector<std::vector<unsigned int>> nNodes{{12}};
  if(addheader){
    std::string header = buildHeader(nHLayers[nHLayers.size()-1], ACT1.size());
    addHeader(annbit.logpath, header, true);
    addheader = false;
  }
  double s_time = omp_get_wtime();
  // Alpha Loop
  for(unsigned int j = 0; j < nHLayers.size(); j++){
    setHLayers(annbit, nHLayers[j], nNodes[j]); /*Use this when selecting variable number of nodes*/
    // setHLayers(annbit, nHLayers[j], nNodes[0]); // Use when a single number of nodes is given
    unsigned int nActs = readbit.actList.size();
    annbit.ActIDSets = std::vector<struct ActID_Set>(nActs);
    getNodeActivations(
      annbit.ActIDSets,
      readbit.actList, 
      annbit.hNodes,
      readbit.aseed
    );
    

    // Inner Most loop //
    double stamp = omp_get_wtime();
    // Print Meta Data (within loop)
    printTo(annbit, readbit, alpha, data, stamp);
    

    // Run ANN
    runANN(
      alpha,
      annbit,
      data,
      stamp
    );
    // Inner Most Loop // 
    printTo(annbit.logpath, annbit.actCnts, annbit.hNodes);
    
    // Updating aseed for next run
    // readbit.aseed += 1;
    fprintf(stderr, "%f, ", stamp); 
  }
  double e_time = omp_get_wtime();
  fprintf(stderr, "%f\n", e_time - s_time);
  return;
}