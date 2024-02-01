#include "include/utility.h"

///////////////////////////////////////////////////////////////////////////////
/// Find Functions
///////////////////////////////////////////////////////////////////////////////
bool hasZero(std::vector<unsigned int> v){
  for(unsigned int i = 0; i < v.size(); i++){
    if(v[i] <= 0){return true;}
  }
  return false;
}

bool match(std::vector<double> A, std::vector<double> B){
  if(A.size() != B.size()){
    std::cout << "ERROR - match: A and B are not the same size." << std::endl;
  }

  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i] != B[i]){return false;}
  }

  return true;
}
bool match(char * A, std::string B){
  for(unsigned int i = 0; i < B.length(); i++){
    if(A[i] != B[i]){
      return false;
    }
  }
  return true;
}

unsigned int max(std::vector<unsigned int> v){
  unsigned int max_ = 0;
  for(unsigned int i = 0; i < v.size(); i++){
    if(v[i] > max_){max_ = v[i];}
  }
  return max_;
}

///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
///////////////////////////////////////////////////////////////////////////////
/// Sum Functions
DTYPE sum(std::vector<DTYPE> v){
  DTYPE temp = 0;
  for (unsigned int i = 0; i < v.size (); i++){
    temp += v[i];
  }
  return temp;
}
unsigned int sum(std::vector<unsigned int> v){
  unsigned int temp = 0;
  for(unsigned int i = 0; i < v.size(); i++){
    temp += v[i];
  }
  return temp;
}

/// Rand Functions
DTYPE rng(DTYPE ll /*0*/, DTYPE ul /*1*/){
  DTYPE rn = (DTYPE)rand()/RAND_MAX;
  if(ll != 0 || ul != 1){rn =  (ul - ll) * rn + ll;}
  return rn;
}
std::vector<DTYPE> rng(unsigned int size, DTYPE ll /*0*/, DTYPE ul /*1*/){
  std::vector<DTYPE> v;
  for(unsigned int i = 0; i < size; i++){
    DTYPE rn = (DTYPE)rand()/RAND_MAX;
    if(ll != 0 || ul != 1){rn =  (ul - ll) * rn + ll;}
    v.push_back(rn);
  }
  return v;
}

/// Size Functions
unsigned int size(std::vector<std::vector<DTYPE>> v){
  unsigned int temp = 0;
  for(unsigned int i = 0; i < v.size(); i++){
    temp += v[i].size();
  }
  return temp;
}
void setSize(std::vector<unsigned int> &v, unsigned int size, unsigned int p /* = 0*/){
  if(size > v.size()){
    // Insert
    unsigned int temp = size-v.size();
    for(unsigned int i = 0; i < temp; i++){
      insert(v, p);
    }
    return;
  }
  else if(size < v.size()){
    // remove
    unsigned int temp = v.size() - size;
    for(unsigned int i = 0; i < temp; i++){
      rm(v, p);
    }
    return;
  }
  else{return;}
  return;
}
void setSize(std::vector<DTYPE> &v, unsigned int size, unsigned int p /* = 0*/){
  if(size > v.size()){
    // Insert
    unsigned int temp = size - v.size();
    for(unsigned int i = 0; i < temp; i++){
      insert(v, p);
    }
    return;
  }
  else if(size < v.size()){
    // remove
    unsigned int temp = v.size()-size;
    for(unsigned int i = 0; i < temp; i++){
      rm(v, p);
    }
    return;
  }
  else{return;}
  return;
}
void setSize(std::vector<std::vector<DTYPE>> &v, unsigned int size, unsigned int p /* = 0*/){
  if(size > v.size()){
    // Insert 
    unsigned int temp = size - v.size();
    for(unsigned int i = 0; i < temp; i++){
      insert(v, p);
    }
    return;
  }
  else if(size < v.size()){
    // remove
    unsigned int temp = v.size()-size;
    for(unsigned int i = 0; i < temp; i++){
      rm(v, p);
    }
    return;
  }
  else{return;}
  return;
}
void setSizeRand(std::vector<std::vector<DTYPE>> &v, unsigned int size, unsigned int p /* = 0*/){
  if(size > v.size()){
    // Insert 
    unsigned int temp = size - v.size();
    for(unsigned int i = 0; i < temp; i++){
      insertRand(v, p);
    }
    return;
  }
  else if(size < v.size()){
    // remove
    unsigned int temp = v.size()-size;
    for(unsigned int i = 0; i < temp; i++){
      rm(v, p);
    }
    return;
  }
  else{return;}
  return;
}

/// Insert Functions
void insert(std::vector<unsigned int> &v, unsigned int p /* = 0*/){
  if(p == 0){
    unsigned int temp = v[v.size()-2];
    v.insert(v.end()-1, temp);
  }else{
    unsigned int temp = v[p];
    v.insert(v.begin()+p, temp);
  }
  return;
}
void insert(std::vector<DTYPE> &v, unsigned int p /* = 0*/){
  if(p == 0){
    DTYPE temp = v[v.size()-2];
    v.insert(v.end()-1, temp);
  }else{
    DTYPE temp = v[p];
    v.insert(v.begin()+p, temp);
  }
  return;
}
void insert(std::vector<std::vector<DTYPE>> &v, unsigned int p /* = 0*/){
  if(p == 0){
    std::vector<DTYPE> temp = v[v.size()-2];
    v.insert(v.end()-1, temp);
  }else{
    std::vector<DTYPE> temp = v[p];
    v.insert(v.begin()+p, temp);
  }
  return;
}

void insertRand(
  std::vector<DTYPE> &v, 
  unsigned int p /* = 0*/, 
  DTYPE ll /*0*/, 
  DTYPE ul /*1*/
){
  if(p == 0){
    DTYPE temp = rng(ll, ul);
    v.insert(v.end()-1, temp);
  }else{
    DTYPE temp = rng(ll, ul);;
    v.insert(v.begin()+p, temp);
  }
  return;
}
void insertRand(
  std::vector<std::vector<DTYPE>> &v,
  unsigned int p /* = 0*/,
  DTYPE ll /*0*/, 
  DTYPE ul /*1*/
){
  if(p == 0){
    std::vector<DTYPE> temp = rng(v[v.size()-1].size(), ll, ul);
    v.insert(v.end()-1, temp);
  }else{
    std::vector<DTYPE> temp = rng(v[p].size(), ll, ul);
    v.insert(v.begin()+p, temp);
  }
  return;
}

/// Remove Functions
void rm(std::vector<unsigned int> &v, unsigned int p /* = 0*/){
  if(p == 0){
    v.erase(v.end()-2);
  }else{
    v.erase(v.begin()+p);
  }
  return;
}
void rm(std::vector<DTYPE> &v, unsigned int p /* = 0*/){
  if(p == 0){
    v.erase(v.end()-2);
  }else{
    v.erase(v.begin()+p);
  }
  return;
}
void rm(std::vector<std::vector<DTYPE>> &v, unsigned int p /* = 0*/){
  if(p == 0){
    v.erase(v.end()-2);
  }else{
    v.erase(v.begin()+p);
  }
  return;
}

///////////////////////////////////////////////////////////////////////////////
/// Read Functions
///////////////////////////////////////////////////////////////////////////////
int getSetup(
  Adam &adam, 
  ANN_Ambit &annbit, 
  Read_Ambit &read, 
  int numInputs, 
  char * inputs[]
){
  // Check that the correct number of inputs is given
  if(numInputs < 2){
    std::cout << "ERROR - main input: missing data filepath." << std::endl;
    return 1;
  }
  // Print the inputs
  for(unsigned int i = 0; i < numInputs; i++){
    std::cout << inputs[i] << " ";
  }
  std::cout << std::endl;


  if(numInputs > 2){
    for(unsigned int i = 2; i < numInputs; i++){
      if(match(inputs[i], "ID_column")){
        read.idp = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "skip_row")){
        read.skipRow = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "skip_column")){
        read.skipCol = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "ratio")){
        read.ratio = std::stod(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "sseed")){
        read.sseed = std::stoi(inputs[i+1]);
        i++;
      }

      else if(match(inputs[i],"Adam")){
        adam.adam = true;
      }else if(match(inputs[i], "alpha")){
        adam.alpha = std::stod(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "beta")){
        adam.beta1 = std::stod(inputs[i+1]);
        adam.beta2 = std::stod(inputs[i+2]);
        i += 2;
      }
      
      else if(match(inputs[i],"maxIter")){
        annbit.maxIter = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "wseed")){
        annbit.wseed = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "Layers")){
        annbit.nLayers = std::stoi(inputs[i+1]);
        i++;
        annbit.hNodes[0] = std::stoi(inputs[i+1]);
        for(unsigned int j = 1; j < annbit.nLayers-2; j++){
          annbit.hNodes.push_back(std::stoi(inputs[i+1]));
        }
        i++;
      }else if(match(inputs[i], "hNodes")){
        annbit.nLayers = std::stoi(inputs[i+1])+2;
        std::cout << "\nNumber of Layers: " << annbit.nLayers << std::endl;
        i++;
        annbit.hNodes[0] = std::stoi(inputs[i+1]);
        i++;
        for(unsigned int j = 1; j < annbit.nLayers-2; j++){
          annbit.hNodes.push_back(std::stoi(inputs[i+1]));
          i++;
        }
        // print("Hidden Nodes", nHiddenNodes);
        std::cout << std::endl;
      }
      
      else if(match(inputs[i], "setActs")){
        i++;
        std::vector<unsigned int> temp;
        unsigned int cnt;
        while(!match(inputs[i], "-stp")){
          if(cnt >= 5){
            std::cout <<
            "ERROR - SetUp: setActs was not followed by -stp after 5 or less integers."
            << std::endl;
            return 1;
          }
          temp.push_back(std::stoi(inputs[i]));
          i++;
          cnt++;
        }
        annbit.setActID_inputs.push_back(temp);
      }
      
      else{
        std::cout << "ERROR - main input: input["<< i <<"], " << inputs[i] <<  ", not found." << std::endl;
      }
    }
  }

  return 0;
}