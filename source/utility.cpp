#include "include/utility.h"

///////////////////////////////////////////////////////////////////////////////
/// Read Functions
//////////////////////////////////////////////////////////////////////////////
int setup(MetaRead &read, Meta &meta, int numInputs, char * inputs[]){
  if(numInputs < 2){
    std::cout << "ERROR - main input: missing data filepath." << std::endl;
    return 1;
  }
  for(unsigned int i = 0; i < numInputs; i++){
    std::cout << inputs[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
///////////////////////////////////////////////////////////////////////////////
/// Find Functions
bool hasZero(std::vector<unsigned int> v){
  for(unsigned int i = 0; i < v.size(); i++){
    if(v[i] <= 0){return true;}
  }
  return false;
}

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
