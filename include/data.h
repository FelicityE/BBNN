#pragma once

using namespace std;

///////////////////////////////////////////////////////////////////////////////
/// Allocating Memory
///////////////////////////////////////////////////////////////////////////////
DTYPE** allocate(unsigned int N, unsigned int * n, bool setRand = false){
  DTYPE** temp = (DTYPE**)malloc(sizeof(DTYPE*)*N);
  for(unsigned int i = 0; i < N; i++){
    temp[i] = (DTYPE *)malloc(sizeof(DTYPE)*n[i]);
    for(unsigned int j = 0; j < n[i]; j++){
      if(setRand){
        temp[i][j] =  ((DTYPE)rand()/RAND_MAX);
      }else{
        temp[i][j] = 0;
      }
    }
  } 

  return temp;
}