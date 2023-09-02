#pragma once

#define DTYPE double
using namespace std;

///////////////////////////////////////////////////////////////////////////////
/// Allocating Memory
///////////////////////////////////////////////////////////////////////////////
inline DTYPE** allocate(unsigned int N, unsigned int * n, bool setRand = false){
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

inline DTYPE* allocate(unsigned int N, bool setRand = false){
  DTYPE* temp = (DTYPE*)malloc(sizeof(DTYPE)*N);
  for(unsigned int i = 0; i < N; i++){
    if(setRand){
      temp[i] =  ((DTYPE)rand()/RAND_MAX);
    }else{
      temp[i] = 0;
    }
  }
  return temp;
}

inline unsigned int* allocateUint(unsigned int N, bool setRand = false){
  unsigned int * temp = (unsigned int*)malloc(sizeof(unsigned int)*N);
  for(unsigned int i = 0; i < N; i ++){
    if(setRand){
      temp[i] = (unsigned int)rand();
    }else{
      temp[i] = 0;
    }
  }
  return 0;
}

inline unsigned int* allocateUint(unsigned int N, unsigned int * set, int opt){
  unsigned int * temp = (unsigned int*)malloc(sizeof(unsigned int)*N);
  for(unsigned int i = 0; i < N; i ++){
    switch(opt){
      // Multiply
      case 0:
        temp[i] = set[i]*set[i];
        break;
      case 1: 
        temp[i] = set[i]*set[i+1];
        break;
      // Add
      case 2:
        temp[i] = set[i]+set[i];
        break;
      case 3: 
        temp[i] = set[i]+set[i+1];
        break;
      // Subtract
      case 4: 
        temp[i] = set[i+1]-set[i];
        break;
      case 5:
        temp[i] = set[i]-set[i+1];
        break;
      // Else
      default:
        temp[i] = set[i];
        break;
    }
  }
  return temp;
}