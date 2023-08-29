#pragma once

#include "bbnn.h"
#include "data.h"

namespace TEST{
  void readIn(int numInputs, char * inputs[]){
    if(inputs[1] != NULL){
      string filename = inputs[1];
      if(fExist(filename)){
        vector<vector<string>> data;
        readFile(filename, data);
        cout << endl;
        printData(data);
        cout << endl;
      }
    }

    return;
  }

  void matrix(){
    DTYPE A[4] = {1,2,3,4};
    DTYPE B[4] = {5,6,7,8};
    
    DTYPE C[4];
    
    matrixMultiply(A,2,2,B,2,2,C);

    std::cout << C[3] << std::endl;

    DTYPE D[4];
    matrixAdd(A,2,2,B,2,2,D);

    std::cout << D[3] << std::endl; 

    return;
  }

  void forward(){
    srand(42);
    unsigned int nLayers = 3;
    unsigned int lSize [nLayers] = {3, 4, 2};

    // Allocating Space
    DTYPE ** layers = (DTYPE**)malloc(sizeof(DTYPE*)*nLayers);
    for(unsigned int i = 0; i < nLayers; i++){
      layers[i] = (DTYPE *)malloc(sizeof(DTYPE)*lSize[i]);
      for(unsigned int j = 0; j < lSize[i]; j++){
        layers[i][j] = 0;
      }
    }
    DTYPE ** weights = (DTYPE**)malloc(sizeof(DTYPE*)*(nLayers-1));
    for(unsigned int i = 0; i < nLayers-1; i++){
      weights[i] = (DTYPE *)malloc(sizeof(DTYPE)*(lSize[i]*lSize[i+1]));
      for(unsigned int j = 0; j < (lSize[i]*lSize[i+1]); j++){
        weights[i][j] = ((DTYPE)rand()/RAND_MAX);
        cout << weights[i][j] << " ";
      }
      cout << endl;
    }
    cout << endl;
    DTYPE ** bias = (DTYPE**)malloc(sizeof(DTYPE*)*(nLayers-1));
    for(unsigned int i = 0; i < nLayers-1; i++){
      bias[i] = (DTYPE *)malloc(sizeof(DTYPE)*lSize[i+1]);
      for(unsigned int j = 0; j < lSize[i+1]; j++){
        bias[i][j] = 0;
      }
    }
    activationFunc * activations = (activationFunc*)malloc(sizeof(activationFunc)*(nLayers-1));
    for(unsigned int i = 0; i < nLayers-1; i++){
      activations[i] = sigmoid;
    }

    // Giving random input values
    for(int i = 0; i < lSize[0]; i++){
      layers[0][i] = ((DTYPE)rand()/RAND_MAX);
      cout << layers[0][i] << " ";
    }
    cout << endl << endl;

    // Running Forward function
    runForward(
      layers, 
      nLayers, 
      lSize, 
      weights, 
      bias, 
      activations
    );

    for(int i = 0; i < (lSize[nLayers-1]); i++){
      cout << layers[nLayers-1][i] << " ";
    }
    cout << endl << endl;
  }
}
