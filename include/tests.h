#pragma once

#include "bbnn.h" // The ANN
#include "files.h" // Reading/writing files and string utility
#include "data.h" // Allocating memory and data handling
#include "math.h"


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

  void forward(){
    srand(42);
    unsigned int nLayers = 4;
    unsigned int lSize [nLayers] = {3, 4, 4, 2};

    // Allocating Space
    DTYPE ** layers = allocate(nLayers, lSize);
    unsigned int * wSize = allocateUint((nLayers-1), lSize, (int)1);
    DTYPE ** weights = allocate(nLayers-1, wSize, true);
    DTYPE ** bias = allocate(nLayers-1, &lSize[1]);

    // Setting Activations
    activationFunc * activations = 
      (activationFunc*)malloc(sizeof(activationFunc)*(nLayers-1));

    for(unsigned int i = 0; i < nLayers-1; i++){
      activations[i] = sigmoid;
    }

    // Printing Weights
    cout << "Weights" << endl;
    for(unsigned int i = 0; i < nLayers-1; i++){
      for(unsigned int j = 0; j < (lSize[i]*lSize[i+1]); j++){
        cout << weights[i][j] << " ";
      }
      cout << endl;
    }
    cout << endl;

    // Giving random input values
    cout << "Input Layer" << endl;
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

    // Printing Output
    cout << "Output Layer" << endl;
    for(int i = 0; i < (lSize[nLayers-1]); i++){
      cout << layers[nLayers-1][i] << " ";
    }
    cout << endl << endl; 
  }
  
  void cosTest(){
    cout << "\ncosTest" << endl;
    // Setting inputs and observed outputs
    DTYPE inputs [16] = {
      0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330
    };
    DTYPE observed [16] = {
      1.000, 0.866, 0.707, 0.500, 
      0.000, -0.500, -0.707, -0.866, 
      -1.00, -0.866, -0.707, -0.500,
      0.000, 0.500, 0.707, 0.866
    };

    // Setting ANN size
    srand(42);
    unsigned int nLayers = 3;
    unsigned int lSize [nLayers] = {1, 2, 1};

    // Allocating Space
    DTYPE ** layers = allocate(nLayers, lSize);
    unsigned int * wSize = allocateUint((nLayers-1), lSize, (int)1);
    DTYPE ** weights = allocate(nLayers-1, wSize, true);
    DTYPE ** bias = allocate(nLayers-1, &lSize[1]);
    DTYPE * obs = allocate(lSize[nLayers-1]);

    // Setting Activations
    activationFunc * activations = 
      (activationFunc*)malloc(sizeof(activationFunc)*(nLayers-1));
    for(unsigned int i = 0; i < nLayers-1; i++){
      activations[i] = sigmoid;
    }
    activationFunc * activationsPrime = 
      (activationFunc*)malloc(sizeof(activationFunc)*(nLayers-1));
    for(unsigned int i = 0; i < nLayers-1; i++){
      activationsPrime[i] = sigmoidPrime;
    }

    // Set input layer to input
    
    cout << "Input: ";
    for(unsigned int i = 0; i < 1; i++){
      // i = 1 temp for one trial
      layers[0][i] = inputs[i]; // Inputs will later be DTYPE **
      cout  <<  layers[0][i] << " ";
    }
    cout << endl;
    cout << "Observed: ";
    for(unsigned int i = 0; i < 1; i++){
      // i < 1 temp for one trial
      obs[i] = observed[i]; // Observed will later be DTYPE ** 
      cout  <<  obs[i] << " ";
    }
    cout << endl;

    // Running forward function for each input
    runForward(
      layers,
      nLayers,
      lSize,
      weights,
      bias,
      activations
    );

    // Printing Output
    cout << "Predicted: ";
    for(int i = 0; i < (lSize[nLayers-1]); i++){
      cout << layers[nLayers-1][i] << " ";
    }
    cout << endl << endl; 

    backProbGradDecent(
      layers,
      nLayers,
      lSize,
      obs,
      weights,
      bias,
      halfSquaredErrorPrime,
      activationsPrime,
      1
    );

    runForward(
      layers,
      nLayers,
      lSize,
      weights,
      bias,
      activations
    );

    // Printing Output
    cout << "Predicted after BProp: ";
    for(int i = 0; i < (lSize[nLayers-1]); i++){
      cout << layers[nLayers-1][i] << " ";
    }
    cout << endl << endl; 




  }

}
