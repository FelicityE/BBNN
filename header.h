#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>


#define DTYPE double

typedef DTYPE (*lossFunc)(DTYPE,DTYPE);
typedef DTYPE (*activationFunc)(DTYPE);


using namespace std;

DTYPE sigmoid(DTYPE activation){

    return 1.0 / (1.0+exp(-1.0*activation));
}

DTYPE sigmoidPrime(DTYPE activation){
    return sigmoid(activation)*(1.0 - sigmoid(activation));
}

DTYPE halfSquaredError(DTYPE x, DTYPE y){
    return 0.5*(x-y)*(x-y);
}

DTYPE halfSquaredErrorPrime(DTYPE x, DTYPE y){
    return (x-y);
}

void matrixMultiply(DTYPE * A, unsigned int aRow, unsigned int aCol, DTYPE * B, unsigned int bRow, unsigned int bCol, DTYPE * C)
{
    if(aCol != bRow){
        std::cout << "Wrong Matrix Sizes in Multiply" << std::endl << std::flush;
        return;
    }
    unsigned int cRow = aRow;
    unsigned int cCol = bCol;
    
    for(unsigned int i = 0; i < cRow; i++){
        for(unsigned int j = 0; j < cCol; j++){
            DTYPE temp = 0;
            for(unsigned int k = 0; k < aCol; k++){
                temp += A[i*aCol+k] * B[k*bCol+j];
            }
            C[i*cCol+j] = temp;
            
        }

    }

    
    return;

}

void matrixAdd(DTYPE * A, unsigned int aRow, unsigned int aCol, DTYPE * B, unsigned int bRow, unsigned int bCol, DTYPE * C)
{
        if(aCol != bCol || aRow != bRow){
        std::cout << "Wrong Matrix Sizes in Add" << std::endl << std::flush;
        return;
    }
    unsigned int cRow = aRow;
    unsigned int cCol = bCol;

    for(unsigned int i = 0; i < cRow*cCol; i++){
        C[i] = A[i] + B[i];

    }

    
    return;

}

void matrixSubtract(DTYPE * A, unsigned int aRow, unsigned int aCol, DTYPE * B, unsigned int bRow, unsigned int bCol, DTYPE * C)
{
        if(aCol != bCol || aRow != bRow){
        std::cout << "Wrong Matrix Sizes in Add" << std::endl << std::flush;
        return;
    }
    unsigned int cRow = aRow;
    unsigned int cCol = bCol;
    
    for(unsigned int i = 0; i < cRow*cCol; i++){
        C[i] = A[i] - B[i];

    }

    
    return;

}

void updateLayer(unsigned int inCount, unsigned int outCount, DTYPE * Lin, DTYPE * Lout, DTYPE * W, DTYPE * B, activationFunc ActivationFunction){

    DTYPE * temp1 = (DTYPE*)malloc(sizeof(DTYPE*)*outCount);
    matrixMultiply(W, outCount, inCount, Lin, inCount, 1, temp1);
    matrixAdd(temp1, outCount, 1, B, outCount, 1, Lout);
    free(temp1);
    for(unsigned int i = 0; i < outCount; i++){
        Lout[i] = (*ActivationFunction)(Lout[i]);
    }
}

void runForward(DTYPE ** layers, unsigned int numLayers, unsigned int * layerSizes, DTYPE ** weights, DTYPE ** bias, activationFunc * ActivationFunction){

    for(unsigned int i = 0; i < numLayers-1; i++){
        updateLayer(layerSizes[i], layerSizes[i+1], layers[i], layers[i+1], weights[i], bias[i], ActivationFunction[i]);
    }

}

void backProbGradDecent(DTYPE ** layers, 
                        unsigned int numLayers, 
                        unsigned int * layerSizes,
                        DTYPE * expectedOutcome,
                        DTYPE ** weights, 
                        DTYPE ** bias, 
                        lossFunc LossFuncPrime, 
                        activationFunc * ActivationFunctionPrime, 
                        DTYPE learningRate){


    DTYPE ** deltas = (DTYPE **)malloc(sizeof(DTYPE*)*numLayers);
    for(unsigned int i = 0 ; i < numLayers; i ++){
        deltas[i] = (DTYPE *)malloc(sizeof(DTYPE)*layerSizes[i]);
    }

    for(unsigned int i = 0; i < layerSizes[numLayers-1]; i++){
        // get derivates of the loss values
        deltas[numLayers-1][i] = (*LossFuncPrime)(layers[numLayers-1][i], expectedOutcome[i]);
        // get the deraviteve of the activation functions
        deltas[numLayers-1][i] = deltas[numLayers-1][i] * (*ActivationFunctionPrime[numLayers-1])(layers[numLayers-1][i]);
    }

    for(unsigned int i = numLayers-2; i > 0; i--){
        for(unsigned int j = 0; j < layerSizes[i]; j++){
            deltas[i][j] = (*ActivationFunctionPrime[i])(layers[i][j]);
            DTYPE sum = 0;
            for(unsigned int k = 0; k < layerSizes[i+1]; k++){
                sum += weights[i][j*layerSizes[i] + k]*deltas[i+1][k];
            }
            deltas[i][j] = deltas[i][j]*sum;
        }
    }

    for(unsigned int l = 0; l < numLayers-1; l++){
        for(unsigned int i = 0; i < layerSizes[l]; i++){
            for(unsigned int j = 0; j < layerSizes[l+1]; j++){
                weights[l][i*layerSizes[l] + j] -= learningRate * layers[l][i] * deltas[l+1][j];
            }
        }
    }
}