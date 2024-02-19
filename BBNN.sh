#!/bin/bash

#SBATCH --job-name=BBNN-Test
#SBATCH --output=/home/fhe2/Code/BBNN/results/log.out
#SBATCH --error=/home/fhe2/Code/BBNN/results/log.err

#SBATCH --time=1:00
#SBATCH --mem=300
#SBATCH --nodes=1

make

# Options
# first option must always be the filename
# ID_column x -> set column number of class ID = x; default 0
# skip_row x -> skips the first x rows, use for headers; default 1
# skip_column x -> skips the first x columns; default 0

# maxIter x -> sets maxIter = x; default 1000
# ratio x -> set ratio of training to test set to x:(100-x); default 70
# sseed x -> set sample seed to x; default 0
# wseed x -> set random initial weights to x; default 42

# Adam -> turns on Adam; default off
# alpha x -> set learning rate alpha = x; default 0.01
# beta x y -> set first moment decay  = x and second moment decay = y; default 0.9 0.99

# Layers x y -> set number of layers = x and number of nodes for each hidden layer = y, default 3 2
# hNodes x y1 y2 ... yn -> set number hidden of layers = x and nodes for each hidden layer = yl, y2, ... yn (all hidden layers must be defined); default 2

# Activation Functions: default ReLu(1)
# Activations ID: sigmoid 0, relu 1, softmax 2, argmax 3; 
# To change an activation function use the following
# setActs <ID> <starting layer number> <ending layer number (exclusive)> <starting node> <ending node> -stp
# example 1: set layers 2 to the end to sigmoid
# setActs 0 2 -stp
# example 2: set layer 2 to sigmoid
# setActs 0 2 3 -stp
# example 3: set nodes 2 to the end in layer 2 to sigmoid
# setActs 0 2 3 2 -stp
# example 4: set nodes 2 and 3 in layer 2 to sigmoid
# setActs 0 2 3 2 4
# example 5: set layer 2 and layer 4 to sigmoid
# setActs 0 2 3 -stp setActs 0 4 5 -stp

echo -e "epoch, maxIter, alpha, ratio, sseed, wseed, test, train, total, testLoss, trainLoss, totalLoss" >&2 
cd build/
./main ../data/Test.txt Adam