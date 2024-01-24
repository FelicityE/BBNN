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
# -Adam -> turns on Adam; default off
# alpha x -> set learning rate alpha = x; default 0.01
# beta x y -> set first moment decay  = x and second moment decay = y; default 0.9 0.99

# Layers x y -> set number of layers = x and number of nodes for each hidden layer = y, default 3 2
# hNodes x y1 y2 ... yn -> set number hidden of layers = x and nodes for each hidden layer = yl, y2, ... yn (all hidden layers must be defined); default 1 2
# ID_column x -> set column number of class ID = x; default 0
# skip_column x -> skips the first x columns; default 0
# skip_row x -> skips the first x rows, use for headers; default 1

# maxIter x -> sets maxIter = x; default 1000
# ratio x -> set ratio of training to test set to x:(100-x); default 70
# sseed x -> set sample seed to x; default 0
# wseed x -> set random initial weights to x; default 42

# Activation Functions: default ReLu
# acts ... -acts -> all activation instructions should start with acts and end with -acts otherwise code WILL endless loop.
# L x y -> Starting from layer x repeat for y layers, set y to R to repeat for remaining layers
# N x y -> Starting from node x repeat for y nodes, set y to R to repeat for remaining nodes
# <name> -> name should be the name of the function to be used. 
# example: acts L 0 0 N 0 2 sigmoid N 3 R elu -acts
# The example would create an ANN with Layer 0 nodes 0 to 2 sigmoid and nodes 3 to the end of layer elu, all other layers will be default

echo -e "maxIter,alpha,ratio,sseed,wseed,epoch,test,train,total" >&2 
cd build/
./main ../data/Test.txt -Adam