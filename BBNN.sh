#!/bin/bash

#SBATCH --job-name=BBNN-Analysis
#SBATCH --output=/home/fhe2/Code/BBNN/results/log.out
#SBATCH --error=/home/fhe2/Code/BBNN/results/log.err

#SBATCH --time=300:00
#SBATCH --mem=500
#SBATCH -c 12

make clean
make

echo -e "stamp, time" >&2
cd build/
for c in {91..100}
do
    ./main ../data/wine.csv Adam alpha 0.001 maxIter 10000 hNodes 4 10 10 10 10 wseed 55 set_actLayer 4 4 aseed $c list: 1 2 3 :list >> ../results/log.log
    ./main ../data/wine.csv Adam alpha 0.001 maxIter 10000 hNodes 4 10 10 10 10 wseed 55 set_actDefault 4 set_actLayer 4 4 aseed $c list: 5 6 :list >> ../results/log.log
done

for a in {56..100}
do
    for b in {0..6}
    do
        ./main ../data/wine.csv Adam alpha 0.001 maxIter 10000 hNodes 4 10 10 10 10 wseed $a set_actDefault $b set_actLayer 4 4 >> ../results/log.log
    done
    for c in {1..100}
    do
        ./main ../data/wine.csv Adam alpha 0.001 maxIter 10000 hNodes 4 10 10 10 10 wseed $a set_actLayer 4 4 aseed $c list: 1 2 3 :list >> ../results/log.log
        ./main ../data/wine.csv Adam alpha 0.001 maxIter 10000 hNodes 4 10 10 10 10 wseed $a set_actDefault 4 set_actLayer 4 4 aseed $c list: 5 6 :list >> ../results/log.log
    done
done

for a in {1..100}
do
    for b in {1..100}
    do
        ./main ../data/wine.csv Adam alpha 0.001 maxIter 10000 hNodes 4 10 10 10 10 wseed $a set_actLayer 4 4 aseed $b >> ../results/log.log
    done
done


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

# LogPath <filepath> -> set the output log to go to <filepath>

# set_actNodes x y1 y2 -> Set nodes y1 for y2 (exclusive) to activation function x; (setNodes <actID> <starting node> <for n nodes>)
# set_actNodes x list: y1 y2 -list -> Set node poitions y1 and y2 to actID x 

# Analyze -> No longer supported

# aseed x -> Set the seed for the activation function random selection to x
# set_actDefault x -> Set the default activation function for all hidden layers to x
# set_actLayer x y -> Set Layer y to activation function x
# set_actLayers x y1 y2 -> Set Layers from y1 to y2 (exclusive) to activation function x
# set_actLayers x list: y1 y2 ... yn :list -> Set layers y1, y2, ... and yn to activation function x

# Activation Functions:
# 0 ReLU (Default)
# 1 ELU
# 2 Leaky ReLU
# 3 GeLU
# 4 Sigmoid
# 5 Bipolar Sigmoid
# 6 Tanh
# ----------
# Required Last Layer:
# 7 Softmax (Training)
# 8 Argmax (Testing)

# Loss Functions:
# 0 Cross Entropy