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
# -Adam -> turns off Adam; default on
# alpha x -> set learning rate alpha = x; default 0.01
# beta x y -> set first moment decay  = x and second moment decay = y; default 0.9 0.99

# Layers x y -> set number of layers = x and number of nodes for each hidden layer = y, default 3 2
# hNodes x y1 y2 ... yn -> set number hidden of layers = x and nodes for each hidden layer = yl, y2, ... yn (all hidden layers must be defined); default 1 2
# ID_column x -> set column number of class ID = x; default 0
# skip_column x -> skips the first x columns; default 0

# maxIter x -> sets maxIter = x; default 1000
# ratio x -> set ratio of training to test set to x:(100-x); default 70
# sseed x -> set sample seed to x; default 0
# wseed x -> set random initial weights to x; default 42

echo -e "maxIter,alpha,ratio,sseed,wseed,epoch,test,train,total" >&2 
cd build/
./main ../data/DB2_E1_S8-3-8_G3_C12.txt