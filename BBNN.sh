#!/bin/bash

#SBATCH --job-name=BBNN-Analysis
#SBATCH --output=/home/fhe2/Code/BBNN/results/log.out
#SBATCH --error=/home/fhe2/Code/BBNN/results/log.err

#SBATCH --time=300:00
#SBATCH --mem=500
#SBATCH --nodes=1

make clean
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

# setNodes x y z -> Set to actID x starting from node position y for z nodes; (setNodes <actID> <starting node> <for n nodes>)
# setNodes x list: y1 y2 -list -> Set node poitions y1 and y2 to actID x 


# echo -e "maxIter, alpha, ratio, sseed, wseed, test, train, total, testLoss, trainLoss, totalLoss" >&2 
echo -e "stamp, time" >&2
cd build/
for a in {1..10}
do
    ./main ../data/DB2_E1_S8-3-8_G3_C12.txt Adam alpha 0.001 maxIter 10000 hNodes 2 9 6 wseed $a > ../results/log.log
    for b in {1..10}
    do
        ./main ../data/DB2_E1_S8-3-8_G3_C12.txt Adam alpha 0.001 maxIter 10000 hNodes 2 9 6 wseed $a aseed $b > ../results/log.log
    done
done


# ./main ../data/winequality-red.csv ID_column 11 Adam maxIter 100000 Layers 6 8 > ../results/log.log
# for a in {1..10}
# do
#     for b in 18 10 6 4
#     do
#         for c in 6 8 10
#         do
#             ./main ../data/winequality-red.csv ID_column 11 Adam maxIter 100000 Layers $b $c aseed $a > ../results/log.log
#         done
#     done
# done