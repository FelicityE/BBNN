#!/bin/bash

#SBATCH --job-name=BBNN-Test
#SBATCH --output=/home/fhe2/Code/BBNN/results/log.out
#SBATCH --error=/home/fhe2/Code/BBNN/results/log.err

#SBATCH --time=1000:00
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

# setNodes x y z -> Set to actID x starting from node position y for z nodes; (setNodes <actID> <starting node> <for n nodes>)
# setNodes x list: y1 y2 -list -> Set node poitions y1 and y2 to actID x 


echo -e "maxIter, alpha, ratio, sseed, wseed, test, train, total, testLoss, trainLoss, totalLoss" >&2 
cd build/
for h in {1..10}
do
    for l in {1..10}
    do
        for i in $(seq 0 $h)
        do
            for j in $(seq 0 $l)
            do
                for p in {1..10}
                do
                    >&2 echo -n "$p, $h, $l, $i, $j, "
                    ./main ../data/DB2_E1_S8-3-8_G3_C12.txt Adam wseed $p maxIter 1000 hNodes 2 $h $l setNodes 0 0 $i setNodes 0 $h $j > ../results/log.log
                done
            done
        done
    done
done
# ./main ../data/Test.txt Adam setActs 0 2 3 -stp maxIter 1
# ./main ../data/DB2_E1_S8-3-8_G3_C12.txt Adam 