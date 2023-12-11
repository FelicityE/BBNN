#!/bin/bash

#SBATCH --job-name=BBNN-Test
#SBATCH --output=/home/fhe2/Code/BBNN/results/log.out
#SBATCH --error=/home/fhe2/Code/BBNN/results/log.err

#SBATCH --time=1:00
#SBATCH --mem=300
#SBATCH --nodes=1

make
echo -e "maxIter,alpha,ratio,sseed,wseed,epoch,test,train,total" >&2 
./main ../data/DB2_E1_S8-3-8_G3_C12.txt maxIter 1 -Adam