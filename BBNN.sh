#!/bin/bash

#SBATCH --job-name=BBNN-Test
#SBATCH --output=/home/fhe2/Code/BBNN/log.out
#SBATCH --error=/home/fhe2/Code/BBNN/log.err

#SBATCH --time=1:00
#SBATCH --mem=300
#SBATCH --nodes=1

make