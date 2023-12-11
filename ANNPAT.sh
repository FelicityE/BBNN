#!/bin/bash

#SBATCH --job-name=ANNPAT-Test
#SBATCH --output=/home/fhe2/Code/ANNPAT/log.out
#SBATCH --error=/home/fhe2/Code/ANNPAT/log.err

#SBATCH --time=1:00
#SBATCH --mem=300
#SBATCH --nodes=1

make