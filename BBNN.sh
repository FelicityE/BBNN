# # !/bin/bash

# # SBATCH --job-name=BBNN

# # SBATCH --output=/home/fhe2/Code/BBNN/log.out

# # SBATCH --error=/home/fhe2/Code/BBNN/log.err

# # SBATCH --time=1:00     ## for all ranks

# # SBATCH --mem=10     
# # SBATCH --nodes=1


###############################################################################
## Input File
###############################################################################
FILE=test2.csv

###############################################################################
# Terminal Log 
## Set this variable to data or output loctation.
## .ans filetype allows for color in some text readers [VSCode].
## Log will be updated each run. 
## Save logs to a separate location to keep them.
## Comment LOG to output to terminal.
###############################################################################
# LOG=Sila-Nunam/log.ans

###############################################################################
# Users should not change code below this line
###############################################################################
cd build
bash make.sh $FILE $LOG