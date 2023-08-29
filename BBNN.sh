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