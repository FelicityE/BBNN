###############################################################################
# Developer Parameters
###############################################################################
## Make clean
MC=false
## Make New
MN=true
## Run Program
RUN=true
## Clean for Github
CLEAN=false

###############################################################################
# Inputs from User Script
###############################################################################
# CSV Filepath
FILE=$1

# Terminal Log 
LOG=$5

###############################################################################
# Non-changing names
###############################################################################
## Executable name
EX=./BBNN

###############################################################################
# Run Clean for Github
###############################################################################
if $CLEAN ; 
then
  MC=true;
  RUN=false;
fi

###############################################################################
# Change to Code Diretory
###############################################################################
cd make

###############################################################################
# Run Makeclean and Make
###############################################################################
if $MC ; then bash makeclean.sh; fi
if $MN ; then make; fi

###############################################################################
# Run BBNN
###############################################################################
if  $RUN ; 
then
  # Build if not compiled
  if  ! command -v $EX /dev/null;
  then
    echo "$EX not found";
    echo "make clean";
    bash makeclean.sh;
    echo "cmake ..";
    cmake ..;
    echo "make";
    make;
    if [ -z "$LOG" ]; then
      echo "$EX $FILE";
      $EX ../../$FILE;
    else
      echo "$EX $FILE > $LOG"
      $EX ../../$FILE > ../../$LOG;
    fi
  else
    if [ -z "$LOG" ]; then
      echo "$EX $FILE";
      $EX ../../$FILE;
    else
      echo "$EX $FILE > $LOG";
      $EX ../../$FILE > ../../$LOG;
    fi
  fi

  if [ ! $? -eq 0 ]; then
    exit 1
  fi
fi
