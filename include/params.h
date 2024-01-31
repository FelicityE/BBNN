#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <math.h>
#include <climits>

#define ERRPRINT true
#define TRAINPRINT false
#define DEBUG false

#define DTYPE double

typedef std::vector<DTYPE> (*lossF)(std::vector<DTYPE> /*last Layer*/, int /*observed class*/);
typedef std::vector<DTYPE> (*actF)(std::vector<DTYPE> /*Layer*/, std::vector<DTYPE>);

#if(DEBUG)
  #define BUG(x) x
#else
  #define BUG(x)
#endif

