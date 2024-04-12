#pragma once

#include <iostream> // For std::cout
#include <vector> // For std::vector
#include <math.h> // For exp(), log(), sqrt(), ect.
#include <climits> // For UINT_MAX
#include <float.h> // For DBL_MAX
#include <algorithm> // For std::find

#include <fstream> // For std::fstream
#include <string> // For std::strings
#include <sstream> // For std::stringstream


#define ERRPRINT true
#define TRAINPRINT false
#define DEBUG false
#define COLOR true

#define DTYPE double

typedef std::vector<DTYPE> (*lossF)(std::vector<DTYPE> /*last Layer*/, unsigned int /*observed class*/);
typedef DTYPE (*actT1)(DTYPE /*value*/);
typedef std::vector<DTYPE> (*actT2)(std::vector<DTYPE> /*Layer*/, unsigned int /*obs or meta*/);

#if(DEBUG)
  #define BUG(x) x
#else
  #define BUG(x)
#endif

#if(COLOR)
  #define RYB(x) x
#else
  #define RYB(x)
#endif

namespace RGB{
  const std::string R = "\033[0m"; // reset
  const std::string k = "\033[30m"; // black
  const std::string r = "\033[31m"; // red
  const std::string g = "\033[32m"; // green
  const std::string y = "\033[33m"; // yellow
  const std::string b = "\033[34m"; // blue
  const std::string m = "\033[35m"; // magenta
  const std::string c = "\033[36m"; // cyan
  const std::string o = "\033[38;5;208m"; // orange
  const std::string p = "\033[38;5;206m"; // pink
  const std::string w = "\033[37m"; // white
  const std::string bold = "\033[1m"; // bold
}