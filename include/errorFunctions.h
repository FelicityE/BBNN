#pragma once

#include <math.h> // For sqrt() pow()
#include <vector> // For vectors
#include "params.h"
#include "utility.h"


double crossEntropy(double predictiveProbability);
double dCrossEntropy(double predictiveProbability);
std::vector<double> crossEntropy(
  std::vector<double> layer,
  std::vector<double> obs
);
std::vector<double> dCrossEntropy(
  std::vector<double> layer,
  std::vector<double> obs
);

