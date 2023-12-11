#pragma once

#define ERRPRINT true
#define TRAINPRINT false

#define DTYPE double

typedef  std::vector<double> (*lossFunction)(
  std::vector<double>, /*lastLayer*/
  std::vector<double> /*observations*/
);
// typedef  std::vector<double> (*dlossFunction)(
//   std::vector<double>, /*lastLayer*/
//   std::vector<double> /*observations*/
// );
// typedef std::vector<double> (*activationFunction)(double*, int, int);
typedef std::vector<double> (*activationFunction)(std::vector<double>, int, int);
// typedef  std::vector<double> (*dActivationFunction)(double*, int);
