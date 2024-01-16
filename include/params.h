#pragma once

#define ERRPRINT true
#define TRAINPRINT false
#define DEBUG true

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


#if(false)
  #define BUGT1(x) x
#else 
  #define BUGT1(x)
#endif

#if(DEBUG)
  #define BUGT2(x) x
#else 
  #define BUGT2(x)
#endif