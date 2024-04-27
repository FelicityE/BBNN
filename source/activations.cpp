#include "include/activations.h"
///////////////////////////////////////////////////////////////////////////////
// Activation Functions
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// ReLu
DTYPE relu(DTYPE x){
  if(x <= 0){return 0;}
  return x;
}
DTYPE drelu(DTYPE x){
  if(x <= 0){return 0;}
  return 1;
}
std::vector<DTYPE> relu(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> activation(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    activation[i] = relu(layer[i]);
  }
  return activation;
}
std::vector<DTYPE> drelu(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> dA(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    dA[i] = drelu(layer[i]);
  }
  return dA;
}

/// eLu
DTYPE elu(DTYPE x){
  DTYPE ambit = 1;
  if(x <= 0){
    return ambit*(exp(x)-1);
  }
  return x;
}
DTYPE delu(DTYPE x){
  DTYPE ambit = 1;
  if(x <= 0){
    return ambit*exp(x);
  }
  return 1;
}
std::vector<DTYPE> elu(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> activation(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    activation[i] = elu(layer[i]);
  }
  return activation;
}
std::vector<DTYPE> delu(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> dA(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    dA[i] = delu(layer[i]);
  }
  return dA;
}

// LeakyReLU
DTYPE leakyrelu(DTYPE x){
  DTYPE ambit = 0.01;
  if(x < 0){
    return ambit*x;
  }
  return x;
}
DTYPE dleakyrelu(DTYPE x){
  DTYPE ambit = 0.01;
  if(x <= 0){
    return ambit;
  }
  return 1;
}
std::vector<DTYPE> leakyrelu(std::vector<DTYPE> layer, OBS_TYPE obs){
  DTYPE ambit = 0.01;
  std::vector<DTYPE> activation(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    activation[i] = leakyrelu(layer[i]);
  }
  return activation;
}
std::vector<DTYPE> dleakyrelu(std::vector<DTYPE> layer, OBS_TYPE obs){
  DTYPE ambit = 0.01;
  std::vector<DTYPE> dA(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    dA[i] = dleakyrelu(layer[i]);
  }
  return dA;
}

/// GeLu
DTYPE gelu(DTYPE x){
  DTYPE z = x/sqrt(2);
  DTYPE cdf = 0.5*(1+erf(z));
  return x*cdf;
}
DTYPE dgelu(DTYPE x){
  DTYPE z = x/sqrt(2);
  DTYPE cdf = 0.5*(1+erf(z));
  DTYPE derf = 2/M_PI * exp(pow(-z,2));
  return cdf + x*derf/(2*sqrt(2));
}
std::vector<DTYPE> gelu(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> activation(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    activation[i] = gelu(layer[i]);
  }
  return activation;
}
std::vector<DTYPE> dgelu(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> dA(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    dA[i] = dgelu(layer[i]);
  }
  return dA;
}

/// Swish
DTYPE swish(DTYPE x){
  return x*sigmoid(x);
}
DTYPE dswish(DTYPE x){
  return (1+exp(-x)+x*exp(-x))/pow(1+exp(-x),2);
}
std::vector<DTYPE> swish(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(), 0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = swish(layer[i]);
  }
  return temp;
}
std::vector<DTYPE> dswish(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = dswish(layer[i]);
  }
  return temp;
}

///////////////////////////////////////////////////////////////////////////////
/// Sigmoid
DTYPE sigmoid(DTYPE x){
  DTYPE ambit = 1;
  return 1/(1+exp(-x*ambit));
}
DTYPE dsigmoid(DTYPE x){
  DTYPE ambit = 1;
  // return sigmoid(x)*(1-sigmoid(x*ambit));
  return (exp(-x*ambit)*ambit)/pow(1+exp(-x*ambit),2);
}
std::vector<DTYPE> sigmoid(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(), 0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = sigmoid(layer[i]);
  }
  return temp;
}
std::vector<DTYPE> dsigmoid(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = dsigmoid(layer[i]);
  }
  return temp;
}

/// Bipolar Sigmoid
DTYPE bisigmoid(DTYPE x){
  DTYPE ambit = 1;
  return (1-exp(-x*ambit))/(1+exp(-x*ambit));
}
DTYPE dbisigmoid(DTYPE x){
  DTYPE ambit = 1;
  // return sigmoid(x)*(1-sigmoid(x));
  return (2*exp(-x*ambit)*ambit)/pow(1+exp(-x*ambit),2);
}
std::vector<DTYPE> bisigmoid(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(), 0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = bisigmoid(layer[i]);
  }
  return temp;
}
std::vector<DTYPE> dbisigmoid(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = dbisigmoid(layer[i]);
  }
  return temp;
}

/// Tanh
DTYPE tanh_(DTYPE x){
  return std::tanh(x);
}
DTYPE dtanh(DTYPE x){
  return 1-pow(std::tanh(x),2);
}
std::vector<DTYPE> tanh_(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(), 0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = tanh_(layer[i]);
  }
  return temp;
}
std::vector<DTYPE> dtanh(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = dtanh(layer[i]);
  }
  return temp;
}

///////////////////////////////////////////////////////////////////////////////
/// Gaussian
DTYPE gaussian(DTYPE x){
  return exp(pow(-x,2));
}
DTYPE dgaussian(DTYPE x){
  return -2*x*exp(pow(-x,2));
}
std::vector<DTYPE> gaussian(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(), 0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = gaussian(layer[i]);
  }
  return temp;
}
std::vector<DTYPE> dgaussian(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i] = dgaussian(layer[i]);
  }
  return temp;
}


///////////////////////////////////////////////////////////////////////////////
/// Softmax
std::vector<DTYPE> softmax(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(),0);
  DTYPE maxA = *std::max_element(layer.begin(),layer.end());
  DTYPE denom = 0; 
  for(unsigned int i = 0; i < layer.size(); i++){
    denom += exp(layer[i]-maxA);
  }
  BUG(std::cout << "Denominator: " << denom << std::endl;)

  BUG(std::cout << "a, Softmax: ";)
  for(unsigned int i = 0; i < layer.size(); i++){
    temp[i]= exp(layer[i]-maxA)/denom;
    BUG(
      std::cout 
        << layer[i] << ", " 
        << temp[i] << "; ";
    )
  }
  BUG(std::cout << std::endl;)
  
  return temp;
}
std::vector<DTYPE> dsoftmax(std::vector<DTYPE> layer, OBS_TYPE obs){
  // dp_i/da_j e.g. dpda[0] = dp0/da0, dpda[1] = dp0/da1, dpda[2] = dp1/da0 etc.
  std::vector<std::vector<DTYPE>> dpda(layer.size());
  DTYPE maxA = *std::max_element(layer.begin(),layer.end());
  DTYPE denom = 0;
  for(unsigned int i = 0; i < layer.size(); i++){
    denom += exp(layer[i]-maxA);
  }
  BUG(
    std::cout << "Denominator: " << denom << std::endl;
    std::cout << "dpda[], Numers: " ;
  )
  for(unsigned int i = 0; i < layer.size(); i++){
    DTYPE numer = exp(layer[i]-maxA);
    dpda[i] = std::vector<DTYPE>(layer.size(),0);
    BUG(std::cout << "[";)
    for(unsigned int j = 0; j < layer.size(); j++){
      if(j == i){
        dpda[i][j] = ((numer*denom)-(exp(layer[j]-maxA)*numer))/(denom*denom);
      }else{
        dpda[i][j] = -(exp(layer[j]-maxA)*numer)/(denom*denom);
      }
      BUG(std::cout << dpda[i][j] << " ";)
    }
    BUG(std::cout << "] " << numer << "; ";)
  }
  BUG(std::cout << std::endl;)

  BUG(
    std::cout << "dpda[obs]: ";
    for(unsigned int i = 0; i < dpda[obs].size(); i++){
      std::cout << dpda[obs][i] << ", ";
    }
    std::cout << std::endl;
  )
  return dpda[obs]; 
}

/// Argmax
std::vector<DTYPE> argmax(std::vector<DTYPE> layer, OBS_TYPE obs){
  DTYPE max = layer[0];
  unsigned int index = 0;

  for(unsigned int i = 1; i < layer.size(); i++){
    if(layer[i] > max){
      max = layer[i];
      index = i;
    }
  }
  // return std::vector<DTYPE>(layer.size(), (DTYPE)index);

  std::vector<DTYPE> out(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    if(i == index){
      out[i] = 1;
      return out;
    }
  }
  return out;
}
std::vector<DTYPE> dargmax(std::vector<DTYPE> layer, OBS_TYPE obs){
  std::vector<DTYPE> temp(layer.size(), 0.0);
  return temp;
}

///////////////////////////////////////////////////////////////////////////////
// Loss Functions
///////////////////////////////////////////////////////////////////////////////
DTYPE crossentropy(DTYPE x){
  return -log(x);
}
DTYPE dcrossentropy(DTYPE x){
  BUG(std::cout << "x: " << x << std::endl; )
  return -1.0/x;
}
std::vector<DTYPE> crossentropy(
  std::vector<DTYPE> layer,
  OBS_TYPE obs
){
  std::vector<DTYPE> cross(layer.size(),0);
  for(unsigned int i = 0; i < layer.size(); i++){
    cross[i] = crossentropy(layer[i]);
  }
  return cross;
}
std::vector<DTYPE> dcrossentropy(
  std::vector<DTYPE> layer,
  OBS_TYPE obs
){
  std::vector<DTYPE> dcross(layer.size());
  for(unsigned int i = 0; i < layer.size(); i++){
    dcross[i] = dcrossentropy(layer[i]);
  }
  return std::vector<DTYPE>(dcross.size(), dcross[obs]);
}

