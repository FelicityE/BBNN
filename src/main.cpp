#include "bbnn.h"

int main(){
  DTYPE A[4] = {1,2,3,4};
  DTYPE B[4] = {5,6,7,8};
  
  DTYPE C[4];
  
  matrixMultiply(A,2,2,B,2,2,C);

  std::cout << C[3] << std::endl;

  DTYPE D[4];
  matrixAdd(A,2,2,B,2,2,D);

  std::cout << D[3] << std::endl; 
}