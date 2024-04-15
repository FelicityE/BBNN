#include "include/vector_math.h"

///////////////////////////////////////////////////////////////////////////////
/// Multiplication
///////////////////////////////////////////////////////////////////////////////
// Scalar
void dot(std::vector<DTYPE> &A, DTYPE b){
  for(unsigned int i = 0; i < A.size(); i++){
    A[i] *= b;
  }
  return;
}
std::vector<DTYPE> dotR(std::vector<DTYPE> A, DTYPE b){
  std::vector<DTYPE> temp(A.size(), 0);
  for(unsigned int i = 0; i < A.size(); i++){
    temp[i] = A[i]*b;
  }
  return temp;
}

void dot(std::vector<std::vector<DTYPE>> &A, DTYPE b){
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] *= b;
    }
  }
  return;
}
std::vector<std::vector<DTYPE>> dotR(std::vector<std::vector<DTYPE>> A, DTYPE b){
  std::vector<std::vector<DTYPE>> temp(A.size());
  for(unsigned int i = 0; i < A.size(); i++){
    temp[i] = std::vector<DTYPE>(A[i].size(),0);
    for(unsigned int j = 0; j < A[i].size(); j++){
      temp[i][j] = A[i][j] * b;
    }
  }
  return temp;
}

// Element-wise
std::vector<DTYPE> dot(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B
){
  if(A.size() != B.size()){
    errPrint("ERROR ewm: A.size() != B.size().", A.size(), B.size());
    return A;
  }
  std::vector<DTYPE> C(A.size(), 0);
  for(unsigned int i = 0; i < A.size(); i++){
    C[i] = A[i]*B[i];
  }
  return C;
}

// Matrix Multiplcation
std::vector<DTYPE> dot(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B,
  unsigned int stride
){
  // Note: B is assumed transposed
  if(A.size()%stride != 0){
    errPrint("ERROR dot: vector A.size()\% stride != 0.", A.size(), stride);
    return A;
  }else if(B.size()%stride != 0){
    errPrint("ERROR dot: vector B.size()\% stride != 0.", B.size(), stride);
    return B;
  }
  unsigned int a_row = A.size()/stride;
  unsigned int a_col = stride;
  unsigned int b_row = stride;
  unsigned int b_col = B.size()/stride;
  unsigned int c_row = a_row;
  unsigned int c_col = b_col;
  unsigned int c_size = c_row*c_col;
  std::vector<DTYPE> C(c_size, 0);
  for(unsigned int i = 0; i < a_row; i++){
    for(unsigned int j = 0; j < a_col; j++){
      for(unsigned int k = 0; k < b_col; k++){
        BUG(
          std::cout 
            << "C[" << i*c_col+k 
            << "] = A[" << i*a_col+j 
            << "] * B[" << k*b_row+j << "]" 
          << std::endl;
        )
        C[i*c_col+k] += A[i*a_col+j] * B[k*b_row+j];
      }
    }
  }
  return C;
}

std::vector<DTYPE> dotT(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B,
  unsigned int a_col
){
  if(A.size() % a_col != 0){
    errPrint("ERROR dotT: A.size\%a_col != 0.", A.size(), a_col);
    return A;
  }
  std::vector<DTYPE> temp = transposeR(A, a_col);
  return dot(temp, B, a_col);
}

std::vector<DTYPE> tensor(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B
){
  std::vector<DTYPE> C(A.size()*B.size(), 0);
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < B.size(); j++){
      C[i*B.size()+j] = A[i]*B[j];
    }
  }
  return C;
}

///////////////////////////////////////////////////////////////////////////////
/// Division
///////////////////////////////////////////////////////////////////////////////
void divide(
  std::vector<DTYPE> &A, 
  std::vector<DTYPE> B
){
  if(A.size() != B.size()){
    errPrint("ERROR - divide: A.size != B.size.", A.size(), B.size());
    return;
  }
  for(unsigned int i = 0; i < A.size(); i++){
    A[i] /= B[i];
  }
  return;
}
std::vector<DTYPE> divideR(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B
){
  if(A.size() != B.size()){
    errPrint("ERROR - divide: A.size != B.size.", A.size(), B.size());
    return A;
  }

  std::vector<DTYPE> C(A.size(),0);
  for(unsigned int i = 0; i < A.size(); i++){
    C[i] = A[i] / B[i];
  }

  return C;
}

void divide(
  std::vector<std::vector<DTYPE>> &A, 
  std::vector<std::vector<DTYPE>> B
){
  if(A.size() != B.size()){
    errPrint("ERROR - divide: A.size != B.size.", A.size(), B.size());
    return;
  }
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      errPrint("ERROR - divide: A[i].size != B[i].size.", A[i].size(), B[i].size());
      return;
    }
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] /= B[i][j];
    }
  }
  return;
}
std::vector<std::vector<DTYPE>> divideR(
  std::vector<std::vector<DTYPE>> A,
  std::vector<std::vector<DTYPE>> B
){
  if(A.size() != B.size()){
    errPrint("ERROR - divide: A.size != B.size.", A.size(), B.size());
    return A;
  }

  std::vector<std::vector<DTYPE>> C(A.size());
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      errPrint("ERROR - divide: A[i].size != B[i].size.", A[i].size(), B[i].size());
    }
    C[i] = std::vector<DTYPE>(A[i].size(), 0);
    for(unsigned int j = 0; j < A[i].size(); j++){
      C[i][j] = A[i][j] / B[i][j];
    }
  }

  return C;
}

///////////////////////////////////////////////////////////////////////////////
/// Sum
///////////////////////////////////////////////////////////////////////////////
DTYPE sum(std::vector<DTYPE> v){
  DTYPE temp = 0;
  for (unsigned int i = 0; i < v.size (); i++){
    temp += v[i];
  }
  return temp;
}
unsigned int sum(std::vector<unsigned int> v){
  unsigned int temp = 0;
  for(unsigned int i = 0; i < v.size(); i++){
    temp += v[i];
  }
  return temp;
}

///////////////////////////////////////////////////////////////////////////////
/// Add
///////////////////////////////////////////////////////////////////////////////
// Scalar
void add(std::vector<DTYPE> &A, DTYPE b){
  for(unsigned int i = 0; i < A.size(); i++){
    A[i] += b;
  }
  return;
}
std::vector<DTYPE> addR(std::vector<DTYPE> A, DTYPE b){
  std::vector<DTYPE> C(A.size(), 0);
  for(unsigned int i = 0; i < A.size(); i++){
    C[i] = A[i] + b;
  }
  return C;
}

void add(std::vector<std::vector<DTYPE>> &A, DTYPE b){
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] += b;
    }
  }
  return;
}
std::vector<std::vector<DTYPE>> addR(
  std::vector<std::vector<DTYPE>> A, 
  DTYPE b
){
  std::vector<std::vector<DTYPE>> C(A.size());
  for(unsigned int i = 0; i < A.size(); i++){
    C[i] = std::vector<DTYPE>(A[i].size(), 0);
    for(unsigned int j = 0; j < A[i].size(); j++){
      C[i][j] = A[i][j] + b;
    }
  }
  return C;
}

// Element Wise
void add(std::vector<DTYPE> &A, std::vector<DTYPE> B){
  if(A.size() != B.size()){
    errPrint("ERROR add: A.size() != B.size().", A.size(), B.size());
    return;
  }
  for(unsigned int i = 0; i < A.size(); i++){
    A[i] += B[i];
  }
  return;
}
std::vector<DTYPE> addR(std::vector<DTYPE> A, std::vector<DTYPE> B){
  std::vector<DTYPE> C(A.size(), 0);
  if(A.size() != B.size()){
    errPrint("ERROR add: A.size() != B.size().", A.size(), B.size());
    return A;
  }
  for(unsigned int i = 0; i < A.size(); i++){
    C[i] = A[i] + B[i];
  }
  return C;
}

void add(
  std::vector<std::vector<DTYPE>> &A, 
  std::vector<std::vector<DTYPE>> B
){
  if(A.size() != B.size()){
    errPrint("ERROR add: A.size() != B.size().", A.size(), B.size());
    return;
  }
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      errPrint("ERROR add: A[i].size() != B[i].size().", A[i].size(), B[i].size());
      return;
    }
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] += B[i][j];
    }
  }
  return;
}
std::vector<std::vector<DTYPE>> addR(
  std::vector<std::vector<DTYPE>> A, 
  std::vector<std::vector<DTYPE>> B
){
  if(A.size() != B.size()){
    errPrint("ERROR add: A.size() != B.size().", A.size(), B.size());
    return A;
  }
  std::vector<std::vector<DTYPE>> C(A.size());
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      errPrint("ERROR add: A[i].size() != B[i].size().", A[i].size(), B[i].size());
      return A;
    }
    C[i] = std::vector<DTYPE>(A[i].size(), 0);
    for(unsigned int j = 0; j < A[i].size(); j++){
      C[i][j] = A[i][j] + B[i][j];
    }
  }
  return C;
}

// Add to Position
void add(
  std::vector<DTYPE> &A,
  std::vector<DTYPE> B,
  unsigned int idx,
  unsigned int size
){
  if(A.size() < idx+size){
    errPrint("ERROR add: A.size() < idx+size.", A.size(), idx+size);
    exit(1); // Exit not return otherwise segfault.
  } 
  for(unsigned int i = 0; i < size; i++){
    A[i+idx] += B[i];
  }
  return;
}

///////////////////////////////////////////////////////////////////////////////
/// Subtract
///////////////////////////////////////////////////////////////////////////////
void subtract(std::vector<unsigned int> &A, std::vector<unsigned int> B){
  if(A.size() != B.size()){
    errPrint("ERROR - subMat: A.size does not match B.size.", A.size(), B.size());
    return;
  }
  for(unsigned int i = 0; i < A.size(); i++){
    A[i] -= B[i];
  }
  return;
}
void subtract(std::vector<DTYPE> &A, std::vector<DTYPE> B){
  if(A.size() != B.size()){
    errPrint("ERROR - subMat: A.size does not match B.size.", A.size(), B.size());
    return;
  }
  for(unsigned int i = 0; i < A.size(); i++){
    A[i] -= B[i];
  }
  return;
}
std::vector<DTYPE> subtractR(
  std::vector<DTYPE> A, 
  std::vector<DTYPE> B
){
  if(A.size() != B.size()){
    errPrint("ERROR - subtract: A.size does not match B.size.", A.size(), B.size());
    return A;
  }

  std::vector<DTYPE> C(A.size(),0);
  for(unsigned int i = 0; i < A.size(); i++){
    C[i] = A[i] - B[i];
  }
  return C;
}

void subtract(
  std::vector<std::vector<DTYPE>> &A, 
  std::vector<std::vector<DTYPE>> B
){
  if(A.size() != B.size()){
    errPrint("ERROR - subMat: A.size does not match B.size.", A.size(), B.size());
    return;
  }
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      errPrint("ERROR - subtract: A[i].size does not match B[i].size.", A[i].size(), B[i].size());
      return;
    }
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] -= B[i][j];
    }
  }

  return;
}
std::vector<std::vector<DTYPE>> subtractR(
  std::vector<std::vector<DTYPE>> A, 
  std::vector<std::vector<DTYPE>> B
){
  if(A.size() != B.size()){
    errPrint("ERROR - subtractR: A.size does not match B.size.", A.size(), B.size());
    return A;
  }

  std::vector<std::vector<DTYPE>> C(A.size());
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      errPrint("ERROR - subtractR: A[i].size does not match B[i].size.", A[i].size(), B[i].size());
      return A;
    }
    C[i] = std::vector<DTYPE>(A[i].size(), 0);
    for(unsigned int j = 0; j < A[i].size(); j++){
      C[i][j] = A[i][j] - B[i][j];
    }
  }
  return C;
}

///////////////////////////////////////////////////////////////////////////////
/// Square
///////////////////////////////////////////////////////////////////////////////
void square(std::vector<DTYPE> &A){
  for(unsigned int i = 0; i < A.size(); i++){
    A[i] *= A[i];
  }
  return;
}
std::vector<DTYPE> squareR(std::vector<DTYPE> A){
  std::vector<DTYPE> B(A.size(), 0);
  for(unsigned int i = 0; i < A.size(); i++){
    B[i] = A[i] * A[i];
  }
  return B;
}


void square(std::vector<std::vector<DTYPE>> &A){
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] *= A[i][j];
    }
  }
  return;
}
std::vector<std::vector<DTYPE>> squareR(std::vector<std::vector<DTYPE>> A){
  std::vector<std::vector<DTYPE>> B(A.size());
  for(unsigned int i = 0; i < A.size(); i++){
    B[i] = std::vector<DTYPE>(A[i].size(), 0);
    for(unsigned int j = 0; j < A[i].size(); j++){
      B[i][j] = A[i][j] * A[i][j];    
    }
  }
  return B;
}

///////////////////////////////////////////////////////////////////////////////
/// Square Root
///////////////////////////////////////////////////////////////////////////////
void root(std::vector<DTYPE> &A){
  for(unsigned int i = 0; i < A.size(); i++){
    A[i] = sqrt(A[i]);
  }
  return;
}
std::vector<DTYPE> rootR(std::vector<DTYPE> A){
  std::vector<DTYPE> B(A.size(), 0);
  for(unsigned int i = 0; i < A.size(); i++){
    B[i] = sqrt(A[i]);
  }
  return B;
}

void root(std::vector<std::vector<DTYPE>> &A){
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] = sqrt(A[i][j]);
    }
  }
  return;
}
std::vector<std::vector<DTYPE>> rootR(std::vector<std::vector<DTYPE>> A){
  std::vector<std::vector<DTYPE>> B(A.size());
  for(unsigned int i = 0; i < A.size(); i++){
    B[i] = std::vector<DTYPE>(A[i].size(), 0);
    for(unsigned int j = 0; j < A[i].size(); j++){
      B[i][j] = sqrt(A[i][j]);    
    }
  }
  return B;
}


