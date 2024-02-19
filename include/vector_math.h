#pragma once
#include "utility.h"

///////////////////////////////////////////////////////////////////////////////
/// Multiplication
///////////////////////////////////////////////////////////////////////////////
// Scalar
void dot(std::vector<DTYPE> &A, DTYPE b);
std::vector<DTYPE> dotR(std::vector<DTYPE> A, DTYPE b);

void dot(std::vector<std::vector<DTYPE>> &A, DTYPE b);
std::vector<std::vector<DTYPE>> dotR(std::vector<std::vector<DTYPE>> A, DTYPE b);

// Element-wise
std::vector<DTYPE> dot(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B
);

// Matrix Multiplcation
std::vector<DTYPE> dot(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B,
  unsigned int stride
);
std::vector<DTYPE> dotT(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B,
  unsigned int stride
);

std::vector<DTYPE> tensor(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B
);


///////////////////////////////////////////////////////////////////////////////
/// Division
///////////////////////////////////////////////////////////////////////////////
void divide(
  std::vector<DTYPE> &A, 
  std::vector<DTYPE> B
);
std::vector<DTYPE> divideR(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B
);

void divide(
  std::vector<std::vector<DTYPE>> &A, 
  std::vector<std::vector<DTYPE>> B
);
std::vector<std::vector<DTYPE>> divideR(
  std::vector<std::vector<DTYPE>> A,
  std::vector<std::vector<DTYPE>> B
);

///////////////////////////////////////////////////////////////////////////////
/// Sum
///////////////////////////////////////////////////////////////////////////////
DTYPE sum(std::vector<DTYPE> v);
unsigned int sum(std::vector<unsigned int> v);

///////////////////////////////////////////////////////////////////////////////
/// Add
///////////////////////////////////////////////////////////////////////////////
// Scaler
void add(std::vector<DTYPE> &A, DTYPE b);
std::vector<DTYPE> addR(std::vector<DTYPE> A, DTYPE b);

void add(std::vector<std::vector<DTYPE>> &A, DTYPE b);
std::vector<std::vector<DTYPE>> addR(
  std::vector<std::vector<DTYPE>> A, 
  DTYPE b
);

// Element wise
void add(std::vector<DTYPE> &A, std::vector<DTYPE> B);
std::vector<DTYPE> addR(std::vector<DTYPE> A, std::vector<DTYPE> B);

void add(
  std::vector<std::vector<DTYPE>> &A, 
  std::vector<std::vector<DTYPE>> B
);
std::vector<std::vector<DTYPE>> addR(
  std::vector<std::vector<DTYPE>> A, 
  std::vector<std::vector<DTYPE>> B
);

// Add to Position
void add(
  std::vector<DTYPE> &A,
  std::vector<DTYPE> B,
  unsigned int idx,
  unsigned int size
);

///////////////////////////////////////////////////////////////////////////////
/// Subtract
///////////////////////////////////////////////////////////////////////////////
void subtract(std::vector<DTYPE> &A, std::vector<DTYPE> B);
std::vector<DTYPE> subtractR(
  std::vector<DTYPE> A, 
  std::vector<DTYPE> B
);

void subtract(
  std::vector<std::vector<DTYPE>> &A, 
  std::vector<std::vector<DTYPE>> B
);
std::vector<std::vector<DTYPE>> subtractR(
  std::vector<std::vector<DTYPE>> A, 
  std::vector<std::vector<DTYPE>> B
);
///////////////////////////////////////////////////////////////////////////////
/// Square
///////////////////////////////////////////////////////////////////////////////
void square(std::vector<DTYPE> &A);
std::vector<DTYPE> squareR(std::vector<DTYPE> A);

void square(std::vector<std::vector<DTYPE>> &A);
std::vector<std::vector<DTYPE>> squareR(std::vector<std::vector<DTYPE>> A);

///////////////////////////////////////////////////////////////////////////////
/// Square Root
///////////////////////////////////////////////////////////////////////////////
void root(std::vector<DTYPE> &A);
std::vector<DTYPE> rootR(std::vector<DTYPE> A);

void root(std::vector<std::vector<DTYPE>> &A);
std::vector<std::vector<DTYPE>> rootR(std::vector<std::vector<DTYPE>> A);
