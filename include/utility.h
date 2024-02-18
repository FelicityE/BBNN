#pragma once

#include "structures.h"

///////////////////////////////////////////////////////////////////////////////
/// Strings
///////////////////////////////////////////////////////////////////////////////
void rmSpace(std::string &str);
std::string str(char * value);

///////////////////////////////////////////////////////////////////////////////
/// Print
///////////////////////////////////////////////////////////////////////////////
void print(struct Data data);
void print(struct Results re);
void print(unsigned int x, std::string name = "na", bool endl = true);
void print(DTYPE x, std::string name = "na", bool endl = true);
void print(std::vector<unsigned int> v, std::string name = "na");
void print(std::vector<DTYPE> v, std::string name = "na");
int errPrint(std::string error_message);
int errPrint(std::string error_message, unsigned int a, unsigned int b);
// int errPrint(std::string error_message, double a, double b);

///////////////////////////////////////////////////////////////////////////////
/// Find Functions
///////////////////////////////////////////////////////////////////////////////
bool hasZero(std::vector<unsigned int> v);

bool match(std::vector<double> A, std::vector<double> B);
bool match(char * A, std::string B);

bool inVec(unsigned int v, std::vector<unsigned int> V);
unsigned int dupe(std::vector<unsigned int> v);

unsigned int max(std::vector<unsigned int> v);
DTYPE max(std::vector<DTYPE> v, bool returnIdx = false);

bool same(std::vector<unsigned int> v);
///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
///////////////////////////////////////////////////////////////////////////////
/// Multiply 
// std::vector<DTYPE> dot1D(
//   std::vector<DTYPE> A, 
//   std::vector<DTYPE> B, 
//   unsigned int Ar, 
//   unsigned int Ac
// );
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

std::vector<DTYPE> ewm(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B
);

std::vector<DTYPE> tensor(
  std::vector<DTYPE> A,
  std::vector<DTYPE> B
);

/// Vector manipluation
void transpose(std::vector<DTYPE> &v, unsigned int v_col);
std::vector<DTYPE> transposeR(std::vector<DTYPE> &v, unsigned int v_col);

/// Sub vectors
std::vector<unsigned int> subVector(std::vector<unsigned int> v, unsigned int strt, unsigned int size);
std::vector<DTYPE> subVector(std::vector<DTYPE> v, unsigned int strt, unsigned int size);

/// Sum Functions
DTYPE sum(std::vector<DTYPE> v);
unsigned int sum(std::vector<unsigned int> v);

/// Add 
void add(std::vector<DTYPE> &A, std::vector<DTYPE> B);
std::vector<DTYPE> addR(std::vector<DTYPE> A, std::vector<DTYPE> B);

void add(
  std::vector<DTYPE> &A,
  std::vector<DTYPE> B,
  unsigned int idx,
  unsigned int size
);
/// Zero Functions
std::vector<std::vector<DTYPE>> zero(std::vector<unsigned int> v);

/// Rand Functions
DTYPE rng(DTYPE ll = 0, DTYPE ul = 1);
std::vector<DTYPE> rng(unsigned int size, DTYPE ll = 0, DTYPE ul = 1);
unsigned int rng(unsigned int ll, unsigned int ul);
std::vector<unsigned int> rng(unsigned int size, unsigned int ll, unsigned int ul);
std::vector<unsigned int> rng_unq(unsigned int size, unsigned int ll, unsigned int ul);

/// Size Functions
unsigned int getSize(std::vector<std::vector<DTYPE>> v);
std::vector<unsigned int> getSizeVec(std::vector<std::vector<DTYPE>> v);

void setSize(std::vector<unsigned int> &v, unsigned int size, unsigned int p = 0);
void setSize(std::vector<DTYPE> &v, unsigned int size, unsigned int p = 0);
void setSize(std::vector<std::vector<DTYPE>> &v, unsigned int size, unsigned int p = 0);
void setSizeRand(std::vector<std::vector<DTYPE>> &v, unsigned int size, unsigned int p = 0);

/// Insert Functions
void insert(std::vector<unsigned int> &v, unsigned int p = 0);
void insert(std::vector<DTYPE> &v, unsigned int p = 0);
void insert(std::vector<std::vector<DTYPE>> &v, unsigned int p = 0);
void insertRand(
  std::vector<DTYPE> &v, 
  unsigned int p = 0, 
  DTYPE ll = 0, 
  DTYPE ul = 1
);
void insertRand(
  std::vector<std::vector<DTYPE>> &v,
  unsigned int p = 0,
  DTYPE ll = 0, 
  DTYPE ul = 1
);

/// Remove Functions
void rm(std::vector<unsigned int> &v, unsigned int p = 0);
void rm(std::vector<DTYPE> &v, unsigned int p = 0);
void rm(std::vector<std::vector<DTYPE>> &v, unsigned int p = 0);

/// Set Functions
void set(
  std::vector<DTYPE> &A,
  std::vector<DTYPE> B,
  unsigned int idx
);
void set(
  std::vector<DTYPE> &A,
  std::vector<DTYPE> B,
  unsigned int idx,
  unsigned int size
);
void set(
  std::vector<unsigned int> &v,
  unsigned int x,
  unsigned int idx,
  unsigned int size
);
///////////////////////////////////////////////////////////////////////////////
/// Read Functions
///////////////////////////////////////////////////////////////////////////////
int getSetup(
  struct Adam &adam,
  struct ANN_Ambit &ann_ambit,
  struct Read_Ambit &read,
  int numInputs, 
  char * inputs[]
);

/// Files
int readFile(std::vector<std::vector<std::string>> &content, std::string filename);
int getData(struct Data &data, struct Read_Ambit read);