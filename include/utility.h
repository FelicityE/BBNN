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
int errPrint(std::string error_message);

///////////////////////////////////////////////////////////////////////////////
/// Find Functions
///////////////////////////////////////////////////////////////////////////////
bool hasZero(std::vector<unsigned int> v);

bool match(std::vector<double> A, std::vector<double> B);
bool match(char * A, std::string B);

unsigned int max(std::vector<unsigned int> v);
///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
///////////////////////////////////////////////////////////////////////////////
/// Sum Functions
DTYPE sum(std::vector<DTYPE> v);
unsigned int sum(std::vector<unsigned int> v);

/// Rand Functions
DTYPE rng(DTYPE ll = 0, DTYPE ul = 1);
std::vector<DTYPE> rng(unsigned int size, DTYPE ll = 0, DTYPE ul = 1);

/// Size Functions
unsigned int size(std::vector<std::vector<DTYPE>> v);
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
// void getData(
//   std::string filename, 
//   std::vector<std::vector<double>> &data,
//   std::vector<std::vector<double>> &observations,
//   unsigned int IDpos,
//   unsigned int skipRow,
//   unsigned int skipColumn,
//   unsigned int skipColPattern
// );
int getData(struct Data &data, struct Read_Ambit read);