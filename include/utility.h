#pragma once

#include "structures.h"

///////////////////////////////////////////////////////////////////////////////
/// Read Functions
//////////////////////////////////////////////////////////////////////////////
int setup(MetaRead &read, Meta &meta, int numInputs, char * inputs[]);

///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
//////////////////////////////////////////////////////////////////////////////
/// Find Functions
bool hasZero(std::vector<unsigned int> v);

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