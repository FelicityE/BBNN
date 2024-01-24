#pragma once

#include "params.h"

///////////////////////////////////////////////////////////////////////////////
/// Set Defaults
//////////////////////////////////////////////////////////////////////////////
struct Meta{
  Meta(): maxIter(1000), ratio(70), sseed(0), wseed(42){}
  unsigned int maxIter;
  double ratio;
  unsigned int sseed;
  unsigned int wseed;
};

struct MetaRead{
  MetaRead(): idp(0), skipRow(1), skipCol(1){}
  unsigned int idp;
  unsigned int skipRow;
  unsigned int skipCol;
};

void setMeta(unsigned int &ambit, unsigned int value);
void setMeta(double &ambit, double value);
void setMeta(int &ambit, int value);

///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
//////////////////////////////////////////////////////////////////////////////
unsigned int sum(std::vector<unsigned int> v);