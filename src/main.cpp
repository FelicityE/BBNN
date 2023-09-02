// #include "bbnn.h"
// #include "data.h"
#include "tests.h"

int main(int numInputs, char * inputs[]){
  cout << endl;
  // Testing readFile Function
  // TEST::readIn(numInputs, inputs);

  // Testing Forward
  TEST::forward();
  TEST::cosTest();

  return 0;
}