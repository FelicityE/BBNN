#include "include/utility.h"

///////////////////////////////////////////////////////////////////////////////
/// Misc Utility
///////////////////////////////////////////////////////////////////////////////
double getMax(std::vector<double> v){
  double max = v[0];
  for(unsigned int i = 1 ; i < v.size(); i++){
    if(max < v[i]){max = v[i];}
  }
  return max;
}
int getMax(std::vector<int> v){
  double max = v[0];
  for(unsigned int i = 1 ; i < v.size(); i++){
    if(max < v[i]){max = v[i];}
  }
  return max;
}

bool match(std::vector<double> A, std::vector<double> B){
  if(A.size() != B.size()){
    std::cout << "ERROR - match: A and B are not the same size." << std::endl;
  }

  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i] != B[i]){return false;}
  }

  return true;
}
bool match(char * A, std::string B){
  for(unsigned int i = 0; i < B.length(); i++){
    if(A[i] != B[i]){
      return false;
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
/// Files
///////////////////////////////////////////////////////////////////////////////
void rmSpace(std::string &str){
    bool start = false;
  std::string s = "";
  for(int i = 0; i < str.length(); i++){
    if(str[i] == ' '){
      continue;
    }
    s += str[i];
  }
  str=s;
}

///////////////////////////////////////////////////////////////////////////////
/// Print
///////////////////////////////////////////////////////////////////////////////
// template <typename T> void print(std::vector<T> data){
//   for(unsigned int i = 0; i < data.size(); i++){
//     std::cout << data[i] << ", ";
//   }
//   std::cout << std::endl;
// }
// template <typename T> void print(std::string head, std::vector<T> data){
//   std::cout << head << ": ";
//   for(unsigned int i = 0; i < data.size(); i++){
//     std::cout << data[i] << ", ";
//   }
//   std::cout << std::endl;
// }
// template <typename T> void print(std::vector<std::vector<T>> data){
//   for(unsigned int i = 0; i < data.size(); i++){
//     for(unsigned int j = 0; j < data[i].size(); j++){
//       std::cout << data[i][j] << ", ";    
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;
// }
// template <typename T> void print(std::string head, std::vector<std::vector<T>> data){
//   std::cout << head << ": " << std::endl;
//   for(unsigned int i = 0; i < data.size(); i++){
//     for(unsigned int j = 0; j < data[i].size(); j++){
//       std::cout << data[i][j] << ", ";    
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;
// }

// Print Doubles
void print(std::string head, double data){
  std::cout << head << ": " << data << std::endl;
}
void print(std::vector<double> data){
  for(unsigned int i = 0; i < data.size(); i++){
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}
void print(std::string head, std::vector<double> data){
  std::cout << head << ": ";
  for(unsigned int i = 0; i < data.size(); i++){
    std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}
void print(std::vector<std::vector<double>> data){
  for(unsigned int i = 0; i < data.size(); i++){
    for(unsigned int j = 0; j < data[i].size(); j++){
      std::cout << data[i][j] << ", ";    
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
void print(std::string head, std::vector<std::vector<double>> data){
  std::cout << head << ": " << std::endl;
  for(unsigned int i = 0; i < data.size(); i++){
    for(unsigned int j = 0; j < data[i].size(); j++){
      std::cout << data[i][j] << ", ";    
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Print to file
void write(std::string filename, std::vector<double> data){
  std::ofstream file(filename);
  for(unsigned int i = 0; i < data.size()-1; i++){
    file << data[i] << std::endl;
  }
  file << data[data.size()-1];
  file.close();
}
void writeTo(std::string filename, std::vector<double> data){
  std::ofstream file;
  file.open(filename, std::ofstream::app);
  file << std::endl;
  for(unsigned int i = 0; i < data.size()-1; i++){
    file << data[i] << std::endl;
  }
  file << data[data.size()-1];
  file.close();
}
void writeLine(std::string filename, std::vector<double> data){
  std::ofstream file(filename);
  for(unsigned int i = 0; i < data.size()-1; i++){
    file << data[i] << ", ";
  }
  file << data[data.size()-1];
  file.close();
}
void writeLineTo(std::string filename, std::vector<double> data){
  std::ofstream file;
  file.open(filename, std::ofstream::app);
  file << std::endl;
  for(unsigned int i = 0; i < data.size()-1; i++){
    file << data[i] << ", ";
  }
  file << data[data.size()-1];
  file.close();
}

// print WB
void printWB(
  std::vector<std::vector<double>> W, 
  std::vector<std::vector<double>> B, 
  std::vector<unsigned int> nNodes
){
  std::cout << "Weights; Bias" << std::endl;
  for(unsigned int i = 0; i < nNodes.size()-1; i++){
    unsigned int count = 0;
    std::cout << "L[" << i << "]" << std::endl;
    for(unsigned int j = 0; j < nNodes[i]*nNodes[i+1]; j++){
      std::cout << W[i][j];
      if((j+1)%nNodes[i] == 0){
        std::cout <<"; " << B[i][count] << std::endl;
        count++;
      }
      else{
        std::cout << ", ";
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
///////////////////////////////////////////////////////////////////////////////
int sumVectR(std::vector<bool> A){
  int sum = 0;
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i]){sum += 1;}
  }
  return sum;
}

// Add Vectors 
void addVec(std::vector<double> &A, std::vector<double> B){
  if(A.size() != B.size()){
    std::cout << "ERROR - addVec(): Matrices are not the same size." << std::endl;
    return;
  }

  for(unsigned int i = 0; i < A.size(); i++){
    A[i] += B[i];
  }
  return;
}
std::vector<double> addVecR(std::vector<double> A, std::vector<double> B){
  std::vector<double> C;
  if(A.size() != B.size()){
    std::cout << "ERROR - addVecR(): Matrices are not the same size." << std::endl;
    return C;
  }
  
  for(unsigned int i = 0; i < A.size(); i++){
    C.push_back(A[i]+B[i]);
  }

  return C;
}

// Subtract Vectors
void subVec(std::vector<double> &A, std::vector<double> B){
  if(A.size() != B.size()){
    std::cout << "ERROR - addVec(): Matrices are not the same size." << std::endl;
    return;
  }

  for(unsigned int i = 0; i < A.size(); i++){
    A[i] -= B[i];
  }
  return;
}
std::vector<double> subVecR(std::vector<double> A, std::vector<double> B){
  if(A.size() != B.size()){
    std::cout << "ERROR - addVec(): Matrices are not the same size." << std::endl;
    return A;
  }
  std::vector<double> C;
  for(unsigned int i = 0; i < A.size(); i++){
    C.push_back(A[i] - B[i]);
  }
  return C;
}

// Multiply Vectors Element-wise
void multVec(std::vector<double> &A, std::vector<double> B){
  if(A.size() != B.size()){
    std::cout 
      << "ERROR - multVec: A size " << A.size() 
      << " does not match B size " << B.size() << "."
    << std::endl;
  }

  for(unsigned int i = 0; i < A.size(); i++){
    A[i] *= B[i];
  }

  return;
}
std::vector<double> multVecR(std::vector<double> &A, std::vector<double> B){
  if(A.size() != B.size()){
    std::cout 
      << "ERROR - multVecR: A size " << A.size() 
      << " does not match B size " << B.size() << "."
    << std::endl;
  }

  std::vector<double> C(A.size());
  for(unsigned int i = 0; i < A.size(); i++){
    C[i] = A[i] * B[i];
  }

  return C;
}

// Multiply Scaler to Vector
void multScal(std::vector<double> &A, double scaler){
  for(unsigned int i = 0; i < A.size(); i++){
    A[i] *= scaler;
  }
  return;
}
std::vector<double> multScalR(std::vector<double> A, double scaler){
  std::vector<double> B;
  for(unsigned int i = 0; i < A.size(); i++){
    B.push_back(A[i]*scaler);
  }
  return B;
}

// Append two vectors
void vecAppend(std::vector<unsigned int> &A, std::vector<unsigned int> B){
  for(unsigned int i = 0; i < B.size(); i++){
    A.push_back(B[i]);
  }
  return;
}

// Seach in Vector for value
bool inVec(std::vector<unsigned int> v, unsigned int val){
  for(unsigned int i = 0; i < v.size(); i++){
    if(v[i] == val){return true;}
  }
  return false;
}

///////////////////////////////////////////////////////////
// Misc Vector Math
///////////////////////////////////////////////////////////
double AvgSSRS(std::vector<double> A){
  double total = 0; 
  double count = 0;
  for(unsigned int i = 0; i < A.size(); i++){
    total += sqrt(A[i]*A[i]);
    count++;
  }
  return total/count;
}

///////////////////////////////////////////////////////////////////////////////
/// Matrix Functions
///////////////////////////////////////////////////////////////////////////////
// The average sum of the square root squared
double AvgAbsSum(std::vector<std::vector<double>> A){
  double total = 0;
  double count = 0;
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < A[i].size(); j++){
      total += sqrt(A[i][j]*A[i][j]);
      count++;
    }
  }
  return total/count;
}

// Multiply Matrices
std::vector<double> multMat(
  std::vector<double> A, 
  unsigned int Ar, 
  unsigned int Ac, 
  std::vector<double> B,
  unsigned int Br,
  unsigned int Bc
){
  std::vector<double> C;
  if(Ac != Br){
    std::cout << "ERROR - multMat1D(): A columns do not match B rows" << std::endl; 
    return C;
  }
  
  unsigned int Cr = Ar;
  unsigned int Cc = Bc;

  for(unsigned int i = 0; i < Cr; i++){
    for(unsigned int j = 0; j < Cc; j++){
      double sum = 0;
      for(unsigned int k = 0; k < Ac; k++){
        sum += A[i*Ac+k] * B[k*Bc+j];
      }
      C.push_back(sum);
    }
  }
  return C;
}

// Element-wise multiplication of Matrices
void ewMM(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> B){
  if(A.size() != B.size()){
    std::cout << "ERROR - ewMM: A.size does not match B.size." << std::endl;
    return;
  }

  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      std::cout 
        << "ERROR - ewMM: A[" << i << "].size does not match B[" << i 
        << "].size."
      << std::endl;
      return;
    }
    
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] *= B[i][j];
    }
  }
  return;

}
std::vector<std::vector<double>> ewMMR(
  std::vector<std::vector<double>> A,
  std::vector<std::vector<double>> B
){
  if(A.size() != B.size()){
    std::cout << "ERROR - ewMMR: A.size does not match B.size." << std::endl;
    return A;
  }

  std::vector<std::vector<double>> C;
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      std::cout 
        << "ERROR - ewMMR: A[" << i << "].size does not match B[" << i 
        << "].size."
      << std::endl;
    }
    std::vector<double> temp;
    for(unsigned int j = 0; j < A[i].size(); j++){
      temp.push_back(A[i][j] * B[i][j]);
    }
    C.push_back(temp);
  }

  return C;
}

// Element-wise division of Matrices
void ewDM(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> B){
  if(A.size() != B.size()){
    std::cout << "ERROR - ewDM: A.size does not match B.size." << std::endl;
    return;
  }

  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      std::cout 
        << "ERROR - ewDM: A[" << i << "].size does not match B[" << i 
        << "].size."
      << std::endl;
      return;
    }
    
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] /= B[i][j];
    }
  }
  return;
}
std::vector<std::vector<double>> ewDMR(
  std::vector<std::vector<double>> A,
  std::vector<std::vector<double>> B
){
  if(A.size() != B.size()){
    std::cout << "ERROR - ewDMR: A.size does not match B.size." << std::endl;
    return A;
  }

  std::vector<std::vector<double>> C;
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      std::cout 
        << "ERROR - ewDMR: A[" << i << "].size does not match B[" << i 
        << "].size."
      << std::endl;
    }
    std::vector<double> temp;
    for(unsigned int j = 0; j < A[i].size(); j++){
      temp.push_back(A[i][j] / B[i][j]);
    }
    C.push_back(temp);
  }

  return C;
}

// Square Matrix
void sqMat(std::vector<std::vector<double>> &A){
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] *= A[i][j];
    }
  }
  return;
}
std::vector<std::vector<double>> sqMatR(std::vector<std::vector<double>> A){
  std::vector<std::vector<double>> B;
  for(unsigned int i = 0; i < A.size(); i++){
    std::vector<double> temp;
    for(unsigned int j = 0; j < A[i].size(); j++){
      temp.push_back(A[i][j] * A[i][j]);
    }
    B.push_back(temp);
  }
  return B;
}

// Square Matrix
void sqrtMat(std::vector<std::vector<double>> &A){
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] = sqrt(A[i][j]);
    }
  }
  return;
}
std::vector<std::vector<double>> sqrtMatR(std::vector<std::vector<double>> A){
  std::vector<std::vector<double>> B;
  for(unsigned int i = 0; i < A.size(); i++){
    std::vector<double> temp;
    for(unsigned int j = 0; j < A[i].size(); j++){
      temp.push_back(sqrt(A[i][j]));
    }
    B.push_back(temp);
  }
  return B;
}

// Add Matrices
void addMat(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> B){
  if(A.size() != B.size()){
    std::cout << "ERROR - addMat: A.size does not match B.size." << std::endl;
    return;
  }

  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      std::cout 
        << "ERROR - addMat: A[" << i << "].size does not match B[" << i 
        << "].size."
      << std::endl;
      return;
    }

    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] += B[i][j];
    }
  }
  return;
}
std::vector<std::vector<double>> addMatR(
  std::vector<std::vector<double>> A, 
  std::vector<std::vector<double>> B
){
  if(A.size() != B.size()){
    std::cout << "ERROR - addMatR: A.size does not match B.size." << std::endl;
    return A;
  }

  std::vector<std::vector<double>> C;
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      std::cout 
        << "ERROR - addMatR: A[" << i << "].size does not match B[" << i 
        << "].size."
      << std::endl;
    }
    std::vector<double> temp;
    for(unsigned int j = 0; j < A[i].size(); j++){
      temp.push_back(A[i][j] + B[i][j]);
    }
    C.push_back(temp);
  }

  return C;
}
// Subtract Matrices
void subMat(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> B){
  if(A.size() != B.size()){
    std::cout << "ERROR - subMat: A.size does not match B.size." << std::endl;
    return;
  }

  std::vector<std::vector<double>> C;
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      std::cout 
        << "ERROR - subMat: A[" << i << "].size does not match B[" << i 
        << "].size."
      << std::endl;
      return;
    }
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] -= B[i][j];
    }
  }

  return;
}
std::vector<std::vector<double>> subMatR(
  std::vector<std::vector<double>> A, 
  std::vector<std::vector<double>> B
){
  if(A.size() != B.size()){
    std::cout << "ERROR - subMatR: A.size does not match B.size." << std::endl;
    return A;
  }

  std::vector<std::vector<double>> C;
  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i].size() != B[i].size()){
      std::cout 
        << "ERROR - subMatR: A[" << i << "].size does not match B[" << i 
        << "].size."
      << std::endl;
    }
    std::vector<double> temp;
    for(unsigned int j = 0; j < A[i].size(); j++){
      temp.push_back(A[i][j] - B[i][j]);
    }
    C.push_back(temp);
  }

  return C;
}


// Multiply Scaler to Matrix
void multScal(std::vector<std::vector<double>> &A, double scaler){
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] *= scaler;
    }
  }
  return;
}
std::vector<std::vector<double>> multScalR(std::vector<std::vector<double>> A, double scaler){
  std::vector<std::vector<double>> B;
  for(unsigned int i = 0; i < A.size(); i++){
    std::vector<double> temp;
    for(unsigned int j = 0; j < A[i].size(); j++){
      temp.push_back(A[i][j]*scaler);
    }
    B.push_back(temp);
  }
  return B;
}

// Add Scaler to Matrix
void addScal(std::vector<std::vector<double>> &A, double scaler){
  for(unsigned int i = 0; i < A.size(); i++){
    for(unsigned int j = 0; j < A[i].size(); j++){
      A[i][j] += scaler;
    }
  }
  return;
}
std::vector<std::vector<double>> addScalR(std::vector<std::vector<double>> A, double scaler){
  std::vector<std::vector<double>> B;
  for(unsigned int i = 0; i < A.size(); i++){
    std::vector<double> temp;
    for(unsigned int j = 0; j < A[i].size(); j++){
      temp.push_back(A[i][j]+scaler);
    }
    B.push_back(temp);
  }
  return B;
}

///////////////////////////////////////////////////////////////////////////////
/// Read Data
///////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<std::string>> readFile(std::string filename){
  // Creating Vector to hold Content
  std::vector<std::vector<std::string>> content;
  std::vector<std::string> row;
  std::string line;
  std::string word;
  std::fstream file(filename, std::ios::in);

  // Opening File
  if(file.is_open()){
    while(getline(file, line) && !line.empty()){
      row.clear();
      
      rmSpace(line); // Removes excess space
      std::stringstream str(line);
      
      while(getline(str, word, ',')){
        
        row.push_back(word);
      }
      content.push_back(row);
    }
  }else{
    std::cout 
      << "ERROR - data.readFile: Could not open file " << filename << std::endl;
    return content;
  }
  file.close();

  return content;
}

// Get data with ID at Position
void getDataID(
  std::string filename, 
  std::vector<std::vector<double>> &data,
  std::vector<std::vector<double>> &observations,
  unsigned int IDpos,
  unsigned int skipRow,
  unsigned int skipColumn,
  unsigned int skipColPattern
){
  std::vector<std::vector<std::string>> dataS = readFile(filename);
  std::vector<int> ID;

  for(unsigned int i = skipRow; i < dataS.size(); i++){
    std::vector<double> temp;
    for(unsigned int j = skipColumn; j < dataS[i].size(); j++){
      if(j == IDpos){
        ID.push_back(std::stoi(dataS[i][j]));
      }
      else{
        temp.push_back(std::stod(dataS[i][j]));
      }
      if(skipColPattern != 0){
        for(unsigned int k = 0; k < skipColPattern; k++){
          j++;
        }
      }
    }
    data.push_back(temp);
  }

  int nClass = getMax(ID);

  for(unsigned int i = 0; i < ID.size(); i++){
    std::vector<double> temp;
    for(unsigned int j = 0; j < nClass; j++){
      if(j == ID[i]-1){
        temp.push_back(1);
      }else{
        temp.push_back(0);
      }
    }
    observations.push_back(temp);
  }

  return;
}

///////////////////////////////////////////////////////////////////////////////
/// String functions
///////////////////////////////////////////////////////////////////////////////
bool find(char * A[], int nA, std::string B){
  for(unsigned int i = 0; i < nA; i++){
    if(match(A[i], B)){return true;}
  }
  return false;
}
bool find(char * A[], int strt, int end, std::string B){
  for(unsigned int i = strt; i < end; i++){
    if(match(A[i], B)){return true;}
  }
  return false;
}