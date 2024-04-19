#include "include/utility.h"


///////////////////////////////////////////////////////////////////////////////
/// Strings
///////////////////////////////////////////////////////////////////////////////
void rmSpace(std::string &str){
  bool start = false;
  std::string s = "";
  for(int i = 0; i < str.length(); i++){
    if(str[i] == ' '){continue;}
    s += str[i];
  }
  str=s;
  return;
}
std::string str(char * value){std::string s = value; return s;}

///////////////////////////////////////////////////////////////////////////////
/// Print
///////////////////////////////////////////////////////////////////////////////
// File Handling
bool is_empty(std::ifstream& pFile){
  return pFile.peek() == std::ifstream::traits_type::eof();
}

std::string buildHeader(unsigned int nClasses){
  std::string header = "stamp, maxIter, ratio, aseed, wseed, sseed, adam, alpha, nFeat, nClass, nSamp, nLayers, tNodes, epoch, total-a, test-a, ";

  std::vector<std::string> headers{"test-p", "test-r", "test-f"};
  for(unsigned int i = 0; i < nClasses; i++){
    for(unsigned int j = 0; j < headers.size(); j++){
     header += headers[j] + std::to_string(i) +", "; 
    }
  }

  header += "train-a";
  headers = {"train-p", "train-r", "train-f"};
  for(unsigned int i = 0; i < nClasses; i++){
    for(unsigned int j = 0; j < headers.size(); j++){
      header += ", " + headers[j] + std::to_string(i); 
    }
  }
  
  return header;
}
std::string buildHeader(unsigned int nLayers, unsigned int nActIDs){
  std::string header = "";
  for(unsigned int i = 1; i < nLayers+1; i++){
    header += ", L" + std::to_string(i);
    for(unsigned int j = 0; j < nActIDs; j++){
      header += ", L" + std::to_string(i) + "A" + std::to_string(j);
    }
  }
  return header;
}
bool addHeader(std::string filename, std::string header, bool addheader /*= false*/){
  std::ifstream check(filename);
  if(!check || is_empty(check)){
    check.close();
    addheader = true;
  }
  
  std::ofstream file;
  file.open(filename, std::ofstream::app);
  if(addheader){file << header;}
  file.close();
  
  return addheader;
}

void printTo(
  struct ANN_Ambit annbit,
  struct Read_Ambit read,
  struct Alpha alpha,
  struct Data data,
  double stamp
){
  std::ofstream file;
  file.open(annbit.logpath, std::ofstream::app);
  
  unsigned int temp = 0;
  for(unsigned int i = 0; i < annbit.hNodes.size(); i++){
    temp += annbit.hNodes[i];
  }
  unsigned int tNodes = temp+data.nFeat+data.nClasses;
  
  file << "\n" << std::setprecision(13) << stamp;
  file << ", " << annbit.maxIter << ", " << read.ratio[0]
    << ", " << annbit.aseed << ", " << annbit.wseed << ", " << read.sseed[0]
    << ", " << alpha.adam << ", " << alpha.alpha
    << ", " << data.nFeat << ", " << data.nClasses << ", " << data.nSamp
    << ", " << annbit.nLayers << ", " << tNodes;

  file.close();
}
void printTo(
  std::string filename,
  unsigned int epoch,
  struct Scores testscores, 
  struct Scores trainscores, 
  double totalAccuracy
){
  std::ofstream file;
  file.open(filename, std::ofstream::app);

  file << ", " << epoch <<  ", " << totalAccuracy << ", " << testscores.accuracy;
  for(unsigned int i = 0; i < testscores.F1.size(); i++){
    file 
      << ", " << testscores.precision[i]
      << ", " << testscores.recall[i]
      << ", " << testscores.F1[i];
  }
  file << ", " << trainscores.accuracy;
  for(unsigned int i = 0; i < trainscores.F1.size(); i++){
    file 
      << ", " << trainscores.precision[i]
      << ", " << trainscores.recall[i]
      << ", " << trainscores.F1[i];
  }
  
  file.close();
}
void printTo(struct Ann ann, std::string filename, double stamp){
  std::ofstream file;
  file.open(filename, std::ofstream::app);

  unsigned int p = 0;
  // file << "Stamp: " << stamp;
  file << std::setprecision(13) << stamp;
  for(unsigned int i = 1; i < ann.nNodes.size(); i++){
    file << ", L" << i << "("<< ann.nNodes[i] <<"):";
    for(unsigned int j = 0; j < ann.nNodes[i]; j++){
      file << ", " << ann.actIDs[p];
      p++;
    }
  }
  file << "\n";
  file.close();
}
void printTo(
  std::string filepath, 
  std::vector<unsigned int> actCnts,
  std::vector<unsigned int> hNodes
){
  std::ofstream file;
  file.open(filepath, std::ofstream::app);

  file << ", " << hNodes[0];

  unsigned int step = actCnts.size()/hNodes.size();
  unsigned int lcnt = 0;
  for(unsigned int i = 0; i < actCnts.size(); i++){
    if(i%step == 0 && i != 0){
      file << ", " << hNodes[++lcnt];
    }
    file << ", " << actCnts[i];
  }
  file.close();
}

void print(struct Data data){
  std::cout 
    << "Number of Features: " << data.nFeat << "; "
    << "Number of Classes: " << data.nClasses << "; "
    << "Number of Samples: " << data.nSamp << "; "
    << "Sample Seed: " << data.sseed << "; "
    << "Percent of Samples: " << data.ratio << "; "
  << std::endl;
  std:: cout << "Data: " << std::flush;
  unsigned int cnt = 0;
  for(unsigned int i = 0; i < data.feat.size(); i++){
    if(i%data.nFeat == 0){
      std::cout << "\nObs: " << data.obs[cnt] << " Feat: ";
      cnt++;
    }
    std::cout << data.feat[i] << " ";
  }
  std::cout << std::endl;
}
void print(struct Results re){
  std::cout << "Number of Correct Predictions (for Classification): " << re.uint_ambit << std::endl;
  std::cout << "Error (Summed Loss Function): " << re.double_ambit << std::endl;
  std::cout << "Sample Prediction Correct: ";
  for(unsigned int i = 0; i < re.vector_bool.size(); i++){
    std::cout << re.vector_bool[i] << ", ";
  }
  std::cout << std::endl;
  std::cout << "Sample Prediction uint(" << re.vector_uint.size() << "):";
  for(unsigned int i = 0; i < re.vector_uint.size(); i++){
    std::cout << re.vector_uint[i] << ", ";
  }
  std::cout << std::endl;
  std::cout << "Sample Prediction dtype(" << re.vector_dtype.size() << "):";
  for(unsigned int i = 0; i < re.vector_dtype.size(); i++){
    std::cout << re.vector_dtype[i] << ", ";
  }
  std::cout << std::endl;
}

void print(bool x, std::string name /*na*/, bool endl /*true*/){
  if(name != "na"){
    std::cout << name << ": ";
  }

  if(x){
    std::cout << "true" << std::flush;
  }else{
    std::cout << "false" << std::flush;
  }

  if(endl){
    std::cout << std::endl;
  }else{
    std::cout << ", ";
  }
}
void print(unsigned int x, std::string name /*na*/, bool endl /*true*/){
  if(name != "na"){
    std::cout << name << ": ";
  }
  std::cout << x << std::flush;
  if(endl){
    std::cout << std::endl;
  }else{
    std::cout << ", ";
  }
}
void print(DTYPE x, std::string name /*na*/, bool endl /*true*/){
  if(name != "na"){
    std::cout << name << ": ";
  }
  std::cout << x << std::flush;
  if(endl){
    std::cout << std::endl;
  }else{
    std::cout << ", ";
  }
}

void print(std::vector<unsigned int> v, std::string name /*na*/){
  if(name != "na"){
    std::cout << name << ": " << std::flush;
  }
  for(unsigned int i = 0; i < v.size()-1; i++){
    std::cout << v[i] << ", " << std::flush;
  }
  std::cout << v[v.size()-1] << std::endl << std::flush;
}
void print(std::vector<DTYPE> v, std::string name /*na*/){
  if(name != "na"){
    std::cout << name << ": " << std::flush;
  }
  for(unsigned int i = 0; i < v.size()-1; i++){
    std::cout << v[i] << ", " << std::flush;
  }
  std::cout << v[v.size()-1] << std::endl << std::flush;
}
void print(std::vector<std::vector<DTYPE>> v, std::string name /*na*/){
  if(name != "na"){
    std::cout << name << ": " << std::flush;
  }
  for(unsigned int i = 0; i < v.size(); i++){
    std::cout << "\n\tL" << i << "("<< v[i].size() <<") ";
    for(unsigned int j = 0; j < v[i].size(); j++){
      std::cout << v[i][j] << ", ";
    }
    std::cout << v[i][v[i].size()-1] << std::flush;
  }
  std::cout << std::endl;
}
void print(std::vector<std::vector<unsigned int>> v, std::string name /*na*/){
  if(name != "na"){
    std::cout << name << ": " << std::flush;
  }
  for(unsigned int i = 0; i < v.size(); i++){
    std::cout << "[" << i << "]";
    for(unsigned int j = 0; j < v[i].size(); j++){
      if(j != 0){std::cout << ", ";}
      std::cout << v[i][j];
    }
    std::cout << "; ";
  }

  std::cout << std::endl;
  return;
}

void print(std::string str){
  std::cout << str << std::endl << std::flush;
  return;
}
void print(char* str){
  std::cout << str << std::endl << std::flush;
  return;
}

int errPrint(std::string error_message){
  RGB(std::cout << rgb::r;)
  std::cout << error_message << std::endl << std::flush;
  RGB(std::cout << rgb::R << std::flush;)
  return 1;
}
int errPrint(std::string error_message, unsigned int a, unsigned int b){
  RGB(std::cout << rgb::r;)
  std::cout << error_message << "\t" << a << " : " << b << std::endl << std::flush;
  RGB(std::cout << rgb::R << std::flush;)
  return 1;
}

///////////////////////////////////////////////////////////////////////////////
/// Find Functions
///////////////////////////////////////////////////////////////////////////////
bool hasZero(std::vector<unsigned int> v){
  for(unsigned int i = 0; i < v.size(); i++){
    if(v[i] <= 0){return true;}
  }
  return false;
}

bool match(std::vector<double> A, std::vector<double> B){
  if(A.size() != B.size()){
    errPrint("ERROR - match: A and B are not the same size.", A.size(), B.size());
    exit(1);
  }

  for(unsigned int i = 0; i < A.size(); i++){
    if(A[i] != B[i]){return false;}
  }

  return true;
}
bool match(char * A, std::string B){
  if(A == B){
    return true;
  }
  return false;
}

bool inVec(unsigned int v, std::vector<unsigned int> V){
  for(unsigned int i = 0; i < V.size(); i++){
    if(V[i] == v){return true;}
  }
  return false;
}

unsigned int dupe(std::vector<unsigned int> v){
  for(unsigned int i = 0; i < v.size(); i++){
    for(unsigned int j = i+1; j < v.size(); j++){
      if(v[i] == v[j]){return j;}
    }
  }
  return 0;
}

unsigned int max(std::vector<unsigned int> v){
  unsigned int max_ = 0;
  for(unsigned int i = 0; i < v.size(); i++){
    if(v[i] > max_){max_ = v[i];}
  }
  return max_;
}

DTYPE max(std::vector<DTYPE> v, bool returnIdx /*false*/){
  DTYPE max_ = 0;
  unsigned int idx = 0; 
  for(unsigned int i = 0; i < v.size(); i++){
    if(v[i] > max_){max_ = v[i]; idx=i;}
  }
  if(returnIdx){
    if(idx > DBL_MAX){
      errPrint("ERROR max(): return index, index greater than DBL_MAX.");
      std::cout << idx << std::endl;
    }
    return idx;
  }
  return max_;
}

bool same(std::vector<unsigned int> v){
  for(unsigned int i = 0; i < v.size()-1; i++){
    if(v[i] != v[i+1]){return false;}
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
///////////////////////////////////////////////////////////////////////////////

// Vector manipluation
void transpose(std::vector<DTYPE> &v, unsigned int v_col){
  if(v.size()%v_col != 0){
    errPrint("ERROR transpose: v.size\%stride != 0.", v.size(), v_col);
    return;
  }
  unsigned int v_row = v.size()/v_col;
  std::vector<DTYPE> temp(v.size(), 0);
  for(unsigned int i = 0; i < v_row; i++){
    for(unsigned int j = 0; j < v_col; j++){
      temp[j*v_row+i] = v[i*v_col+j];
    }
  }
  v = temp;
  return;
}

std::vector<DTYPE> transposeR(std::vector<DTYPE> &v, unsigned int v_col){
  if(v.size()%v_col != 0){
    errPrint("ERROR transpose: v.size\%stride != 0.", v.size(), v_col);
    return v;
  }
  unsigned int v_row = v.size()/v_col;
  std::vector<DTYPE> temp(v.size(), 0);
  for(unsigned int i = 0; i < v_row; i++){
    for(unsigned int j = 0; j < v_col; j++){
      temp[j*v_row+i] = v[i*v_col+j];
    }
  }
  return temp;
}

/// Subvectors
std::vector<unsigned int> subVector(std::vector<unsigned int> v, unsigned int strt, unsigned int size){
  std::vector<unsigned int> temp(size, 0);
  for(unsigned int i = 0; i < size; i++){
    temp[i] = v[strt+i];
  }
  return temp;
}
std::vector<unsigned int> subVecR(
  std::vector<unsigned int> v, 
  unsigned int size, 
  unsigned int strt /*=0*/
){
  std::vector<unsigned int> temp(size, 0);
  for(unsigned int i = 0; i < size; i++){
    temp[i] = v[strt+i];
  }
  return temp;
}
std::vector<DTYPE> subVector(std::vector<DTYPE> v, unsigned int strt, unsigned int size){
  std::vector<DTYPE> temp(size, 0);
  for(unsigned int i = 0; i < size; i++){
    temp[i] = v[strt+i];
  }
  return temp;
}

// Zero Functions
std::vector<std::vector<DTYPE>> zero(std::vector<unsigned int> v){
  std::vector<std::vector<DTYPE>> z;
  for(unsigned int i = 0; i < v.size(); i++){
    z.push_back(std::vector<DTYPE>(v[i], 0));
  }
  return z;
}
// std::vector<unsigned int> zeros(unsigned int size){
//   std::vector<unsigned int> v(size,0);
//   return v;
// }

/// Count Functions
std::vector<unsigned int> count(unsigned int size, unsigned int strt){
  std::vector<unsigned int> v(size,0);
  for(unsigned int i = 0; i < size; i++){
    v[i] = strt+i;
  }
  return v;
}


/// Rand Functions
DTYPE rng(DTYPE ll /*0*/, DTYPE ul /*1*/){
  DTYPE rn = (DTYPE)rand()/RAND_MAX;
  if(ll != 0 || ul != 1){rn =  (ul - ll) * rn + ll;}
  return rn;
}
std::vector<DTYPE> rng(unsigned int size, DTYPE ll /*0*/, DTYPE ul /*1*/){
  std::vector<DTYPE> v(size,0);
  for(unsigned int i = 0; i < size; i++){
    v[i] = rng(ll, ul);
  }
  return v;
}
unsigned int rng(unsigned int ll, unsigned int ul){
  unsigned int rn = rand() % ul;
  if(ll != 0){
    while(rn < ll){rn = rand()% ul;}
  }
  return rn;
}
std::vector<unsigned int> rng(unsigned int size, unsigned int ll, unsigned int ul){
  std::vector<unsigned int> v(size,0);
  for(unsigned int i = 0; i < size; i++){
    v[i] = rng(ll, ul);
  }
  return v;
}
std::vector<unsigned int> rng(
  unsigned int size, 
  std::vector<unsigned int> ll, 
  std::vector<unsigned int> ul
){
  if((size != ll.size()) || (size != ul.size())){
    errPrint("ERROR - rng: size != ll.size() != ul.size().");
  }
  std::vector<unsigned int> v(size, 0);
  for(unsigned int i = 0; i < size; i++){
    v[i] = rng(ll[i], ul[i]);
  }
  BUG(print(v, "rng()=V");)
  return v;
}
std::vector<unsigned int> rng_unq(unsigned int size, unsigned int ll, unsigned int ul){
  std::vector<unsigned int> v(size, 0);
  for(unsigned int i = 0; i < size; i++){
    unsigned int rn = rng(ll, ul);
    while(std::find(v.begin(), v.end(), rn) != v.end()){rn = rng(ll, ul);}
    v[i] = rn;
  }
  return v;
}

/// Size Functions
unsigned int getSize(std::vector<std::vector<DTYPE>> v){
  unsigned int temp = 0;
  for(unsigned int i = 0; i < v.size(); i++){
    temp += v[i].size();
  }
  return temp;
}
std::vector<unsigned int> getSizeVec(std::vector<std::vector<DTYPE>> v){
  std::vector<unsigned int> s;
  for(unsigned int i = 0; i < v.size(); i++){
    s.push_back(v[i].size());
  }
  return s;
}

void setSize(std::vector<unsigned int> &v, unsigned int size, unsigned int p /*0*/){
  if(size > v.size()){
    // Insert
    unsigned int temp = size-v.size();
    for(unsigned int i = 0; i < temp; i++){
      insert(v, p);
    }
    return;
  }
  else if(size < v.size()){
    // remove
    unsigned int temp = v.size() - size;
    for(unsigned int i = 0; i < temp; i++){
      rm(v, p);
    }
    return;
  }
  else{return;}
  return;
}
void setSize(std::vector<DTYPE> &v, unsigned int size, unsigned int p /*0*/){
  if(size > v.size()){
    // Insert
    unsigned int temp = size - v.size();
    for(unsigned int i = 0; i < temp; i++){
      insert(v, p);
    }
    return;
  }
  else if(size < v.size()){
    // remove
    unsigned int temp = v.size()-size;
    for(unsigned int i = 0; i < temp; i++){
      rm(v, p);
    }
    return;
  }
  else{return;}
  return;
}
void setSize(std::vector<std::vector<DTYPE>> &v, unsigned int size, unsigned int p /*0*/){
  if(size > v.size()){
    // Insert 
    unsigned int temp = size - v.size();
    for(unsigned int i = 0; i < temp; i++){
      insert(v, p);
    }
    return;
  }
  else if(size < v.size()){
    // remove
    unsigned int temp = v.size()-size;
    for(unsigned int i = 0; i < temp; i++){
      rm(v, p);
    }
    return;
  }
  else{return;}
  return;
}
void setSizeRand(std::vector<std::vector<DTYPE>> &v, unsigned int size, unsigned int p /*0*/){
  if(size > v.size()){
    // Insert 
    unsigned int temp = size - v.size();
    for(unsigned int i = 0; i < temp; i++){
      insertRand(v, p);
    }
    return;
  }
  else if(size < v.size()){
    // remove
    unsigned int temp = v.size()-size;
    for(unsigned int i = 0; i < temp; i++){
      rm(v, p);
    }
    return;
  }
  else{return;}
  return;
}

/// Insert Functions
void insert(std::vector<unsigned int> &v, unsigned int p /*0*/){
  if(p == 0){
    unsigned int temp = v[v.size()-2];
    v.insert(v.end()-1, temp);
  }else{
    unsigned int temp = v[p];
    v.insert(v.begin()+p, temp);
  }
  return;
}
void insert(std::vector<DTYPE> &v, unsigned int p /*0*/){
  if(p == 0){
    DTYPE temp = v[v.size()-2];
    v.insert(v.end()-1, temp);
  }else{
    DTYPE temp = v[p];
    v.insert(v.begin()+p, temp);
  }
  return;
}
void insert(std::vector<std::vector<DTYPE>> &v, unsigned int p /*0*/){
  if(p == 0){
    std::vector<DTYPE> temp = v[v.size()-2];
    v.insert(v.end()-1, temp);
  }else{
    std::vector<DTYPE> temp = v[p];
    v.insert(v.begin()+p, temp);
  }
  return;
}
void insertRand(
  std::vector<DTYPE> &v, 
  unsigned int p /*0*/, 
  DTYPE ll /*0*/, 
  DTYPE ul /*1*/
){
  if(p == 0){
    DTYPE temp = rng(ll, ul);
    v.insert(v.end()-1, temp);
  }else{
    DTYPE temp = rng(ll, ul);;
    v.insert(v.begin()+p, temp);
  }
  return;
}
void insertRand(
  std::vector<std::vector<DTYPE>> &v,
  unsigned int p /*0*/,
  DTYPE ll /*0*/, 
  DTYPE ul /*1*/
){
  if(p == 0){
    std::vector<DTYPE> temp = rng(v[v.size()-1].size(), ll, ul);
    v.insert(v.end()-1, temp);
  }else{
    std::vector<DTYPE> temp = rng(v[p].size(), ll, ul);
    v.insert(v.begin()+p, temp);
  }
  return;
}

/// Remove Functions
void rm(std::vector<unsigned int> &v, unsigned int p /*0*/){
  if(p == 0){
    v.erase(v.end()-2);
  }else{
    v.erase(v.begin()+p);
  }
  return;
}
void rm(std::vector<DTYPE> &v, unsigned int p /*0*/){
  if(p == 0){
    v.erase(v.end()-2);
  }else{
    v.erase(v.begin()+p);
  }
  return;
}
void rm(std::vector<std::vector<DTYPE>> &v, unsigned int p /*0*/){
  if(p == 0){
    v.erase(v.end()-2);
  }else{
    v.erase(v.begin()+p);
  }
  return;
}

/// Set Functions
void set(
  std::vector<DTYPE> &A,
  std::vector<DTYPE> B,
  unsigned int idx
){
  if(A.size() < idx+B.size()){
    errPrint("ERROR set: A.size() < idx+B.size().");
    std::cout << A.size() << ":" << idx * B.size() << std::endl;
    return;
  } 
  for(unsigned int i = 0; i < B.size(); i++){
    A[i+idx] = B[i];
  }
  return;
}

void set(
  std::vector<DTYPE> &A,
  std::vector<DTYPE> B,
  unsigned int idx,
  unsigned int size
){
  if(A.size() < idx+size){
    errPrint("ERROR set: A.size() < idx+B.size().");
    std::cout << A.size() << ":" << idx * size << std::endl;
    return;
  } 
  for(unsigned int i = 0; i < size; i++){
    A[i+idx] = B[i];
  }
  return;
}

void set(
  std::vector<unsigned int> &v,
  unsigned int x,
  unsigned int idx,
  unsigned int size
){
  if(v.size() < idx+size){
    errPrint("ERROR set: A.size() < idx+B.size().");
    std::cout << v.size() << ":" << idx * size << std::endl;
    return;
  } 
  for(unsigned int i = 0; i < size; i++){
    v[i+idx] = x;
  }
  return;
}
///////////////////////////////////////////////////////////////////////////////
/// Read Functions
///////////////////////////////////////////////////////////////////////////////
int getSetup(
  struct Alpha &alpha, 
  struct ANN_Ambit &annbit, 
  struct Read_Ambit &read, 
  int numInputs, 
  char * inputs[]
){
  // Check that the correct number of inputs is given
  if(numInputs < 2){
    errPrint("ERROR - main input: missing data filepath.");
    return 1;
  }
  // Print the inputs
  for(unsigned int i = 0; i < numInputs; i++){
    std::cout << inputs[i] << " ";
  }
  std::cout << std::endl;


  if(numInputs > 2){
    for(unsigned int i = 2; i < numInputs; i++){
      if(match(inputs[i], "LogPath")){
        annbit.logpath = inputs[++i];
      }
      // else if(match(inputs[i], "Analyze")){
      //   read.analyze = true;
      // }
      
      else if(match(inputs[i], "ID_column")){
        i++;
        read.idp = std::stoi(inputs[i]);
      }else if(match(inputs[i], "skip_row")){
        read.skipRow = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "skip_column")){
        read.skipCol = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "ratio")){
        read.ratio[0] = std::stod(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "sseed")){
        read.sseed[0] = std::stoi(inputs[i+1]);
        i++;
      }

      else if(match(inputs[i],"Adam")){
        alpha.adam = true;
      }else if(match(inputs[i], "alpha")){
        alpha.alpha = std::stod(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "beta")){
        alpha.beta1 = std::stod(inputs[i+1]);
        alpha.beta2 = std::stod(inputs[i+2]);
        i += 2;
      }
      
      else if(match(inputs[i],"maxIter")){
        annbit.maxIter = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "wseed")){
        annbit.wseed = std::stoi(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "Layers")){
        annbit.nLayers = std::stoi(inputs[i+1]);
        i++;
        annbit.hNodes[0] = std::stoi(inputs[i+1]);
        for(unsigned int j = 1; j < annbit.nLayers-2; j++){
          annbit.hNodes.push_back(std::stoi(inputs[i+1]));
        }
        i++;
      }else if(match(inputs[i], "hNodes")){
        annbit.nLayers = std::stoi(inputs[++i])+2;
        annbit.hNodes[0] = std::stoi(inputs[++i]);
        for(unsigned int j = 1; j < annbit.nLayers-2; j++){
          annbit.hNodes.push_back(std::stoi(inputs[++i]));
        }
        BUG(
          print("Hidden Nodes", nHiddenNodes);
          std::cout << std::endl;
        )
        std::cout << "Next Input: " << inputs[i+1] << std::endl;
      }

      else if(match(inputs[i], "aseed")){
        annbit.aseed = std::stoi(inputs[++i]);
        read.diversify = true;
        if(match(inputs[++i], "list:")){
          std::vector<unsigned int> actList;
          while(!match(inputs[++i], ":list")){
            actList.push_back(std::stoi(inputs[++i]));
            if(i >= numInputs){
              errPrint("ERROR - Setup: list: ... :list");
              return 1;
            }
          }
          std::sort(actList.begin(), actList.end()); 
          annbit.actList = actList;
        }
      }else if(match(inputs[i], "set_actDefault")){
        annbit.actDefault = std::stoi(inputs[++i]);
      }else if(match(inputs[i], "set_actNodes")){
        BUG(
          std::cout << "Setting Nodes" << std::endl;
          print(inputs[i+1], "0");
        )
        i++;
        unsigned int actID = std::stoi(inputs[i]);
        i++;
        std::vector<unsigned int> nodePos;
        if(match(inputs[i],"list:")){
          i++;
          while(!match(inputs[i], ":list")){
            nodePos.push_back(std::stoi(inputs[i]));
            i++;
            if(i >= numInputs){
              if(COLOR){
                errPrint("ERROR - Setup:" + rgb::b + "list:" + rgb::R + "was not followed by" + rgb::b + ":list." + rgb::R);
              }else{
                errPrint("ERROR - Setup: list: was not followed by :list.");
              }
              return 1;
            }
          }
        }else{
          unsigned int nodeStrt = std::stoi(inputs[i]);
          i++;
          unsigned int for_nNodes = std::stoi(inputs[i]);
          for(unsigned int j = 0; j < for_nNodes; j++){
            nodePos.push_back(nodeStrt+j);
          }
        }
        struct ActID_Set tempStruct(actID, nodePos);
        annbit.ActIDSets.push_back(tempStruct);
      }else if(match(inputs[i], "set_actLayer")){
        unsigned int ID = std::stoi(inputs[++i]);
        std::vector<unsigned int> layers 
          = std::vector<unsigned int>(1, std::stoi(inputs[++i]));
        BUG(
          print(ID, "set_actLayer", false);
          print(layers);
        )
        struct ActID_Set temp(ID, layers, 1);
        annbit.ActIDSets.push_back(temp);
      }
      else if(match(inputs[i], "set_actLayers")){
        std::cout << inputs[i] << std::endl << std::flush;
        i++;
        unsigned int actID = std::stoi(inputs[i]);
        std::cout << inputs[i] << std::endl << std::flush;
        // print(actID, "set_actLayers", false);
        i++;
        std::vector<unsigned int> layers;
        if(match(inputs[i],"list:")){
          i++;
          while(!match(inputs[i],":list")){
            layers.push_back(std::stoi(inputs[i]));
            print(inputs[i], "list", false);
            i++;
            if(i >= numInputs){
              errPrint("ERROR - Setup: list: was not followed by :list.");
              return 1;
            }
          }
          std::cout << ":list" << std::endl << std::flush;
        }else{
          unsigned int s_lyr = std::stoi(inputs[i]);
          unsigned int e_lyr = std::stoi(inputs[++i]);
          unsigned int size = e_lyr-s_lyr;
          layers = count(size, s_lyr);
        }
        
        struct ActID_Set temp(actID, layers, 1);
        annbit.ActIDSets.push_back(temp);
      }
      
      

      else{
        std::string msg = "ERROR - main input: input[" + str(inputs[i]) + "], not found.";
        errPrint(msg);
      }
    }
  }

  return 0;
}

/// Files
int readFile(std::vector<std::vector<std::string>> &content, std::string filename){
  // Creating Vector to hold Content
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
    errPrint("ERROR - data.readFile: Could not open file " + filename);
    return 1;
  }
  file.close();

  return 0;
}

int getData(struct Data &data, struct Read_Ambit read){
  // Get Data
  std::vector<std::vector<std::string>> dataS;
  if(readFile(dataS, read.filepath)){return 1;}

  // Unpack
  unsigned int sRow = read.skipRow;
  unsigned int sCol = read.skipCol;
  unsigned int idp = read.idp;

  unsigned int nFeat = dataS[sRow].size()-sCol-1;
  unsigned int nSamples = dataS.size()-sRow;
  unsigned int nClasses;
  std::vector<DTYPE> feat;
  std::vector<unsigned int> obs;

  // Get observation and feature vectors
  for(unsigned int i = sRow; i < dataS.size(); i++){ // for each row 
    for(unsigned int j =  sCol; j < dataS[i].size(); j++){
      if(j == idp){
        obs.push_back(stoi(dataS[i][j]));
      }else{
        feat.push_back((DTYPE)stod(dataS[i][j]));
      }
    }
  }
  // Get number of classes 
  nClasses = max(obs)+1;

  // Check for errors
  if(nFeat != feat.size()/nSamples){
    
    errPrint(
      "ERROR - getData: Number of features does not equal size of feature set divided by number of samples. ",
      nFeat, feat.size()/nSamples
    );
    return 1;
  }
  if(nSamples != obs.size()){
    errPrint(
      "ERROR - getData: Number of Samples does not match number of Observations.",
      nSamples, obs.size()
    );
    return 1;
  }

  // Pack
  data.nFeat = nFeat;
  data.nClasses = nClasses;
  data.nSamp = nSamples;
  data.feat = feat;
  data.obs = obs;
  data.sseed = read.sseed[0];
  data.ratio = read.ratio[0];

  return 0;
}

