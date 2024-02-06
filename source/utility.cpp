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
void print(std::vector<unsigned int> v){
  for(unsigned int i = 0; i < v.size(); i ++){
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
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

int errPrint(std::string error_message){
  std::cout << RGB::r << error_message << RGB::R << std::endl;
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
    errPrint("ERROR - match: A and B are not the same size.");
    std::cout << A.size() << ":" << B.size() << std::endl;
    exit(1);
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

bool inVec(unsigned int v, std::vector<unsigned int> V){
  for(unsigned int i = 0; i < V.size(); i++){
    if(V[i] == v){return i;}
  }
  return 0;
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

///////////////////////////////////////////////////////////////////////////////
/// Vector Functions
///////////////////////////////////////////////////////////////////////////////
/// Sum Functions
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

/// Rand Functions
DTYPE rng(DTYPE ll /*0*/, DTYPE ul /*1*/){
  DTYPE rn = (DTYPE)rand()/RAND_MAX;
  if(ll != 0 || ul != 1){rn =  (ul - ll) * rn + ll;}
  return rn;
}
std::vector<DTYPE> rng(unsigned int size, DTYPE ll /*0*/, DTYPE ul /*1*/){
  std::vector<DTYPE> v;
  for(unsigned int i = 0; i < size; i++){
    DTYPE rn = rng(ll, ul);
    v.push_back(rn);
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
  std::vector<unsigned int> v;
  for(unsigned int i = 0; i < size; i++){
    unsigned int rn = rng(ll, ul);
    v.push_back(rn);
  }
  return v;
}
std::vector<unsigned int> rng_unq(unsigned int size, unsigned int ll, unsigned int ul){
  std::vector<unsigned int> v;
  for(unsigned int i = 0; i < size; i++){
    unsigned int rn = rng(ll, ul);
    while(std::find(v.begin(), v.end(), rn) != v.end()){rn = rng(ll, ul);}
    v.push_back(rn);
  }
  return v;
}
/// Size Functions
unsigned int size(std::vector<std::vector<DTYPE>> v){
  unsigned int temp = 0;
  for(unsigned int i = 0; i < v.size(); i++){
    temp += v[i].size();
  }
  return temp;
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

///////////////////////////////////////////////////////////////////////////////
/// Read Functions
///////////////////////////////////////////////////////////////////////////////
int getSetup(
  struct Adam &adam, 
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
      if(match(inputs[i], "ID_column")){
        read.idp = std::stoi(inputs[i+1]);
        i++;
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
        adam.adam = true;
      }else if(match(inputs[i], "alpha")){
        adam.alpha = std::stod(inputs[i+1]);
        i++;
      }else if(match(inputs[i], "beta")){
        adam.beta1 = std::stod(inputs[i+1]);
        adam.beta2 = std::stod(inputs[i+2]);
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
        annbit.nLayers = std::stoi(inputs[i+1])+2;
        std::cout << "\nNumber of Layers: " << annbit.nLayers << std::endl;
        i++;
        annbit.hNodes[0] = std::stoi(inputs[i+1]);
        i++;
        for(unsigned int j = 1; j < annbit.nLayers-2; j++){
          annbit.hNodes.push_back(std::stoi(inputs[i+1]));
          i++;
        }
        // print("Hidden Nodes", nHiddenNodes);
        std::cout << std::endl;
      }
      
      else if(match(inputs[i], "setActs")){
        i++;
        std::vector<unsigned int> temp;
        unsigned int cnt;
        while(!match(inputs[i], "-stp")){
          if(cnt >= 5){
            errPrint("ERROR - SetUp: setActs was not followed by -stp after 5 or less integers.");
            return 1;
          }
          temp.push_back(std::stoi(inputs[i]));
          i++;
          cnt++;
        }
        struct ActID_Set tempStruct(temp);
        annbit.ActIDSets.push_back(tempStruct);
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
  nClasses = max(obs);

  // Check for errors
  if(nFeat != feat.size()/nSamples){

    errPrint(
      "ERROR - getData: Number of features does not equal size of feature set divided by number of samples. "
    );
    std::cout << nFeat << ":" << feat.size()/nSamples << std::endl;
    return 1;
  }
  if(nSamples != obs.size()){
    errPrint(
      "ERROR - getData: Number of Samples does not match number of Observations."
    );
    std::cout << nSamples << ":" << obs.size() << std::endl;
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

