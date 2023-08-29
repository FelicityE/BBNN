#include "data.h"


bool fExist(const std::string& name){
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0); 
}

void readFile(string filename, vector<vector<string>> &data){
  // Creating Vector to hold Content
  std::vector<std::vector<std::string>> content;
  std::vector<std::string> row;
  std::string line;
  std::string word;
  std::fstream file(filename, std::ios::in);

  // Opening File
  if(file.is_open()){
    while(getline(file, line)){
      row.clear();
      
      rmSpace(line); // Removes excess space
      std::stringstream str(line);
      
      while(getline(str, word, ' ')){
        
        row.push_back(word);
      }
      content.push_back(row);
    }
  }else{
    cout 
      << "ERROR - data.readFile: Could not open file " << filename << endl;
    return;
  }

  data = content;
}

char*** readFile(string filename, char*** data){
  // Read file to vector 
  vector<vector<string>> vvs;
  readFile(filename, vvs);

  char*** temp;
  temp = new char**[vvs.size()];
  for(unsigned int i = 0; i < vvs.size(); i++){
    temp[i] = new char*[vvs[i].size()];
    for(unsigned int j = 0; j < vvs[i].size(); j++){
      temp[i][j] = new char[vvs[i][j].length()];
      for(unsigned int k = 0; k < vvs[i][j].length(); k++){
        temp[i][j][k] = vvs[i][j][k];
      }
    }
  }

  return temp;
}

void rmSpace(string &str){
    bool start = false;
  string s = "";
  for(int i = 0; i < str.length(); i++){
    if(str[i] == ' '){
      continue;
    }
    s += str[i];
  }
  str=s;
}

void printData(vector<vector<string>> data){
  for(unsigned int  i = 0; i < data.size(); i++){
    for(unsigned int j = 0; j < data[i].size(); j++){
      cout << data[i][j] << ", ";
    }
    cout << endl;
  }
}

void printData(char*** data){
  cout << data << endl;
}