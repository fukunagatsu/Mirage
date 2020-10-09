#include "mirage.h"
#include "mirage_parameters.h"
#include <string>

void PrintUsage() {
  cout << "Mirage version 1.0 - Ancestral Genome Estimation based on a phylogenetic mixture model and a RER model." << endl;
  cout << "\n";
  cout << "Options\n";
  cout << "train: training of evolutionary model parameters from a phylogenetic tree and an ortholog table\n";
  cout << "\n";
  cout << "estimate: estimate gene content history from evolutionary model parameters\n";
 
}

int main(int argc, char* argv[]){
  if(argc == 1 || strcmp(argv[1],"-h") == 0){
    PrintUsage();
    exit(1);
  }
  
  MirageParameters mirage_parameters;
  Mirage mirage;
  
  if(strcmp(argv[1],"train") == 0){
    mirage_parameters.SetTrainingParameters(argc-1, argv+1);
    mirage.Train(mirage_parameters);
    
  }else if(strcmp(argv[1],"estimate") == 0){
    mirage_parameters.SetEstimationParameters(argc-1,argv+1);
    mirage.Estimate(mirage_parameters);
  }else{
    cerr << "Error: You must specify the mode of Mirage (train or estimate)." << endl;
    exit(1);
  }
  return(0);
}
