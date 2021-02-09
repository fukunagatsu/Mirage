#include "mirage_parameters.h"
#include <getopt.h>
#include <stdlib.h>

void MirageParameters::SetTrainingParameters(int argc,char* argv[]){
  int c;
  extern char *optarg;
  string input_file_name;
  
  while ((c = getopt(argc, argv, "i:o:l:m:n:k:")) != -1) {
    switch (c) {
    case 'i':
      input_file_name = optarg;
      break;

    case 'o':
      _output_file_name = optarg;
      break;

    case 'l':
      _max_number = atoi(optarg);
      _dim = _max_number+1;
      break;

    case 'm':
      _model_id = atoi(optarg);
      if(_model_id < 0 || _model_id > 3){
	cout << "Error: Invalid model id" << endl;
	exit(1);
      }
      break;

    case 'n':
      _mixture_method_id = atoi(optarg);
      if(_mixture_method_id < 0 || _mixture_method_id > 2){
	cout << "Error: Invalid model id" << endl;
	exit(1);
      }
      break;
      
    case 'k':
      _number_of_mixtures = atoi(optarg);
      break;      

    default:
      cerr << "The argument is an invalid command." << endl;
      exit(1); 
    }      
  }

  ReadData(input_file_name);
  ParameterInitialization();
}

void MirageParameters::SetEstimationParameters(int argc,char* argv[]){
  int c;
  extern char *optarg;
  string input_file_name;
  string input_parameter_file_name;
  
  while ((c = getopt(argc, argv, "i:p:o:")) != -1) {
    switch (c) {
    case 'i':
      input_file_name = optarg;
      break;

    case 'p':
      input_parameter_file_name = optarg;
      break;

    case 'o':
      _output_file_name = optarg;
      break;

    default:
      cerr << "The argument is an invalid command." << endl;
      exit(1); 
    }      
  }

  ReadParameter(input_parameter_file_name);
  ReadData(input_file_name);  
}

void MirageParameters::SetAlpha(int i, double alpha){
  _alpha[i] = alpha;
}

void MirageParameters::SetBeta(int i, double beta){
  _beta[i] = beta;
}

void MirageParameters::SetGamma(int i, double gamma){
  _gamma[i] = gamma;
}

void MirageParameters::SetParameter(int i, int j, double value){
  _parameter[i][j] = value;
}

void MirageParameters::SetMixtureProbability(int i, double value){
  _mixture_probability[i] = value;
}

void MirageParameters::SetRateParameter(int i, double value){
  _rate_parameter[i] = value;
}

void MirageParameters::SetInitProb(int i, int j, double value){
  _init_prob[i](j) = value;
}

void MirageParameters::SetGammaDistributionParameter(double value){
  _gamma_distribution_parameter = value;
}

int MirageParameters::GetLoopMax(void){
  return _loop_max;
}

double MirageParameters::GetLoopThreshold(void){
  return _loop_threshold;
}

double MirageParameters::GetGammaDistributionParameter(void){
  return _gamma_distribution_parameter;
}

int MirageParameters::GetMaxNumber(void){
  return _max_number;
}

int MirageParameters::GetModelID(void){
  return _model_id;
}

int MirageParameters::GetMixtureMethodID(void){
  return _mixture_method_id;
}

int MirageParameters::GetNumberOfMixtures(void){
  return _number_of_mixtures;
}

int MirageParameters::GetNumberOfSamples(void){
  return _number_of_samples;
}

double MirageParameters::GetInitProb(int i, int j){
  return _init_prob[i](j);
}


string MirageParameters::GetOutputFileName(void){
  return _output_file_name;
}

Node* MirageParameters::GetRoot(void){
  return _root;
}

int MirageParameters::GetDim(void){
  return _dim;
}

double MirageParameters::GetInitGradWeight(void){
  return _init_grad_weight;
}

double MirageParameters::GetPartialParThreshold(void){
  return _partial_par_threshold;
}

double MirageParameters::GetWeightDecayRate(void){
  return _weight_decay_rate;
};

double MirageParameters::GetWeightThreshold(void){
  return _weight_threshold;
}

vector<MatrixXd> MirageParameters::GetSubstitutionRateMatrix(void){
  return _substitution_rate_matrix;
}

double MirageParameters::GetMixtureProbability(int i){
  return _mixture_probability[i];
}

double MirageParameters::GetRateParameter(int i){
  return _rate_parameter[i];
}

double MirageParameters::GetAlpha(int i){
  return _alpha[i];
}
double MirageParameters::GetBeta(int i){
  return _beta[i];
}
double MirageParameters::GetGamma(int i){
  return _gamma[i];
}
double MirageParameters::GetParameter(int i, int j){
  return _parameter[i][j];
}

double MirageParameters::GetGammaRate(int i, int j){
  return _gamma_rate[i][j];
}

void MirageParameters::CalcAllGammaRate(){
  _gamma_rate.resize(1000, vector<double>(_number_of_mixtures, 0.0));
  
  for(int i = 1; i <= 1000; i++){
    vector<double> temp_vector = CalcGammaRate(i*0.01);
    for(int j =0; j < _number_of_mixtures; j++){
      _gamma_rate[i-1][j] = temp_vector[j];
    }
  }
}

vector<double> MirageParameters::CalcGammaRate(double g){
  random_device rnd;
  mt19937 mt(rnd());
  vector<double> rate_parameter; rate_parameter.resize(_number_of_mixtures);
  
  gamma_distribution<> gamma_dist(g, 1.0/g);
  vector<double> sample_vector;
  for (int i = 0; i < _gamma_dist_sample_size; ++i) {
    sample_vector.push_back(gamma_dist(mt));
  }
  sort(sample_vector.begin(), sample_vector.end());

  int div = _gamma_dist_sample_size/_number_of_mixtures;
  for(int i = 0; i < _number_of_mixtures; i++){
    double sum = 0;
    for(int j = div*i; j < div*(i+1); j++){
      sum  += sample_vector[j];
    }
    rate_parameter[i] = sum/(double)div;
  }
  return(rate_parameter);
}

void MirageParameters::ReadParameter(string file_name){
  ifstream fp;
  fp.open(file_name.c_str(), ios::in);
  if (!fp) {
    cout << "Cannot open " + file_name << endl;
    exit(1);
  }
  fp >> _max_number;
  _dim = _max_number+1;  
  fp >> _model_id;
  fp >> _mixture_method_id;
  fp >> _number_of_mixtures;
  
  int number_of_raw_matricies = _mixture_method_id == 0 ? _number_of_mixtures : 1;  
  _mixture_probability.resize(_number_of_mixtures, 0.0);
  _init_prob.resize(number_of_raw_matricies, VectorXd::Zero(_dim));

  if(_model_id == 0){
    _parameter.resize(number_of_raw_matricies, vector<double>((_dim - 1)*2, 0.0));
  }else if(_model_id == 1){
    _alpha.resize(number_of_raw_matricies, 0.0);
    _beta.resize(number_of_raw_matricies, 0.0);
    
  }else if(_model_id == 2 || _model_id == 3){
    _alpha.resize(number_of_raw_matricies, 0.0);
    _beta.resize(number_of_raw_matricies, 0.0);
    _gamma.resize(number_of_raw_matricies, 0.0);      
  }

  if(_mixture_method_id <= 1){    
    for(int i = 0; i < _number_of_mixtures; i++){
      fp >> _mixture_probability[i];
    }
  }else{
    for(int i = 0; i < _number_of_mixtures; i++){
      _mixture_probability[i] = 1.0/_number_of_mixtures;
    }
  }

  if(_mixture_method_id == 1){
    _rate_parameter.resize(_number_of_mixtures, 0.0);
    for(int i = 0; i < _number_of_mixtures; i++){
      fp >> _rate_parameter[i];
    }
  }else if(_mixture_method_id == 2){
    _rate_parameter.resize(_number_of_mixtures, 0.0);
    fp >> _gamma_distribution_parameter;
    vector<double> rate_vector = CalcGammaRate(_gamma_distribution_parameter);
    for(int i = 0; i < _number_of_mixtures; i++){
      _rate_parameter[i] = rate_vector[i];
    }
  }

  for(int i = 0; i < number_of_raw_matricies; i++){
    for(int j = 0; j < _dim; j++){
      fp >> _init_prob[i](j);
    }
    if(_model_id == 1){
      fp >> _alpha[i] >> _beta[i];
    }else if(_model_id == 2 || _model_id == 3){
      fp >> _alpha[i] >> _beta[i] >> _gamma[i];;
    }else{
      for(int j = 0; j < (_dim - 1)*2; j++){
	fp >> _parameter[i][j];
      }
    }
  }
  
  _substitution_rate_matrix.resize(_number_of_mixtures, MatrixXd::Zero(_dim,_dim));
  SetSubstitutionRateMatrix();
}

void MirageParameters::SetSubstitutionRateMatrix(){
  for(int i = 0; i < _number_of_mixtures; i++){
    _substitution_rate_matrix[i] = MatrixXd::Zero(_dim,_dim);
    if(_model_id == 1){
      if(_mixture_method_id == 0){
	for(int j = 0; j < _dim-1; j++){	
	  _substitution_rate_matrix[i](j, j+1) = _alpha[i];
	  _substitution_rate_matrix[i](j+1, j) = _beta[i];
	}
      }else{
	for(int j = 0; j < _dim-1; j++){	
	  _substitution_rate_matrix[i](j, j+1) = _rate_parameter[i]*_alpha[0];
	  _substitution_rate_matrix[i](j+1, j) = _rate_parameter[i]*_beta[0];
	}
      }
    }else if(_model_id == 2){
      if(_mixture_method_id == 0){
	for(int j = 0; j < _dim-1; j++){	
	  _substitution_rate_matrix[i](j, j+1) += _alpha[i]+j*_gamma[i];
	  _substitution_rate_matrix[i](j+1, j) += (j+1)*_beta[i];
	}
      }else{
	for(int j = 0; j < _dim-1; j++){	
	  _substitution_rate_matrix[i](j, j+1) += _rate_parameter[i]*(_alpha[0]+j*_gamma[0]);
	  _substitution_rate_matrix[i](j+1, j) += _rate_parameter[i]*(j+1)*_beta[0];
	}
      }
    }else if(_model_id == 0){
      if(_mixture_method_id == 0){
	for(int j = 0; j < _dim-1; j++){	  
	  _substitution_rate_matrix[i](j, j+1) += _parameter[i][j];
	  _substitution_rate_matrix[i](j+1, j) += _parameter[i][_dim-1+j];
	}
      }else{
	for(int j = 0; j < _dim-1; j++){	  
	  _substitution_rate_matrix[i](j, j+1) += _rate_parameter[i]*_parameter[0][j];
	  _substitution_rate_matrix[i](j+1, j) += _rate_parameter[i]*_parameter[0][_dim-1+j];
	}
      }
    }else{
      if(_mixture_method_id == 0){
	_substitution_rate_matrix[i](0, 1) = _gamma[i];
	_substitution_rate_matrix[i](1, 0) = _beta[i];
	for(int j = 1; j < _dim-1; j++){
	  _substitution_rate_matrix[i](j, j+1) = _alpha[i];
	  _substitution_rate_matrix[i](j+1, j) = _beta[i];
	}
      }else{
	_substitution_rate_matrix[i](0, 1) = _rate_parameter[i]*_gamma[0];
	_substitution_rate_matrix[i](1, 0) = _rate_parameter[i]*_beta[0];
	for(int j = 1; j < _dim-1; j++){
	  _substitution_rate_matrix[i](j, j+1) = _rate_parameter[i]*_alpha[0];
	  _substitution_rate_matrix[i](j+1, j) = _rate_parameter[i]*_beta[0];
	}
      }
    }
    

    for(int j = 0; j < _dim ; j++){
      if(j != 0){
	_substitution_rate_matrix[i](j, j) += -_substitution_rate_matrix[i](j, j-1);
      }
      if(j != _dim-1){
	_substitution_rate_matrix[i](j, j) += -_substitution_rate_matrix[i](j, j+1);
      }
    }
  }
}

void MirageParameters::ReadData(string file_name){
  ifstream fp;
  fp.open(file_name.c_str(), ios::in);
  if (!fp) {
    cout << "Cannot open " + file_name << endl;
    exit(1);
  }
  fp >> _number_of_samples;  
  string newick;
  fp >> newick;

  _root = MakeNode();
  vector<Node*> leaf_list;
  ParseTree(newick, leaf_list);
  int number_of_species = leaf_list.size();
  vector<string> name_list; name_list.resize(number_of_species, "");
  vector<int> corresponding_table; corresponding_table.resize(number_of_species, 0);
  string s_temp;
  fp >> s_temp;
  for(int i = 0; i < number_of_species; i++){
    fp >> name_list[i];
  }
  for(int i = 0; i < number_of_species; i++){
    for(int j = 0; j < number_of_species; j++){
      if(name_list[i] == leaf_list[j]->name){
	corresponding_table[i] = j;
	break;
      }
    }
  }
  
  for(int i = 0; i < _number_of_samples; i++){
    fp >> s_temp;
    for(int j = 0; j < number_of_species; j++){
      int temp;
      fp >> temp;
      if(temp > _max_number){
	temp = _max_number;
      }
      leaf_list[corresponding_table[j]]->reconstruction[i] = temp;
      for(int k = 0; k < _number_of_mixtures; k++){
	leaf_list[corresponding_table[j]]->inside_values[GetTripleArrayId(i,k,temp)] = 0.0;
      }
    }
  }
  fp.close();
}

int MirageParameters::GetTripleArrayId(int sample_id, int mixture_id, int element_id){
  return(sample_id*_number_of_mixtures*_dim + mixture_id*_dim + element_id);
}

Node* MirageParameters::MakeNode(void){
  Node* temp = (Node*)malloc(sizeof(Node));
  temp->edge_length = 0;
  temp->inside_values = (double*)malloc(sizeof(double)*_dim*_number_of_mixtures*_number_of_samples);
  temp->outside_values = (double*)malloc(sizeof(double)*_dim*_number_of_mixtures*_number_of_samples);
  temp->fd_values = (double*)malloc(sizeof(double)*_dim*_number_of_mixtures*_number_of_samples);
  temp->ns_values = (double*)malloc(sizeof(double)*_dim*(_dim-1)*_number_of_mixtures*_number_of_samples);
  temp->logL = (double*)malloc(sizeof(double)*_dim*_number_of_mixtures*_number_of_samples);
  temp->c = (int*)malloc(sizeof(int)*_dim*_number_of_mixtures*_number_of_samples);
  temp->reconstruction = (int*)malloc(sizeof(int)*_number_of_samples);

  
  for(int i = 0; i < _dim*_number_of_samples*_number_of_mixtures; i++){
    temp->inside_values[i] = -DBL_MAX;
    temp->outside_values[i] = -DBL_MAX;
    temp->fd_values[i] = 0;
    temp->logL[i] = -DBL_MAX;
    temp->c[i] = -1;
  }
  for(int i = 0; i < _dim*(_dim-1)*_number_of_samples*_number_of_mixtures; i++){
    temp->ns_values[i] = 0;
  }

  for(int i = 0; i < _number_of_samples; i++){
    temp->reconstruction[i] = -1;
  }
  
  temp->parent = NULL;
  temp->right = NULL;
  temp->left = NULL;
  return(temp);
}

string MirageParameters::ParseName(string& newick, int& i){
  string temp_name = "";
  while(newick[i] != '(' && newick[i] != ')' && newick[i] != ',' && newick[i] != ':'){
    temp_name += newick[i];
    i++;
  }
  i--;
  return temp_name;
}

void MirageParameters::ParseTree(string& newick, vector<Node*>& leaf_list){  
  Node* current = _root;
  for(int i = 0; i < newick.size()-1;i++){
    if(newick[i] == '('){
      current->left = MakeNode();
      current->left->parent = current;
      current = current->left;
    }else if(newick[i] == ')'){
      current = current->parent;
    }else if(newick[i] == ','){
      current = current->parent;
      current->right = MakeNode();
      current->right->parent = current;
      current = current->right;
    }else if(newick[i] == ':'){
      i++;
      current->edge_length = stof(ParseName(newick,i));
    }else{
      string temp_name = ParseName(newick,i);
      current->name = (char*)malloc(sizeof(char)*(temp_name.size()+1));
      strcpy(current->name, temp_name.c_str());
      leaf_list.push_back(current);
    }
  }
}

void MirageParameters::ParameterInitialization(){
  
  _mixture_probability.resize(_number_of_mixtures, 0.0);  
  int number_of_raw_matricies = _mixture_method_id == 0 ? _number_of_mixtures : 1;
  _init_prob.resize(number_of_raw_matricies, VectorXd::Zero(_dim));

  _substitution_rate_matrix.resize(_number_of_mixtures, MatrixXd::Zero(_dim,_dim));
  random_device rnd;
  int temp = rnd();
  mt19937 mt(temp);

  if(_mixture_method_id <= 1){
    int sum = 0;
    for(int i = 0; i < _number_of_mixtures; i++){
      _mixture_probability[i] = mt()%1000+1;    
      sum +=  _mixture_probability[i];
    }
    for(int i = 0; i < _number_of_mixtures; i++){
      _mixture_probability[i] /= sum;
    }
  }else{
    for(int i = 0; i < _number_of_mixtures; i++){
      _mixture_probability[i] = 1.0/_number_of_mixtures;
    }
  }
  
  for(int i = 0; i < number_of_raw_matricies; i++){
    int sum = 0;
    for(int j = 0; j < _dim; j++){
      _init_prob[i](j) = mt()%1000+1;
      sum +=  _init_prob[i](j);
    }
    for(int j = 0; j < _dim; j++){
      _init_prob[i](j) /=sum;
      
    }
  }
  
  if(_mixture_method_id >= 1){
    _max_init = 1.5;
  }
  uniform_real_distribution<double> uniform_distribution(_min_init, _max_init);
  
  if(_model_id == 1){
    _alpha.resize(number_of_raw_matricies, 0.0);
    _beta.resize(number_of_raw_matricies, 0.0);
    
    for(int i = 0; i < number_of_raw_matricies; i++){
      _alpha[i] = uniform_distribution(mt);
      _beta[i] = uniform_distribution(mt);
    }
    
  }else if(_model_id == 2 || _model_id == 3){
    _alpha.resize(number_of_raw_matricies, 0.0);
    _beta.resize(number_of_raw_matricies, 0.0);
    _gamma.resize(number_of_raw_matricies, 0.0);
    
    for(int i = 0; i < number_of_raw_matricies; i++){
      _alpha[i] = uniform_distribution(mt);
      _beta[i] = uniform_distribution(mt);
      _gamma[i] = uniform_distribution(mt);
    }
  }else{
    _parameter.resize(number_of_raw_matricies, vector<double>((_dim - 1)*2, 0.0));
    for(int i = 0; i < number_of_raw_matricies; i++){
      for(int j = 0; j < (_dim - 1)*2 ; j++){
	_parameter[i][j] = uniform_distribution(mt);
      }
    }
  }

  if(_mixture_method_id ==1){
    _rate_parameter.resize(_number_of_mixtures, 0.0);
    uniform_real_distribution<double> uniform_distribution_r(1.0, _max_rate);
    double rate_sum = 0.0;
    for(int i = 0; i < _number_of_mixtures; i++){
      _rate_parameter[i] =  uniform_distribution_r(mt);
      rate_sum += _rate_parameter[i] * _mixture_probability[i];
    }
    for(int i = 0; i < _number_of_mixtures; i++){
      _rate_parameter[i] /=  rate_sum;
    }
  }else if(_mixture_method_id == 2){
    _rate_parameter.resize(_number_of_mixtures, 0.0);
    CalcAllGammaRate();
    for(int i = 0; i < _number_of_mixtures; i++){
      _rate_parameter[i] = _gamma_rate[99][i];
    }

  }
  SetSubstitutionRateMatrix();
}
