#include "mirage.h"
#include "fmath.hpp"
#include <unistd.h>

void Mirage::Train(MirageParameters& parameters){
  Initiallize(parameters);
  
  double old_log_likelihood = 0.0;
  double new_log_likelihood = 0.0;
  int count = 0;
  Parameter old_parameter;
  clock_t old = clock();
  while(true){
    cout << "loop:" << count << endl;
    count++;

    CalcInsideValues(parameters.GetRoot(), parameters);
    
    CalcColumnLogLikelihood(parameters.GetRoot(), parameters);

    new_log_likelihood = CalcDataLikelihood(parameters);
    cout << new_log_likelihood << endl;
    if(old_log_likelihood != 0.0){
      double value = new_log_likelihood - old_log_likelihood;
      if(count > parameters.GetLoopMax()  || new_log_likelihood - old_log_likelihood < parameters.GetLoopThreshold() ){
	_number_of_em_iteration = count-1;
	SetOldParameter(old_parameter,parameters);
	break;
      }
    }
    old_log_likelihood = new_log_likelihood;
    _log_likelihood_array.push_back(old_log_likelihood);
    
    CalcResponsibility(parameters);
    if(_mixture_method_id != 3){
      CalcOutsideValues(parameters.GetRoot(), parameters, -1);
    }else{
      CalcOutsideValuesDPM(parameters.GetRoot(), parameters, -1);
    }    

    vector<VectorXd> init(_number_of_mixtures, VectorXd::Zero(_dim));
    vector<VectorXd> fd(_number_of_mixtures, VectorXd::Zero(_dim));
    vector<MatrixXd> ns(_number_of_mixtures, MatrixXd::Zero(_dim, _dim));
    CalcTreeModelSufficientStatistics(parameters.GetRoot(), parameters, -1, init, fd, ns);
    SaveOldParameter(parameters,old_parameter);
    
    _estimation_flag = NewParameterEstimation(parameters.GetRoot(),parameters, init, fd, ns);
    if(!_estimation_flag){
      _number_of_em_iteration = count;
      break;
    }        
  }
  if(_mixture_method_id == 2){
    CalcGammaParameter(parameters);
    CalcInsideValues(parameters.GetRoot(), parameters);
    CalcColumnLogLikelihood(parameters.GetRoot(), parameters);
  }

  
  FreeAndMalloc(parameters.GetRoot());
  HistoryReconstruction(parameters.GetRoot(), parameters);
  Output(parameters,0);
}

void Mirage::FreeAndMalloc(Node* current){
  free(current->inside_cash_index);
  if(_mixture_method_id != 3){
    free(current->outside_cash_index);
  }
  free(current->inside_values);
  free(current->outside_values);
    
  current->c = (char*)malloc(sizeof(char)*_dim*_number_of_mixtures*_number_of_samples);
  for(int i = 0; i < _dim*_number_of_samples*_number_of_mixtures; i++){
    current->c[i] = -1;
  }
  
  if(current->left == NULL && current->right == NULL){   
    return;
  }
  current->reconstruction = (char*)malloc(sizeof(char)*_number_of_samples);
  for(int i = 0; i < _number_of_samples; i++){
    current->reconstruction[i] = -1;
  }
  
  if(current->left != NULL){
    FreeAndMalloc(current->left);
  }
  if(current->right != NULL){
    FreeAndMalloc(current->right);
  }
  return;  
}

void Mirage::Estimate(MirageParameters& parameters){
  Initiallize(parameters);  
  CalcInsideValues(parameters.GetRoot(), parameters);
  CalcColumnLogLikelihood(parameters.GetRoot(), parameters);
  CalcResponsibility(parameters);
 
  FreeAndMalloc(parameters.GetRoot());
  HistoryReconstruction(parameters.GetRoot(), parameters);
  Output(parameters,1);
}

void Mirage::OutputReconstruction(ofstream& ofs, Node* current, int output_style){
  if(output_style == 0){
    ofs << "Node " << current->node_id << " : ";
    for(int i = 0; i < _number_of_samples; i++){
      ofs << (int)current->reconstruction[i] << " ";
    }
    ofs << endl;
  }else if(output_style == 1){
    if(current->parent != NULL){
      ofs << "Edge " << current->parent->node_id << "->" << current->node_id << " : ";
      for(int i = 0; i < _number_of_samples; i++){
	ofs << (int)current->parent->reconstruction[i] - (int)current->reconstruction[i] << " ";
      }
      ofs << endl;
    }
  }
  return;
}

void Mirage::HistoryTraceBack(ofstream& ofs, Node* current, int output_style){
  if(current->left == NULL && current->right == NULL){
    OutputReconstruction(ofs,current, output_style);
    return;
  }
  if(current->parent != NULL){    
    for(int i = 0; i < _number_of_samples; i++){      
      current->reconstruction[i] = current->c[GetTripleArrayId(i,_mixture_id[i],current->parent->reconstruction[i])];
    }
  }
  OutputReconstruction(ofs,current,output_style);
  
  if(current->left != NULL){
    HistoryTraceBack(ofs,current->left,output_style);
  }
  if(current->right != NULL){
    HistoryTraceBack(ofs,current->right,output_style);
  }
  return;  
}

void Mirage::HistoryReconstruction(Node* current, MirageParameters& parameters){
  vector<MatrixXd> substitution_rate_matrix = parameters.GetSubstitutionRateMatrix();
  if(current->left == NULL && current->right == NULL){
    current->logL = (double*)malloc(sizeof(double)*_dim*_number_of_mixtures*_number_of_samples);
    for(int i = 0; i < _dim*_number_of_samples*_number_of_mixtures; i++){
      current->logL[i] = -DBL_MAX;
    }
    
    for(int i = 0; i < _number_of_mixtures; i++){
      MatrixXd substitution_probability_matrix = (current->edge_length * substitution_rate_matrix[i]).exp();
      AddEpsilon(substitution_probability_matrix);	
      for(int j = 0; j < _dim; j++){
	for(int k = 0; k < _dim; k++){
	  substitution_probability_matrix(j, k) = log(substitution_probability_matrix(j, k));
	}
      }
  
      for(int j = 0; j < _number_of_samples; j++){
	char state = current->reconstruction[j];
	for(int k = 0; k < _dim; k++){
	  current->c[GetTripleArrayId(j,i,k)] = state;
	  current->logL[GetTripleArrayId(j,i,k)] = substitution_probability_matrix(k,state);
	}
      }
    }
    return;
  }
  
  if(current->left != NULL){
    HistoryReconstruction(current->left, parameters);
  }
  if(current->right != NULL){
    HistoryReconstruction(current->right, parameters);
  }

  if(current->parent != NULL){
    current->logL = (double*)malloc(sizeof(double)*_dim*_number_of_mixtures*_number_of_samples);
    for(int i = 0; i < _dim*_number_of_samples*_number_of_mixtures; i++){
      current->logL[i] = -DBL_MAX;
    }
    
    for(int i = 0; i < _number_of_mixtures; i++){
      MatrixXd substitution_probability_matrix = (current->edge_length * substitution_rate_matrix[i]).exp();
      AddEpsilon(substitution_probability_matrix);
      for(int j = 0; j < _dim; j++){
	for(int k = 0; k < _dim; k++){
	  substitution_probability_matrix(j, k) = log(substitution_probability_matrix(j, k));
	}
      }
      
      for(int j = 0; j < _number_of_samples; j++){
	for(int k = 0; k < _dim; k++){
	  double temp_max = -DBL_MAX;
	  int temp_l = -1;
	  for(int l = 0; l < _dim; l++){
	    double temp = substitution_probability_matrix(k, l) +
	      current->left->logL[GetTripleArrayId(j,i,l)] + current->right->logL[GetTripleArrayId(j,i,l)];
	    if(temp > temp_max){
	      temp_max = temp;
	      temp_l = l;
	    }
	  }
	  current->logL[GetTripleArrayId(j,i,k)] = temp_max;
	  current->c[GetTripleArrayId(j,i,k)] = temp_l;	  
	}
      }
    }
    free(current->left->logL);
    free(current->right->logL);
  }else{
    for(int j = 0; j < _number_of_samples; j++){
      double max_logL = -DBL_MAX;
	
      for(int i = 0; i < _number_of_mixtures; i++){
	int temp_i = IsPatternMixture() ? i : 0;
	for(int k = 0; k < _dim; k++){
	  double temp = parameters.GetInitProb(temp_i,k) +
	    current->left->logL[GetTripleArrayId(j,i,k)] + current->right->logL[GetTripleArrayId(j,i,k)];
	  if(temp > max_logL){
	    max_logL = temp;
	    current->reconstruction[j] = k;
	    _mixture_id[j] = i;
	  }
	}
      }
    }
    free(current->left->logL);
    free(current->right->logL);
  }
  return;  
}

void Mirage::Output(MirageParameters& parameters, int id){
  string file_name = parameters.GetOutputFileName();
  if(id == 0){
    ofstream ofs_par((file_name+".par").c_str());
    ofs_par << _dim-1 << " " << parameters.GetModelID() << " " <<
      parameters.GetMixtureMethodID() << " " << _number_of_mixtures << endl;

    
    if(_mixture_method_id != 2){
      for(int i = 0; i < _number_of_mixtures; i++){
	ofs_par << parameters.GetMixtureProbability(i) << " ";
	if(_mixture_method_id== 3 && parameters.GetMixtureProbability(i) == 0.0){
	  _estimation_flag = false;
	  for(int j = 0; j < _dim; j++){
	    parameters.SetInitProb(i,j,NAN);
	  }      
	  if(_model_id == 1){
	    parameters.SetAlpha(i,NAN);
	    parameters.SetBeta(i, NAN);
	  }else if(_model_id == 2 || _model_id == 3){
	    parameters.SetAlpha(i, NAN);
	    parameters.SetBeta(i, NAN);
	    parameters.SetGamma(i, NAN);
	  }else{
	    for(int j = 0; j < (_dim - 1)*2 ; j++){
	      parameters.SetParameter(i,j,NAN);
	    }
	  }
	}
      } 
      ofs_par << endl;
    }

    if(_mixture_method_id == 1){
      for(int i = 0; i < _number_of_mixtures; i++){
	ofs_par << parameters.GetRateParameter(i) << " ";
      } 
      ofs_par << endl;
    }else if(_mixture_method_id == 2){
      ofs_par << parameters.GetGammaDistributionParameter() <<endl;
    }

    int number_of_raw_matricies = IsPatternMixture() ? _number_of_mixtures : 1;
    
    for(int i = 0; i < number_of_raw_matricies; i++){
      for(int j = 0; j < _dim; j++){
	ofs_par << parameters.GetInitProb(i,j) << " ";
      }
      ofs_par << endl;
      
      if(_model_id == 1){
	ofs_par << parameters.GetAlpha(i) << " " << parameters.GetBeta(i) << endl;
      }else if(_model_id == 2 || _model_id == 3){
	ofs_par << parameters.GetAlpha(i) << " " << parameters.GetBeta(i) << " " << parameters.GetGamma(i) << endl;
      }else{
	for(int j = 0; j < (_dim - 1)*2 ; j++){
	  ofs_par << parameters.GetParameter(i,j) << " ";
	}
	ofs_par << endl;
      }
    }
  }
  
  ofstream ofs_bas((file_name+".bas").c_str());
  ofs_bas << "Max_Number: " << _dim-1 << endl;
  ofs_bas << "Model_ID: " << parameters.GetModelID() << endl;
  ofs_bas << "Mixture_Method_ID: " << parameters.GetMixtureMethodID() << endl;
  ofs_bas << "Number_of_Mixtures: " << _number_of_mixtures << endl;

  string temp = _estimation_flag ? "success" : "failure";  
  ofs_bas << "Estimation: " << temp << endl;
  ofs_bas << "Log_Likelihood: " << CalcDataLikelihood(parameters) << endl;
  ofs_bas << "Random_Seed: " << parameters.GetSeed() << endl;
  ofs_bas << "Number_of_EM_iteration: " << _number_of_em_iteration << endl;

  temp = "Convergence_Process_of_Log_Likelihood: ";
  for(int i = 0; i < _log_likelihood_array.size(); i++){
    temp += to_string(_log_likelihood_array[i])+" ";
  }
  ofs_bas << temp << endl;
  ofs_bas.close();

  if(_mixture_method_id  != 2){
    ofstream ofs_res((file_name+".res").c_str());
    
    for(int i = 0; i < _number_of_samples; i++){
      for(int j = 0; j < _number_of_mixtures; j++){
	ofs_res<< _responsibility[i][j] << " ";
      }
      ofs_res << endl;
    }
    ofs_res.close();
  }
  
  ofstream ofs_hist((file_name+".ahr").c_str());
  int count = 0;  
  HistoryTraceBack(ofs_hist, parameters.GetRoot(), parameters.GetOutputStyle());
 
  ofs_hist.close();
}

bool Mirage::IsPatternMixture(){
  return(_mixture_method_id == 0 || _mixture_method_id == 3);
}

void Mirage::Initiallize(MirageParameters& parameters){
  _number_of_mixtures = parameters.GetNumberOfMixtures();
  _dim = parameters.GetDim();
  _number_of_samples = parameters.GetNumberOfSamples();
  _model_id = parameters.GetModelID();
  _mixture_method_id = parameters.GetMixtureMethodID();
  _column_log_likelihood.resize(_number_of_samples, vector<double>(_number_of_mixtures, 0.0));
  _init_prob_sufficient_statistics.resize(_number_of_samples, vector<vector<double> >(_number_of_mixtures, vector<double>(_dim, 0.0)));

  _responsibility.resize(_number_of_samples, vector<double>(_number_of_mixtures, 0.0));

  _mixture_id.resize(_number_of_samples, 0.0);
}

MatrixXd Mirage::SetTempSubstitutionRateMatrix(MirageParameters& parameters){  
  MatrixXd temp_rate_matrix = MatrixXd::Zero(_dim,_dim);
  if(_model_id == 1){
    for(int j = 0; j < _dim-1; j++){	
      temp_rate_matrix(j, j+1) = parameters.GetAlpha(0);
      temp_rate_matrix(j+1, j) = parameters.GetBeta(0);
    }
  }else if(_model_id == 2){
    for(int j = 0; j < _dim-1; j++){	
      temp_rate_matrix(j, j+1) += (parameters.GetAlpha(0)+j*parameters.GetGamma(0));
      temp_rate_matrix(j+1, j) += (j+1)*parameters.GetBeta(0);
    }
  }else if(_model_id == 0){
    for(int j = 0; j < _dim-1; j++){	  
      temp_rate_matrix(j, j+1) += parameters.GetParameter(0,j);
      temp_rate_matrix(j+1, j) += parameters.GetParameter(0,_dim-1+j);
    }
  }else{
    temp_rate_matrix(0, 1) = parameters.GetGamma(0);
    for(int j = 1; j < _dim; j++){
      if(j != _dim-1){
	temp_rate_matrix(j, j+1) = parameters.GetAlpha(0);
      }
      temp_rate_matrix(j, j-1) = parameters.GetBeta(0);
    }
  }
    
  for(int j = 0; j < _dim ; j++){
    if(j != 0){
      temp_rate_matrix(j, j) += -temp_rate_matrix(j, j-1);
    }
    if(j != _dim-1){
      temp_rate_matrix(j, j) += -temp_rate_matrix(j, j+1);
    }
  }
  return temp_rate_matrix;
}

void Mirage::CalcGammaParameter(MirageParameters& parameters){
  double temp_i = -1;
  double temp_sum = DBL_MAX;
  for(int i = 0; i < 1000; i++){
    double sum = 0.0;
    for(int j = 0; j < _number_of_mixtures; j++){
      double diff = parameters.GetRateParameter(j) - parameters.GetGammaRate(i, j);	
      sum += pow(diff, 2);
    }
    sum = sqrt(sum);
    if(sum < temp_sum){
      temp_i = i;
      temp_sum = sum;
    }
  }
  parameters.SetGammaDistributionParameter((temp_i+1)*0.01);
  for(int i = 0; i < _number_of_mixtures; i++){
    parameters.SetRateParameter(i, parameters.GetGammaRate(temp_i,i));
  }
  parameters.SetSubstitutionRateMatrix();
}

void Mirage::CalcRateParameter(vector<VectorXd>& fd, vector<MatrixXd>& ns, MirageParameters& parameters){
  
  vector<double> rate_vector; rate_vector.resize(_number_of_mixtures,0.0);
  double rate_sum = 0.0;
  for(int i = 0; i < _number_of_mixtures; i++){    
    MatrixXd rate_matrix = SetTempSubstitutionRateMatrix(parameters);
    double denominator = rate_matrix(0, 0) * fd[i](0);
    double numerator = 0.0;
    
    for(int j = 1; j < _dim; j++){
      numerator += ns[i](j, j-1);
      numerator += ns[i](j-1, j);
      denominator += rate_matrix(j, j) * fd[i](j);
    }
    rate_vector[i] = -numerator/denominator;
    rate_sum += rate_vector[i]*parameters.GetMixtureProbability(i);
  }
  
  for(int i = 0; i < _number_of_mixtures; i++){
    parameters.SetRateParameter(i, rate_vector[i]/rate_sum);
  }
  
  if(_model_id == 0){
    for(int j = 0; j < (_dim - 1)*2; j++){
      parameters.SetParameter(0, j, parameters.GetParameter(0, j)*rate_sum);
    }
  }else if(_model_id == 1){
    parameters.SetBeta(0, parameters.GetBeta(0)*rate_sum);
    parameters.SetAlpha(0, parameters.GetAlpha(0)*rate_sum);
  }else{
    parameters.SetBeta(0, parameters.GetBeta(0)*rate_sum);
    parameters.SetAlpha(0, parameters.GetAlpha(0)*rate_sum);
    parameters.SetGamma(0, parameters.GetGamma(0)*rate_sum);
  }
}

void Mirage::CheckInsideOutside(Node* current, int id){
  for(int i = 0; i < _number_of_samples; i++){
    double temp = 0.0;
    int inside_cash_id = CalcCashID(current->inside_cash_index,i);
    int outside_cash_id = CalcCashID(current->outside_cash_index,i);
    for(int j = 0; j < _dim; j++){
      temp += fmath::expd(current->outside_values[GetTripleArrayId(outside_cash_id,id,j)]+current->inside_values[GetTripleArrayId(inside_cash_id,id,j)] - _column_log_likelihood[i][id]);
    }
    cout << temp << endl;
    
  }
  
  if(current->left == NULL && current->right == NULL){
    return;
  }
  if(current->left != NULL){
    CheckInsideOutside(current->left, id);
  }
  if(current->right != NULL){
    CheckInsideOutside(current->right, id);
  }
}

bool Mirage::NewParameterEstimation(Node* current, MirageParameters& parameters, vector<VectorXd> &init, vector<VectorXd> &fd, vector<MatrixXd> &ns){   
  double sum_init = 0.0;
  for(int i = 0; i < _number_of_mixtures; i++){
    vector<double> sum_responsibility(_number_of_mixtures, 0.0);
    if(_mixture_method_id != 2){      
      for(int j = 0; j < _number_of_samples; j++){
	sum_responsibility[i] += _responsibility[j][i];
      }
      parameters.SetMixtureProbability(i, sum_responsibility[i]/_number_of_samples);
    }
    
    
    if(IsPatternMixture()){
      for(int j = 0; j < _dim; j++){
	parameters.SetInitProb(i,j, init[i](j)/sum_responsibility[i]);
      }
    }else{
      for(int j = 0; j < _dim; j++){
	sum_init += init[i](j);
      }
    }
  }
 

  if(!IsPatternMixture()){
    for(int j = 0; j < _dim; j++){
      double temp = 0.0;
      for(int i = 0; i < _number_of_mixtures; i++){
	temp += init[i](j);
      }
      parameters.SetInitProb(0,j, temp/sum_init);
    }
  }
  
  if(_model_id == 0 || _model_id == 1 || _model_id == 3){
    int outer_loop_size = IsPatternMixture() ? _number_of_mixtures : 1;
    int inner_loop_size = IsPatternMixture() ? 1 : _number_of_mixtures;

    for(int i = 0; i < outer_loop_size; i++){
      if(_mixture_method_id == 3 && parameters.GetMixtureProbability(i)==0.0){
	continue;
      }
      vector<int> value_size_vector = {4*(_dim-1), 4, 0, 6};
      int value_vector_size = value_size_vector[_model_id];
      
      vector<double> value_vector; value_vector.resize(value_vector_size, 0.0);

      for(int j = 0; j < inner_loop_size; j++){
	int mixture_index =  IsPatternMixture() ? i : j;
	double rate_factor = !IsPatternMixture() ? parameters.GetRateParameter(mixture_index) : 1.0;
	
	for(int k = 1; k < _dim; k++){
	  int bas = _model_id == 0 ? (k-1)*4 : 0;
	  value_vector[bas] += ns[mixture_index](k, k-1);
	  value_vector[bas+1] += rate_factor*fd[mixture_index](k);
	  if(!(_model_id == 3 && k == 1)){
	    value_vector[bas+2] += ns[mixture_index](k-1, k);
	    value_vector[bas+3] += rate_factor*fd[mixture_index](k-1);
	  }else{
	    value_vector[4] += ns[mixture_index](k-1, k);
	    value_vector[5] += rate_factor*fd[mixture_index](k-1);
	  }
	}

      }
      
      for(int j = 0; j <value_vector_size; j+=2){
	double value = value_vector[j]/value_vector[j+1];
	
	if(value < 0.0 || std::isnan(value)){return(false);}
      }
      
      if(_model_id == 0){
	for(int j = 0; j < _dim - 1; j++){
	  parameters.SetParameter(i, j, value_vector[j*4+2] / value_vector[j*4+3]);
	  parameters.SetParameter(i, _dim-1+j, value_vector[j*4] / value_vector[j*4+1]);
	}
      }else if(_model_id == 1){
	parameters.SetBeta(i, value_vector[0]/value_vector[1]);
	parameters.SetAlpha(i, value_vector[2]/value_vector[3]);
      }else{
	parameters.SetBeta(i, value_vector[0]/value_vector[1]);
	parameters.SetAlpha(i, value_vector[2]/value_vector[3]);
	parameters.SetGamma(i, value_vector[4]/value_vector[5]);
      }
    }
  }else{
    int outer_loop_size = IsPatternMixture() ? _number_of_mixtures : 1;
    int inner_loop_size = IsPatternMixture() ? 1 : _number_of_mixtures;

    for(int i = 0; i < outer_loop_size; i++){
      double beta_numerator = 0.0;
      double beta_denominator = 0.0;
      
       for(int j = 0; j < inner_loop_size; j++){
	 int mixture_index =  IsPatternMixture() ? i : j;
	 double rate_factor = !IsPatternMixture() ? parameters.GetRateParameter(mixture_index) : 1.0;
	 
	 
	 for(int k = 1; k < _dim; k++){
	   beta_numerator += ns[mixture_index](k, k-1);
	   beta_denominator += rate_factor*k*fd[mixture_index](k);
	 }	 
       }
       
       double beta = beta_numerator/beta_denominator;
       if(beta < 0.0 || std::isnan(beta)){return(false);}
       parameters.SetBeta(i, beta);
       GradientDescent(0, fd, ns, parameters); //alpha
       GradientDescent(1, fd, ns, parameters); //gamma
       
    }
  }

  if(!IsPatternMixture()){    
    CalcRateParameter(fd, ns, parameters);
  } 
    
  parameters.SetSubstitutionRateMatrix();
  return(true);
}

void Mirage::GradientDescent(int id, vector<VectorXd>& fd, vector<MatrixXd>& ns, MirageParameters& parameters){
  int outer_loop_size = IsPatternMixture() ? _number_of_mixtures : 1;
  int inner_loop_size = IsPatternMixture() ? 1 : _number_of_mixtures;
  
  for(int i = 0; i < outer_loop_size; i++){
    bool break_flag = false;
    double weight = parameters.GetInitGradWeight();
    while(true){
      double partial_par = 0.0;
      for(int j = 0; j < inner_loop_size; j++){
	int mixture_index =  IsPatternMixture() ? i : j;
	if(id==0){
	  partial_par += CalcPartialAlpha(mixture_index, fd[mixture_index], ns[mixture_index], parameters);
	
	}else{
	  partial_par += CalcPartialGamma(mixture_index, fd[mixture_index], ns[mixture_index], parameters);
	}
      }
      
      if(abs(partial_par) < parameters.GetPartialParThreshold()){
	break;
      }
      while(true){
	double new_par = id == 0 ? parameters.GetAlpha(i) : parameters.GetGamma(i);

	new_par += weight*partial_par;
	if(new_par > 0.0){
	  double new_q = 0.0; double old_q = 0.0;
	  
	  for(int j = 0; j < inner_loop_size; j++){
	    int mixture_index =  IsPatternMixture() ? i : j;
	    double rate_factor = !IsPatternMixture() ? parameters.GetRateParameter(mixture_index) : 1.0;
	    
	    if(id==0){
	      new_q += Qem(fd[mixture_index],ns[mixture_index],new_par,parameters.GetGamma(i),rate_factor);
	    }else{
	      new_q += Qem(fd[mixture_index],ns[mixture_index],parameters.GetAlpha(i),new_par,rate_factor);
	    }
	    old_q += Qem(fd[mixture_index],ns[mixture_index],parameters.GetAlpha(i),parameters.GetGamma(i),rate_factor);
	  }
	  if(new_q > old_q){
	    if(id == 0){
	      parameters.SetAlpha(i, new_par);
	    }else{
	      parameters.SetGamma(i, new_par);
	    }
	    break;
	  }else{
	    weight = weight * parameters.GetWeightDecayRate();
	  }
	}else{
	  weight = weight * parameters.GetWeightDecayRate();
	}
	if(weight < parameters.GetWeightThreshold()){
	  break_flag = true;
	  break;
	}
      }
      if(break_flag){break;}
    }
  }
}

double Mirage::Qem(VectorXd& fd, MatrixXd& ns, double a, double g, double rate_factor){
  double temp = 0.0;  
 
  for(int i = 0; i < _dim-1; i++){
    temp += -fd(i)*rate_factor*a;
    temp += ns(i, i+1)*log(rate_factor*a + i*rate_factor*g);
    if(i >= 1){
      temp += -i*fd(i)*rate_factor*g;
    }
  }
  return(temp);
}

double Mirage::CalcPartialAlpha(int mixture_id, VectorXd& fd, MatrixXd& ns, MirageParameters& parameters){
  double temp = 0.0;
  double rate_factor = !IsPatternMixture() ? parameters.GetRateParameter(mixture_id) : 1.0;
  if(!IsPatternMixture()){
    mixture_id = 0;
  }
  for(int i = 0; i < _dim-1; i++){
    temp += -rate_factor*fd(i);
    temp += ns(i, i+1)/(parameters.GetAlpha(mixture_id) + i * parameters.GetGamma(mixture_id));
  }
  return temp/_number_of_samples;
}

double Mirage::CalcPartialGamma(int mixture_id,VectorXd& fd, MatrixXd& ns, MirageParameters& parameters){
  double temp = 0.0;
  double rate_factor = !IsPatternMixture() ? parameters.GetRateParameter(mixture_id) : 1.0;
  if(!IsPatternMixture()){
    mixture_id = 0;
  }
  for(int i = 1; i < _dim-1; i++){
    temp += -i*rate_factor*fd(i);
    temp += ns(i, i+1)/(parameters.GetAlpha(mixture_id) + i * parameters.GetGamma(mixture_id));
  }
  return temp/_number_of_samples ;
}

void Mirage::SetOldParameter(Parameter& old_parameter, MirageParameters& m_parameter){
  int number_of_raw_matricies = IsPatternMixture() ? _number_of_mixtures : 1;

  if(_mixture_method_id == 1){
    for(int i = 0; i < _number_of_mixtures; i++){
      m_parameter.SetRateParameter(i, old_parameter.rate_parameter[i]);
    }
  }else if(_mixture_method_id == 2){
    m_parameter.SetGammaDistributionParameter(old_parameter.gamma_distribution_parameter);
  }
  
  for(int i = 0; i < number_of_raw_matricies; i++){
    if(_mixture_method_id != 2){
      m_parameter.SetMixtureProbability(i, old_parameter.mixture_probability[i]);
    }
    for(int j = 0; j < _dim; j++){
      m_parameter.SetInitProb(i, j, old_parameter.init_prob[i](j));    
    }
    if(_model_id == 1){
      m_parameter.SetAlpha(i, old_parameter.alpha[i]);
      m_parameter.SetBeta(i, old_parameter.beta[i]);
    }else if(_model_id == 2 || _model_id == 3){
      m_parameter.SetAlpha(i, old_parameter.alpha[i]);
      m_parameter.SetBeta(i, old_parameter.beta[i]);
      m_parameter.SetGamma(i, old_parameter.gamma[i]);
    }else{
      for(int j = 0; j < (_dim - 1)*2; j++){
	m_parameter.SetParameter(i, j, old_parameter.parameter[i][j]);
      }
    }
  }
}

void Mirage::SaveOldParameter(MirageParameters& m_parameter, Parameter& old_parameter){
  int number_of_raw_matricies = IsPatternMixture() ? _number_of_mixtures : 1;

  if(_mixture_method_id == 1){
    old_parameter.rate_parameter.resize(_number_of_mixtures,0.0);
    for(int i = 0; i < _number_of_mixtures; i++){
      old_parameter.rate_parameter[i] = m_parameter.GetRateParameter(i);
    }
  }else if(_mixture_method_id == 2){
    old_parameter.gamma_distribution_parameter = m_parameter.GetGammaDistributionParameter();
  }

  for(int i = 0; i < number_of_raw_matricies; i++){
    if(_mixture_method_id != 2){
      old_parameter.mixture_probability.resize(_number_of_mixtures,0.0);
      old_parameter.mixture_probability[i] = m_parameter.GetMixtureProbability(i);
    }
    
    for(int j = 0; j < _dim; j++){
      old_parameter.init_prob.resize(_number_of_mixtures, VectorXd::Zero(_dim));
      old_parameter.init_prob[i](j) = m_parameter.GetInitProb(i,j);    
    }
    if(_model_id == 1){
      old_parameter.alpha.resize(_number_of_mixtures,0.0);
      old_parameter.beta.resize(_number_of_mixtures,0.0);
      
      old_parameter.alpha[i] = m_parameter.GetAlpha(i);
      old_parameter.beta[i] = m_parameter.GetBeta(i);
    }else if(_model_id == 2 || _model_id == 3){
      old_parameter.alpha.resize(_number_of_mixtures,0.0);
      old_parameter.beta.resize(_number_of_mixtures,0.0);
      old_parameter.gamma.resize(_number_of_mixtures,0.0);
      
      old_parameter.alpha[i] = m_parameter.GetAlpha(i);
      old_parameter.beta[i] = m_parameter.GetBeta(i);
      old_parameter.gamma[i] = m_parameter.GetGamma(i);      
    }else{
      for(int j = 0; j < (_dim - 1)*2; j++){
	old_parameter.parameter.resize(_number_of_mixtures, vector<double>((_dim - 1)*2, 0.0));
	old_parameter.parameter[i][j] = m_parameter.GetParameter(i, j);
      }
    }
  }
}

void Mirage::CalcOutsideValues(Node* current, MirageParameters& parameters, int id){
  for(int i = 0; i < _number_of_mixtures; i++){
    int temp_i = IsPatternMixture() ? i : 0;
    if(current->parent == NULL){
      for(int j = 0; j < _number_of_samples; j++){
	int cash_id = CalcCashID(current->outside_cash_index,j);	  
	for(int k = 0; k < _dim; k++){
	  double temp_prob = parameters.GetInitProb(temp_i,k);
	  current->outside_values[GetTripleArrayId(cash_id,i,k)] = temp_prob != 0.0 ? log(temp_prob) : -DBL_MAX;
	}
      }
    }else{      
      vector<MatrixXd> substitution_rate_matrix = parameters.GetSubstitutionRateMatrix();
      MatrixXd log_parent_probability_matrix = (current->edge_length * substitution_rate_matrix[i]).exp();
      AddEpsilon(log_parent_probability_matrix);
      Node* sister = id == 0 ? current->parent->right : current->parent->left;
      MatrixXd log_sister_probability_matrix = (sister->edge_length * substitution_rate_matrix[i]).exp();
      AddEpsilon(log_sister_probability_matrix);
      for(int j = 0; j < _dim; j++){
	for(int k = 0; k < _dim; k++){
	  log_parent_probability_matrix(j, k) = log(log_parent_probability_matrix(j, k));
	  log_sister_probability_matrix(j, k) = log(log_sister_probability_matrix(j, k));
	}
      }
      for(int j = 0; j < _number_of_samples; j++){
	int sister_id = CalcCashID(sister->inside_cash_index,j);
	int parent_id = CalcCashID(current->parent->outside_cash_index,j);
	int cash_id = CalcCashID(current->outside_cash_index,j);
	
	vector<double> inner_temp_array(_dim,0.0);
	for(int k = 0; k < _dim; k++){
	  if(current->outside_cash_index[j] < 0){
	    current->outside_values[GetTripleArrayId(cash_id,i,k)] = -DBL_MAX;	  	      
	    for(int l = 0; l < _dim; l++){	 	  
	      double inner_temp = -DBL_MAX;
	      
	      if(k==0){
		for(int m = 0; m < _dim; m++){
		  if(sister->inside_values[GetTripleArrayId(sister_id,i,m)] != -DBL_MAX){
		    double value = log_sister_probability_matrix(l,m) + sister->inside_values[GetTripleArrayId(sister_id,i,m)];
		    inner_temp = inner_temp == -DBL_MAX ? value : logsumexp(inner_temp, value);
		  }		
		}
		inner_temp_array[l] = inner_temp;
	      }else{
		inner_temp = inner_temp_array[l];
	      }	      
	      
	      if(current->parent->outside_values[GetTripleArrayId(parent_id,i,l)] != -DBL_MAX){
		inner_temp += log_parent_probability_matrix(l,k) + current->parent->outside_values[GetTripleArrayId(parent_id,i,l)];
	      }else{
		inner_temp = 0;
	      }
	      
	      if(current->outside_values[GetTripleArrayId(cash_id,i,k)] == -DBL_MAX){
		current->outside_values[GetTripleArrayId(cash_id,i,k)] = inner_temp;
	      }else{
		current->outside_values[GetTripleArrayId(cash_id,i,k)] = logsumexp(current->outside_values[GetTripleArrayId(cash_id,i,k)], inner_temp);
	      }
	    }
	  }
	}
      }
    }
  }
  
  if(current->left == NULL && current->right == NULL){
    return;
  }
  if(current->left != NULL){
    CalcOutsideValues(current->left,parameters,0);
  }
  if(current->right != NULL){
    CalcOutsideValues(current->right,parameters,1);
  }
}

void Mirage::CalcOutsideValuesDPM(Node* current, MirageParameters& parameters, int id){
  
  if(current->parent == NULL){
    for(int j = 0; j < _number_of_samples; j++){
      int temp_i = _mixture_id[j];
      for(int k = 0; k < _dim; k++){
	double temp_prob = parameters.GetInitProb(temp_i,k);
	current->outside_values[j*_dim+k] = temp_prob != 0.0 ? log(temp_prob) : -DBL_MAX;
      }
    }
  }else{
    vector<MatrixXd> log_parent_probability_matrix_vector;
    vector<MatrixXd> log_sister_probability_matrix_vector;
    vector<MatrixXd> substitution_rate_matrix = parameters.GetSubstitutionRateMatrix();
    Node* sister = id == 0 ? current->parent->right : current->parent->left;
    
    for(int i = 0; i < _number_of_mixtures; i++){
      MatrixXd log_parent_probability_matrix = (current->edge_length * substitution_rate_matrix[i]).exp();      
      MatrixXd log_sister_probability_matrix = (sister->edge_length * substitution_rate_matrix[i]).exp();
      AddEpsilon(log_parent_probability_matrix);
      AddEpsilon(log_sister_probability_matrix);
      
      for(int j = 0; j < _dim; j++){
	for(int k = 0; k < _dim; k++){
	  log_parent_probability_matrix(j, k) = log(log_parent_probability_matrix(j, k));
	  log_sister_probability_matrix(j, k) = log(log_sister_probability_matrix(j, k));
	}
      }
      log_parent_probability_matrix_vector.push_back(log_parent_probability_matrix);
      log_sister_probability_matrix_vector.push_back(log_sister_probability_matrix);
    }
    
    for(int j = 0; j < _number_of_samples; j++){
      int temp_i = _mixture_id[j];
      int sister_id = CalcCashID(sister->inside_cash_index,j);
	
      vector<double> inner_temp_array(_dim,0.0);
      for(int k = 0; k < _dim; k++){
	current->outside_values[j*_dim+k] = -DBL_MAX;	  	      
	for(int l = 0; l < _dim; l++){	 	  
	  double inner_temp = -DBL_MAX;
	  
	  if(k==0){
	    for(int m = 0; m < _dim; m++){
	      if(sister->inside_values[GetTripleArrayId(sister_id,temp_i,m)] != -DBL_MAX){
		double value = log_sister_probability_matrix_vector[temp_i](l,m) + sister->inside_values[GetTripleArrayId(sister_id,temp_i,m)];
		inner_temp = inner_temp == -DBL_MAX ? value : logsumexp(inner_temp, value);
	      }		
	    }
	    inner_temp_array[l] = inner_temp;
	  }else{
	    inner_temp = inner_temp_array[l];
	  }	      
	  
	  if(current->parent->outside_values[j*_dim+l] != -DBL_MAX){
	    inner_temp += log_parent_probability_matrix_vector[temp_i](l,k) + current->parent->outside_values[j*_dim+l];
	  }else{
	    inner_temp = 0;
	  }
	  
	  if(current->outside_values[j*_dim+k] == -DBL_MAX){
	    current->outside_values[j*_dim+k] = inner_temp;
	  }else{
	    current->outside_values[j*_dim+k] = logsumexp(current->outside_values[j*_dim+k], inner_temp);
	  }
	}
      }
    }
  }
  
  if(current->left == NULL && current->right == NULL){
    return;
  }
  if(current->left != NULL){
    CalcOutsideValuesDPM(current->left,parameters,0);
  }
  if(current->right != NULL){
    CalcOutsideValuesDPM(current->right,parameters,1);
  }
}

void Mirage::CalcInsideValues(Node* current, MirageParameters& parameters){
  if(current->left == NULL && current->right == NULL){
    return;
  }
  if(current->left != NULL){
    CalcInsideValues(current->left, parameters);
  }
  if(current->right != NULL){
    CalcInsideValues(current->right, parameters);
  }
  
  vector<MatrixXd> substitution_rate_matrix = parameters.GetSubstitutionRateMatrix();
  
  for(int i = 0; i < _number_of_mixtures; i++){
    MatrixXd left_probability_matrix = (current->left->edge_length * substitution_rate_matrix[i]).exp();
    MatrixXd right_probability_matrix = (current->right->edge_length * substitution_rate_matrix[i]).exp();
    AddEpsilon(left_probability_matrix);
    AddEpsilon(right_probability_matrix);
    
    for(int j = 0; j < _number_of_samples; j++){
      for(int k = 0; k < _dim; k++){
	if(current->inside_cash_index[j] < 0){
	  int inside_id = -current->inside_cash_index[j]-1;
	  double l_temp = -DBL_MAX;
	  double r_temp = -DBL_MAX;
	  
	  for(int l = 0; l < _dim; l++){
	    double cur_l, cur_r;
	    int left_id = CalcCashID(current->left->inside_cash_index, j);
	    int right_id = CalcCashID(current->right->inside_cash_index, j);
	    cur_l = current->left->inside_values[GetTripleArrayId(left_id,i,l)] == -DBL_MAX ? -DBL_MAX :
	      current->left->inside_values[GetTripleArrayId(left_id,i,l)] + log(left_probability_matrix(k,l));
	    cur_r = current->right->inside_values[GetTripleArrayId(right_id,i,l)] == -DBL_MAX ? -DBL_MAX :
	      current->right->inside_values[GetTripleArrayId(right_id,i,l)] + log(right_probability_matrix(k,l));
	    
	    if(l_temp == -DBL_MAX){
	      l_temp = cur_l;
	    }else{
	      l_temp = cur_l == -DBL_MAX ? l_temp : logsumexp(l_temp, cur_l);
	    }
	    if(r_temp == -DBL_MAX){
	      r_temp = cur_r;
	    }else{
	      r_temp = cur_r == -DBL_MAX ? r_temp : logsumexp(r_temp, cur_r);
	    }	    
	  }	
	  current->inside_values[GetTripleArrayId(inside_id,i,k)] = l_temp + r_temp;
	}
      }
    }
  }
}

int Mirage::CalcCashID(int* cash_index, int id){
  int cash_id = cash_index[id] < 0 ? -cash_index[id] - 1 : -cash_index[cash_index[id]] - 1;
  return(cash_id);
}

void Mirage::CalcColumnLogLikelihood(Node* root, MirageParameters& parameters){
  for(int i = 0; i < _number_of_mixtures; i++){
    int temp_i = IsPatternMixture() ? i : 0;
    
    for(int j = 0; j < _number_of_samples; j++){
      double temp = -DBL_MAX;
      int cash_id = CalcCashID(root->inside_cash_index,j);
      for(int k = 0; k < _dim ; k++){
	if(parameters.GetInitProb(temp_i,k) != 0.0){	 
	  double value = root->inside_values[GetTripleArrayId(cash_id,i,k)] + log(parameters.GetInitProb(temp_i,k));
	  temp = temp == -DBL_MAX ? value : logsumexp(temp, value);
	}
      }
      
      _column_log_likelihood[j][i] = temp;
      for(int k = 0; k < _dim ; k++){
	_init_prob_sufficient_statistics[j][i][k] =
	  parameters.GetInitProb(temp_i,k) == 0.0 ? 0.0 : fmath::expd(root->inside_values[GetTripleArrayId(cash_id,i,k)] + log(parameters.GetInitProb(temp_i,k)) - temp);

      } 
    }
  }
}

double Mirage::logsumexp(double x, double y){
  double temp = x > y ? x + log1p(fmath::expd(y-x)) : y + log1p(fmath::expd(x-y)) ;
  return(temp);
}

int Mirage::GetTripleArrayId(int sample_id, int mixture_id, int element_id){
  return(sample_id*_number_of_mixtures*_dim + mixture_id*_dim + element_id);
}

double Mirage::CalcDataLikelihood(MirageParameters& parameters){
  double likelihood = 0.0;
  for(int i = 0; i < _number_of_samples; i++){
    double temp = -DBL_MAX;
    for(int j = 0; j < _number_of_mixtures; j++){
      if(parameters.GetMixtureProbability(j)  != 0){
	double value =  log(parameters.GetMixtureProbability(j)) + _column_log_likelihood[i][j];
	temp = temp == -DBL_MAX ? value : logsumexp(temp, value);
      }
    }
   
    likelihood += temp;
  }
  return(likelihood);
}

void Mirage::CalcResponsibility(MirageParameters& parameters){
  for(int i = 0; i < _number_of_samples; i++){
    double sum = -DBL_MAX;
    for(int j = 0; j < _number_of_mixtures; j++){
      if(parameters.GetMixtureProbability(j) != 0){
	double value = _column_log_likelihood[i][j]+log(parameters.GetMixtureProbability(j));
	sum = sum == -DBL_MAX ? value : logsumexp(sum,value);
      }
    }

    int max_j = -1;
    double max_responsibility = 0.0;
    for(int j = 0; j < _number_of_mixtures; j++){
      if(parameters.GetMixtureProbability(j) == 0){	
	_responsibility[i][j] = 0.0;
      }else{
	double value = _column_log_likelihood[i][j]+log(parameters.GetMixtureProbability(j));
	_responsibility[i][j] = fmath::expd(value-sum);
	if(_responsibility[i][j] > max_responsibility){
	  max_responsibility = _responsibility[i][j];
	  max_j = j;
	}
      }
    }
    
    if(_mixture_method_id == 3){
      _mixture_id[i] = max_j;
      for(int j = 0; j < _number_of_mixtures; j++){
	_responsibility[i][j] = max_j == j ? 1.0 : 0.0;	
      }
    }
  }
}

int Mirage::NsId(int j, int k){
  if(j < k){
    return j*(_dim-1)+k-1;
  }else{
    return j*(_dim-1)+k;
  }
}

void Mirage::CalcTreeModelSufficientStatistics(Node* current, MirageParameters& parameters, int id, vector<VectorXd> &init, vector<VectorXd> &fd, vector<MatrixXd> &ns){
  vector<MatrixXd> substitution_rate_matrix = parameters.GetSubstitutionRateMatrix();
  
  if(current->parent != NULL){
    for(int i = 0; i < _number_of_mixtures; i++){      
      vector<double> fd_values(_dim*_number_of_samples, 0.0);
      vector<double> ns_values(_dim*(_dim-1)*_number_of_samples, 0.0);
      MatrixXd current_substitution_rate_matrix = current->edge_length * substitution_rate_matrix[i];
      MatrixXd substituiton_probability_matrix = current_substitution_rate_matrix.exp();
      vector<vector<vector<vector<double> > > > deriv_matrix_exp(_dim, vector<vector<vector<double> > >(_dim, vector<vector<double> >(_dim, vector<double>(_dim, 0.0))));

      MatrixXd auxiliary_matrix = MatrixXd::Zero(_dim*2,_dim*2);
      for(int a = 0; a < _dim; a++){
	for(int b = 0; b < _dim; b++){
	  auxiliary_matrix(a, b) = current_substitution_rate_matrix(a,b);
	  auxiliary_matrix(_dim+a, _dim+b) = current_substitution_rate_matrix(a,b);
	}
      }
      for(int c = 0; c < _dim; c++){
	for(int d = 0; d < _dim; d++){
	  if(abs(c-d) <= 1){
	    auxiliary_matrix(c, _dim+d) = 1;
	    MatrixXd exp_auxiliary_matrix = auxiliary_matrix.exp();
	    for(int a = 0; a < _dim; a++){
	      for(int b = 0; b < _dim; b++){
		deriv_matrix_exp[a][b][c][d] = exp_auxiliary_matrix(a, _dim+b);	      
		if(c != d){
		  deriv_matrix_exp[a][b][c][d] *=  current_substitution_rate_matrix(c,d);
		}	
	      }
	    }
	    auxiliary_matrix(c, _dim+d) = 0;
	  }
	}
      }

      
      Node* sister = id == 0 ? current->parent->right : current->parent->left;
      MatrixXd log_sister_probability_matrix = (sister->edge_length * substitution_rate_matrix[i]).exp();
      AddEpsilon(log_sister_probability_matrix);	
      for(int j = 0 ; j < _dim; j++){
	for(int k = 0; k < _dim; k++){
	  log_sister_probability_matrix(j,k) = log(log_sister_probability_matrix(j,k));
	}
      }

      for(int j = 0; j < _number_of_samples; j++){
	if(_mixture_method_id != 3 || _mixture_id[j] == i){
	  
	  int current_cash_id = CalcCashID(current->inside_cash_index,j);
	  int sister_cash_id = CalcCashID(sister->inside_cash_index,j);
	  int outside_cash_id = _mixture_method_id == 3 ? 0 : CalcCashID(current->parent->outside_cash_index,j);;
	  int ns_bas = j*(_dim-1)*_dim;
	  
	  for(int b = 0; b < _dim; b++){
	    double outside_plus = -DBL_MAX;
	    for(int c = 0; c < _dim; c++){
	      if(sister->inside_values[GetTripleArrayId(sister_cash_id,i,c)] != -DBL_MAX){
		double value = log_sister_probability_matrix(b,c) + sister->inside_values[GetTripleArrayId(sister_cash_id,i,c)];
		outside_plus = outside_plus == -DBL_MAX ? value : logsumexp(outside_plus, value);
	      }
	    }
	    if(_mixture_method_id != 3){
	      outside_plus += current->parent->outside_values[GetTripleArrayId(outside_cash_id,i,b)];
	    }else{
	      outside_plus += current->parent->outside_values[j*_dim+b];
	     
	    }
	    outside_plus -= _column_log_likelihood[j][i];
	      
	    for(int a = 0; a < _dim; a++){
	      double edge_prob = fmath::expd(current->inside_values[GetTripleArrayId(current_cash_id,i,a)] + outside_plus);
	      for(int k = 0; k < _dim; k++){
		fd_values[j*_dim+k] += deriv_matrix_exp[b][a][k][k] * edge_prob;      
		if(k != 0){
		  ns_values[ns_bas+ NsId(k,k-1)] +=  deriv_matrix_exp[b][a][k][k-1] * edge_prob;		
		}
		if(k != _dim-1){
		  ns_values[ns_bas+ NsId(k,k+1)] +=  deriv_matrix_exp[b][a][k][k+1] * edge_prob;		
		}
	      }
	    }
	  }

	  double temp_factor = current->edge_length *_responsibility[j][i];
	  for(int k = 0; k < _dim; k++){
	    fd[i](k) +=  fd_values[j*_dim+k]*temp_factor;
	    if(k != 0){
	      ns[i](k, k-1) += ns_values[ns_bas+ NsId(k,k-1)]*_responsibility[j][i];
	    }
	    if(k != _dim-1){
	      ns[i](k, k+1) += ns_values[ns_bas+ NsId(k,k+1)]*_responsibility[j][i];
	    }
	    
	  }
	  
	  
	}
      }
    }
  }else{
    for(int i = 0; i < _number_of_mixtures; i++){
      for(int j = 0; j < _dim; j++){
	for(int k = 0; k < _number_of_samples; k++){
	  init[i](j) += _responsibility[k][i]*_init_prob_sufficient_statistics[k][i][j];
	}
      }
    }
  }
  
  if(current->left == NULL && current->right == NULL){
    return;
  }
  if(current->left != NULL){
    CalcTreeModelSufficientStatistics(current->left, parameters, 0, init, fd, ns);
  }
  if(current->right != NULL){
    CalcTreeModelSufficientStatistics(current->right, parameters, 1, init, fd, ns);
  }
}

void Mirage::AddEpsilon(MatrixXd& matrix){
  for(int i = 0; i < _dim; i++){
    for(int j = 0; j < _dim; j++){
      if(matrix(i,j) < DBL_EPSILON){
	matrix(i,j) = DBL_EPSILON;
      }
    }    
  }
}
