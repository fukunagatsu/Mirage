#include "mirage.h"
#include "fmath.hpp"

void Mirage::Train(MirageParameters& parameters){
  Initiallize(parameters);
  
  double old_log_likelihood = 0.0;
  double new_log_likelihood = 0.0;
  int count = 0;
  Parameter old_parameter;
  while(true){
    cout << "loop:" << count << endl;      
    count++;
    CalcInsideValues(parameters.GetRoot(), parameters);
    CalcColumnLogLikelihood(parameters.GetRoot(), parameters);
    CalcOutsideValues(parameters.GetRoot(), parameters, -1);
    CalcTreeModelSufficientStatistics(parameters.GetRoot(), parameters, -1);
    CalcResponsibility(parameters);
    SaveOldParameter(parameters,old_parameter);
    
    bool flag = NewParameterEstimation(parameters.GetRoot(),parameters);
    if(!flag){break;}
    
    new_log_likelihood = CalcDataLikelihood(parameters);
    if(old_log_likelihood != 0.0){
      double value = new_log_likelihood - old_log_likelihood;
      if(count > parameters.GetLoopMax()  || new_log_likelihood - old_log_likelihood < parameters.GetLoopThreshold() ){	
	SetOldParameter(old_parameter,parameters);
	break;
      }
    }

    old_log_likelihood = new_log_likelihood;
  }
  HistoryReconstrucion(parameters.GetRoot(), parameters);
  Output(parameters,0);
}

void Mirage::Estimate(MirageParameters& parameters){
  Initiallize(parameters);  
  CalcInsideValues(parameters.GetRoot(), parameters);
  CalcColumnLogLikelihood(parameters.GetRoot(), parameters);
  CalcResponsibility(parameters);
  HistoryReconstrucion(parameters.GetRoot(), parameters);  
  Output(parameters,1);  
}

void Mirage::OutputReconstruction(ofstream& ofs, Node* current, int& count){
  ofs << "Node " << count << " : ";
  count++;
  for(int i = 0; i < _number_of_samples; i++){
    ofs << current->reconstruction[i] << " ";
  }
  ofs << endl;
  return;
}

void Mirage::HistoryTraceBack(ofstream& ofs, Node* current, int& count){
  if(current->left == NULL && current->right == NULL){   
    OutputReconstruction(ofs,current,count);    
    return;
  }
  if(current->parent != NULL){    
    for(int i = 0; i < _number_of_samples; i++){      
      current->reconstruction[i] = current->c[GetTripleArrayId(i,_mixture_id[i],current->parent->reconstruction[i])];
    }     
  }
  OutputReconstruction(ofs,current,count);
  
  if(current->left != NULL){
    HistoryTraceBack(ofs,current->left,count);
  }
  if(current->right != NULL){
    HistoryTraceBack(ofs,current->right,count);
  }
  return;  
}

void Mirage::HistoryReconstrucion(Node* current, MirageParameters& parameters){
  vector<MatrixXd> substitution_rate_matrix = parameters.GetSubstitutionRateMatrix();
  
  if(current->left == NULL && current->right == NULL){
    for(int i = 0; i < _number_of_mixtures; i++){
      MatrixXd substitution_probability_matrix = (current->edge_length * substitution_rate_matrix[i]).exp();
      for(int j = 0; j < _dim; j++){
	for(int k = 0; k < _dim; k++){
	  substitution_probability_matrix(j, k) = log(substitution_probability_matrix(j, k));
	}
      }
  
      for(int j = 0; j < _number_of_samples; j++){
	int state = current->reconstruction[j];
	for(int k = 0; k < _dim; k++){
	  current->c[GetTripleArrayId(j,i,k)] = state;
	  current->logL[GetTripleArrayId(j,i,k)] = substitution_probability_matrix(k,state);
	}
      }
    }
    return;
  }
  
  if(current->left != NULL){
    HistoryReconstrucion(current->left, parameters);
  }
  if(current->right != NULL){
    HistoryReconstrucion(current->right, parameters);
  }

  if(current->parent != NULL){
    for(int i = 0; i < _number_of_mixtures; i++){
      MatrixXd substitution_probability_matrix = (current->edge_length * substitution_rate_matrix[i]).exp();     
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
  }else{
    for(int j = 0; j < _number_of_samples; j++){
      double max_logL = -DBL_MAX;
	
      for(int i = 0; i < _number_of_mixtures; i++){
	for(int k = 0; k < _dim; k++){
	  double temp = parameters.GetInitProb(i,k) +
	    current->left->logL[GetTripleArrayId(j,i,k)] + current->right->logL[GetTripleArrayId(j,i,k)];
	  if(temp > max_logL){
	    max_logL = temp;
	    current->reconstruction[j] = k;
	    _mixture_id[j] = i;
	  }
	}
      }
    }
  }
  return;  
}

void Mirage::Output(MirageParameters& parameters, int id){
  string file_name = parameters.GetOutputFileName();
  if(id == 0){
    ofstream ofs_par((file_name+".par").c_str());
    ofs_par << _dim-1 << " " << parameters.GetModelID() << " " << _number_of_mixtures << endl;
    for(int i = 0; i < _number_of_mixtures; i++){
      ofs_par << parameters.GetMixtureProbability(i) << " ";
    }
    ofs_par << endl;
    
    for(int i = 0; i < _number_of_mixtures; i++){
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
  ofs_bas << "Number_of_Mixtures: " << _number_of_mixtures << endl;
  ofs_bas << "Log_Likelihood: " << CalcDataLikelihood(parameters) << endl;
  ofs_bas.close();

  ofstream ofs_res((file_name+".res").c_str());
  
  for(int i = 0; i < _number_of_samples; i++){
    for(int j = 0; j < _number_of_mixtures; j++){
      ofs_res<< _responsibility[i][j] << " ";
    }
    ofs_res << endl;
  }
  ofs_res.close();

  ofstream ofs_hist((file_name+".ahr").c_str());
  int count = 0;
  HistoryTraceBack(ofs_hist,parameters.GetRoot(), count);
  ofs_hist.close();
}

void Mirage::Initiallize(MirageParameters& parameters){
  _number_of_mixtures = parameters.GetNumberOfMixtures();
  _dim = parameters.GetDim();
  _number_of_samples = parameters.GetNumberOfSamples();
  _model_id = parameters.GetModelID();
  _column_log_likelihood.resize(_number_of_samples, vector<double>(_number_of_mixtures, 0.0));
  _init_prob_sufficient_statistics.resize(_number_of_samples, vector<vector<double> >(_number_of_mixtures, vector<double>(_dim, 0.0)));
  _responsibility.resize(_number_of_samples, vector<double>(_number_of_mixtures, 0.0));
  _mixture_id.resize(_number_of_samples, 0.0);
}

bool Mirage::NewParameterEstimation(Node* current, MirageParameters& parameters){
  vector<double> sum_responsibility(_number_of_mixtures, 0.0);
  vector<VectorXd> init(_number_of_mixtures, VectorXd::Zero(_dim));
  vector<VectorXd> fd(_number_of_mixtures, VectorXd::Zero(_dim));
  vector<MatrixXd> ns(_number_of_mixtures, MatrixXd::Zero(_dim, _dim));
  CalcTotalSS(current, init,fd,ns);
  
  for(int i = 0; i < _number_of_mixtures; i++){
    if(_model_id == 1){
      double beta_numerator = 0.0;
      double beta_denominator = 0.0;
      for(int j = 1; j < _dim; j++){
	beta_numerator += ns[i](j, j-1);
	beta_denominator += fd[i](j);
      }
      double alpha_numerator = 0.0;
      double alpha_denominator = 0.0;
      for(int j = 0; j < _dim-1; j++){
	alpha_numerator += ns[i](j, j+1);
	alpha_denominator += fd[i](j);
      }
      double beta = beta_numerator/beta_denominator;
      double alpha = alpha_numerator/alpha_denominator;
      if(beta < 0.0 || alpha < 0.0
	 || std::isnan(beta) || std::isnan(alpha)){
	return(false);
      }
      
      parameters.SetBeta(i, beta);
      parameters.SetAlpha(i, alpha);
    }else if(_model_id == 2){
      double beta_numerator = 0.0;
      double beta_denominator = 0.0;
      for(int j = 1; j < _dim; j++){
	beta_numerator += ns[i](j, j-1);
	beta_denominator += j*fd[i](j);
      }
      double beta = beta_numerator/beta_denominator;
      if(beta < 0.0 || std::isnan(beta)){return(false);}
      parameters.SetBeta(i, beta);
      
      GradientDescent(0, fd, ns, parameters); //alpha
      GradientDescent(1, fd, ns, parameters); //gamma
    }else if(_model_id == 0){
      for(int j = 0; j < _dim - 1; j++){
	double value = ns[i](j, j+1)/fd[i](j);
	if(value < 0.0 || std::isnan(value)){return(false);}
      }
      for(int j = 1; j < _dim; j++){
	double value = ns[i](j, j-1)/fd[i](j);
	if(value < 0.0 || std::isnan(value)){return(false);}
      }
      
      for(int j = 0; j < _dim - 1; j++){
	parameters.SetParameter(i, j, ns[i](j, j+1)/fd[i](j));
      }
      for(int j = 1; j < _dim; j++){
	parameters.SetParameter(i, _dim-1+j-1, ns[i](j, j-1)/fd[i](j));
      }
    }else{
      double beta_numerator = 0.0;
      double beta_denominator = 0.0;
      for(int j = 1; j < _dim; j++){
	beta_numerator += ns[i](j, j-1);
	beta_denominator += fd[i](j);
      }
      double alpha_numerator = 0.0;
      double alpha_denominator = 0.0;
      for(int j = 1; j < _dim-1; j++){
	alpha_numerator += ns[i](j, j+1);
	alpha_denominator += fd[i](j);
      }
      double beta = beta_numerator/beta_denominator;
      double alpha = alpha_numerator/alpha_denominator;
      double gamma = ns[i](0, 1)/fd[i](0);
      if(beta < 0.0 || alpha < 0.0 || gamma < 0.0
	 || std::isnan(beta) || std::isnan(alpha) || std::isnan(gamma)){
	return(false);
      }
      
      parameters.SetBeta(i, beta);
      parameters.SetAlpha(i, alpha);
      parameters.SetGamma(i, gamma);
    }
    
    for(int j = 0; j < _number_of_samples; j++){
      sum_responsibility[i] += _responsibility[j][i];
    }
    parameters.SetMixtureProbability(i, sum_responsibility[i]/_number_of_samples);

    for(int j = 0; j < _dim; j++){
      parameters.SetInitProb(i,j, init[i](j)/sum_responsibility[i]);
    }
  }
  parameters.SetSubstitutionRateMatrix();
  return(true);
}

void Mirage::GradientDescent(int id, vector<VectorXd>& fd, vector<MatrixXd>& ns, MirageParameters& parameters){  
  for(int i = 0; i < _number_of_mixtures; i++){
    bool break_flag = false;
    double weight = parameters.GetInitGradWeight();
    while(true){
      double partial_par = id == 0 ? CalcPartialAlpha(i, fd[i], ns[i], parameters) : CalcPartialGamma(i, fd[i], ns[i], parameters);
      
      if(abs(partial_par) < parameters.GetPartialParThreshold()){
	break;
      }
      while(true){
	double new_par = id == 0 ? parameters.GetAlpha(i) : parameters.GetGamma(i);
	new_par += weight*partial_par;	
	if(new_par > 0.0){
	  double new_q = id == 0 ? Qem(fd[i],ns[i],new_par,parameters.GetGamma(i)) : Qem(fd[i],ns[i],parameters.GetAlpha(i),new_par);
	  double old_q =  Qem(fd[i],ns[i],parameters.GetAlpha(i),parameters.GetGamma(i));	 
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

double Mirage::Qem(VectorXd& fd, MatrixXd& ns, double a, double g){
  double temp = 0.0;
  for(int i = 0; i < _dim-1; i++){
    temp += -fd(i)*a;
    temp += ns(i, i+1)*log(a + i *g);
    if(i >= 1){
      temp += -i*fd(i)*g;
    }
  }
  return(temp);
}

double Mirage::CalcPartialAlpha(int mixture_id, VectorXd& fd, MatrixXd& ns, MirageParameters& parameters){
  double temp = 0.0;
  for(int i = 0; i < _dim-1; i++){
    temp += -fd(i);
    temp += ns(i, i+1)/(parameters.GetAlpha(mixture_id) + i * parameters.GetGamma(mixture_id));
  }
  return temp/_number_of_samples;
}

double Mirage::CalcPartialGamma(int mixture_id,VectorXd& fd, MatrixXd& ns, MirageParameters& parameters){
  double temp = 0.0;
  for(int i = 1; i < _dim-1; i++){
    temp += -i*fd(i);
    temp += ns(i, i+1)/(parameters.GetAlpha(mixture_id) + i * parameters.GetGamma(mixture_id));
  }
  return temp/_number_of_samples ;
}

void Mirage::CalcTotalSS(Node* current, vector<VectorXd>& init, vector<VectorXd>& fd,  vector<MatrixXd>& ns){  
  if(current->parent != NULL){
    for(int i = 0; i < _number_of_mixtures; i++){
      for(int j = 0; j < _dim; j++){
	for(int k = 0; k < _number_of_samples; k++){
	  fd[i](j) += current->edge_length * current->fd_values[GetTripleArrayId(k,i,j)]*_responsibility[k][i];
	  
	  for(int l = 0; l < _dim; l++){
	    if(j != l){
	      ns[i](j, l) += current->ns_values[k*_number_of_mixtures*(_dim-1)*_dim + i*(_dim-1)*_dim+ NsId(j, l)]*_responsibility[k][i];
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
    CalcTotalSS(current->left, init, fd, ns);
  }
  if(current->right != NULL){
    CalcTotalSS(current->right, init, fd, ns);
  }
}

void Mirage::SetOldParameter(Parameter& old_parameter, MirageParameters& m_parameter){
  for(int i = 0; i < _number_of_mixtures; i++){
    m_parameter.SetMixtureProbability(i, old_parameter.mixture_probability[i]);
    
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
  for(int i = 0; i < _number_of_mixtures; i++){
    old_parameter.mixture_probability.resize(_number_of_mixtures,0.0);
    old_parameter.mixture_probability[i] = m_parameter.GetMixtureProbability(i);
    
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
    if(current->parent == NULL){
      for(int j = 0; j < _number_of_samples; j++){
	for(int k = 0; k < _dim; k++){
	  double temp_prob = parameters.GetInitProb(i,k);
	  current->outside_values[GetTripleArrayId(j,i,k)] = temp_prob != 0.0 ? log(temp_prob) : -DBL_MAX;
	}
      }
    }else{
      vector<MatrixXd> substitution_rate_matrix = parameters.GetSubstitutionRateMatrix();
      MatrixXd log_parent_probability_matrix = (current->edge_length * substitution_rate_matrix[i]).exp();
      Node* sister = id == 0 ? current->parent->right : current->parent->left;
      MatrixXd log_sister_probability_matrix = (sister->edge_length * substitution_rate_matrix[i]).exp();
      
      for(int j = 0; j < _dim; j++){
	for(int k = 0; k < _dim; k++){
	  log_parent_probability_matrix(j, k) = log(log_parent_probability_matrix(j, k));
	  log_sister_probability_matrix(j, k) = log(log_sister_probability_matrix(j, k));
	}
      }
      
      for(int j = 0; j < _number_of_samples; j++){
	vector<double> inner_temp_array(_dim,0.0);
	for(int k = 0; k < _dim; k++){
	  current->outside_values[GetTripleArrayId(j,i,k)] = -DBL_MAX;
	  
	  for(int l = 0; l < _dim; l++){	 	  
	    double inner_temp = -DBL_MAX;
	    
	    if(k==0){
	      for(int m = 0; m < _dim; m++){
		if(sister->inside_values[GetTripleArrayId(j,i,m)] != -DBL_MAX){
		  double value = log_sister_probability_matrix(l,m) + sister->inside_values[GetTripleArrayId(j,i,m)];
		  inner_temp = inner_temp == -DBL_MAX ? value : logsumexp(inner_temp, value);
		}		
	      }
	      inner_temp_array[l] = inner_temp;
	    }else{
	      inner_temp = inner_temp_array[l];
	    }	      
	    
	    if(current->parent->outside_values[GetTripleArrayId(j,i,l)] != -DBL_MAX){
	      inner_temp += log_parent_probability_matrix(l,k) + current->parent->outside_values[GetTripleArrayId(j,i,l)];
	    }else{
	      inner_temp = 0;
	    }
	    
	    if(current->outside_values[GetTripleArrayId(j,i,k)] == -DBL_MAX){
	      current->outside_values[GetTripleArrayId(j,i,k)] = inner_temp;
	    }else{
	      current->outside_values[GetTripleArrayId(j,i,k)] = logsumexp(current->outside_values[GetTripleArrayId(j,i,k)], inner_temp);
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
  
    for(int j = 0; j < _number_of_samples; j++){
      for(int k = 0; k < _dim; k++){
	double l_temp = -DBL_MAX;
	double r_temp = -DBL_MAX;
      
	for(int l = 0; l < _dim; l++){
	  double cur_l, cur_r;
	  cur_l = current->left->inside_values[GetTripleArrayId(j,i,l)] == -DBL_MAX ? -DBL_MAX : current->left->inside_values[GetTripleArrayId(j,i,l)] + log(left_probability_matrix(k,l));
	  cur_r = current->right->inside_values[GetTripleArrayId(j,i,l)] == -DBL_MAX ? -DBL_MAX : current->right->inside_values[GetTripleArrayId(j,i,l)] + log(right_probability_matrix(k,l));

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
	
	current->inside_values[GetTripleArrayId(j,i,k)] = l_temp + r_temp;
      }
    }
  }
}

void Mirage::CalcColumnLogLikelihood(Node* root, MirageParameters& parameters){
  for(int i = 0; i < _number_of_mixtures; i++){
    for(int j = 0; j < _number_of_samples; j++){
      double temp = -DBL_MAX;
      for(int k = 0; k < _dim ; k++){
	if(parameters.GetInitProb(i,k) != 0.0){	 
	  double value = root->inside_values[GetTripleArrayId(j,i,k)] + log(parameters.GetInitProb(i,k));
	  temp = temp == -DBL_MAX ? value : logsumexp(temp, value);
	}
      }
      _column_log_likelihood[j][i] = temp;
      for(int k = 0; k < _dim ; k++){
	_init_prob_sufficient_statistics[j][i][k] =
	  parameters.GetInitProb(i,k) == 0.0 ? 0.0 : fmath::expd(root->inside_values[GetTripleArrayId(j,i,k)] + log(parameters.GetInitProb(i,k)) - temp);
      }
    }
  }
}

double Mirage::logsumexp(double x, double y){
  double temp = x > y ? x + log(fmath::expd(y-x) + 1.0) : y + log(fmath::expd(x-y) + 1.0) ;
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

    for(int j = 0; j < _number_of_mixtures; j++){
      if(parameters.GetMixtureProbability(j) == 0){
	_responsibility[i][j] = 0.0;
      }else{
	double value = _column_log_likelihood[i][j]+log(parameters.GetMixtureProbability(j));
	_responsibility[i][j] = fmath::expd(value-sum);
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

complex<double> Mirage::Kappa(complex<double> a, complex<double> b){
  if(a==b){
    return(fmath::expd(a.real()));
  }else{
    return((fmath::expd(a.real()) - fmath::expd(b.real()))/(a - b));
  }
}

double Mirage::DerivMatrixExp(VectorXcd& eigen_values, MatrixXcd& eigen_vectors, MatrixXcd& inversed_eigen_vectors, int a, int b, int k, int l){
  complex<double> temp = 0.0;
  
  for(int u = 0; u < _dim; u++){
    for(int v = 0; v < _dim; v++){
      complex<double> x = eigen_vectors(a, u) * inversed_eigen_vectors(u, k) * eigen_vectors(l, v) * inversed_eigen_vectors(v, b)* Kappa(eigen_values(u),eigen_values(v));
      temp += x;
    }
  }
  return(temp.real());
}

void Mirage::CalcTreeModelSufficientStatistics(Node* current, MirageParameters& parameters, int id){
  vector<MatrixXd> substitution_rate_matrix = parameters.GetSubstitutionRateMatrix();
  for(int i = 0; i < _number_of_mixtures; i++){
    if(current->parent != NULL){      
      MatrixXd substituiton_probability_matrix = (current->edge_length * substitution_rate_matrix[i]).exp();
      EigenSolver<MatrixXd> es(current->edge_length * substitution_rate_matrix[i]);
      
      VectorXcd eigen_values = es.eigenvalues();
      MatrixXcd eigen_vectors = es.eigenvectors();
      MatrixXcd inversed_eigen_vectors = eigen_vectors.inverse();
      
      Node* sister = id == 0 ? current->parent->right : current->parent->left;
      MatrixXd log_sister_probability_matrix = (sister->edge_length * substitution_rate_matrix[i]).exp();
      for(int j = 0 ; j < _dim; j++){
	for(int k = 0; k < _dim; k++){
	  log_sister_probability_matrix(j,k) = log(log_sister_probability_matrix(j,k));
	}
      }
      vector<vector<vector<vector<double> > > > deriv_matrix_exp(_dim, vector<vector<vector<double> > >(_dim, vector<vector<double> >(_dim, vector<double>(_dim, 0.0))));
      
      for(int a = 0; a < _dim; a++){
	for(int b = 0; b < _dim; b++){
	  for(int c = 0; c < _dim; c++){
	    for(int d = 0; d < _dim; d++){
	      deriv_matrix_exp[a][b][c][d] = DerivMatrixExp(eigen_values, eigen_vectors, inversed_eigen_vectors, a, b, c, d);
	      
	      if(c != d){
		deriv_matrix_exp[a][b][c][d] =  current->edge_length * substitution_rate_matrix[i](c,d) * deriv_matrix_exp[a][b][c][d];
	      }	      
	    }
	  }
	}
      }

      for(int j = 0; j < _number_of_samples; j++){
	int ns_base_id = j*_number_of_mixtures*(_dim-1)*_dim + i*(_dim-1)*_dim;
	
	for(int k = 0; k < _dim; k++){
	  current->fd_values[GetTripleArrayId(j,i,k)] = 0.0;	
	  for(int l = 0; l <_dim; l++){
	    if(k != l){
	      current->ns_values[ns_base_id+ NsId(k,l)] = 0.0;
	    }
	  }
	}
	double sum = 0.0;
	for(int b = 0; b < _dim; b++){
	  double outside_plus = -DBL_MAX;
	  for(int c = 0; c < _dim; c++){
	    if(sister->inside_values[GetTripleArrayId(j,i,c)] != -DBL_MAX){
	      double value = log_sister_probability_matrix(b,c) + sister->inside_values[GetTripleArrayId(j,i,c)];
	      outside_plus = outside_plus == -DBL_MAX ? value : logsumexp(outside_plus, value);
	    }
	  }
	  outside_plus += current->parent->outside_values[GetTripleArrayId(j,i,b)];
	  
	  for(int a = 0; a < _dim; a++){
	    double edge_prob = fmath::expd(current->inside_values[GetTripleArrayId(j,i,a)] + outside_plus - _column_log_likelihood[j][i]);
	    for(int k = 0; k < _dim; k++){
	      current->fd_values[GetTripleArrayId(j,i,k)] += deriv_matrix_exp[b][a][k][k] * edge_prob;      
	      for(int l = 0; l <_dim; l++){
		if(substitution_rate_matrix[i](k,l) > 0){	
		  current->ns_values[ns_base_id + NsId(k,l)] +=  deriv_matrix_exp[b][a][k][l] * edge_prob;		
		}
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
    CalcTreeModelSufficientStatistics(current->left,parameters,0);
  }
  if(current->right != NULL){
    CalcTreeModelSufficientStatistics(current->right,parameters,1);
  }
}
