#ifndef MIRAGE_H
#define MIRAGE_H

#define EIGEN_NO_DEBUG 
#define EIGEN_DONT_PARALLELIZE 
#define EIGEN_MPL2_ONLY 

#include "mirage_parameters.h"
#include <Eigen/LU>
#include <float.h>
#include <unsupported/Eigen/MatrixFunctions>

class Mirage{
 public:
  Mirage(){
    _number_of_mixtures = 0;
    _dim = 0;
    _number_of_samples = 0;
    _model_id = 0;
    _mixture_method_id = 0;
    _number_of_em_iteration = 0;
    _estimation_flag = true;
  }
  void Train(MirageParameters& parameters);
  void Estimate(MirageParameters& parameters);
 private:
  bool _estimation_flag;
  int _number_of_mixtures;
  int _dim;
  int _model_id;
  int _mixture_method_id;
  int _number_of_samples;
  int _number_of_em_iteration;
  vector<vector<double> > _column_log_likelihood;
  vector<vector<vector<double> > > _init_prob_sufficient_statistics;
  vector<vector<double> > _responsibility;
  vector<double> _mixture_id;
  vector<double> _log_likelihood_array;
  
  double logsumexp(double x,double y);
  void CalcInsideValues(Node* current, MirageParameters& parameters);
  void CalcOutsideValues(Node* current, MirageParameters& parameters, int id);
  void CalcTreeModelSufficientStatistics(Node* current, MirageParameters& parameters, int id, vector<VectorXd> &init, vector<VectorXd> &fd, vector<MatrixXd> &ns);
  int GetTripleArrayId(int sample_id, int mixture_id, int element_id);
  void CalcColumnLogLikelihood(Node* root, MirageParameters& parameters);
  double CalcDataLikelihood(MirageParameters& parameters);
  void CalcResponsibility(MirageParameters& parameters);
  void SaveOldParameter(MirageParameters& m_parameter, Parameter& old_parameter);
  void Output(MirageParameters& parameters, int id);
  void Initiallize(MirageParameters& parameters);
  void CalcRateParameter(vector<VectorXd>& fd, vector<MatrixXd>& ns, MirageParameters& parameters);
  void CalcGammaParameter(MirageParameters& parameters);
  int NsId(int j, int k);

  bool NewParameterEstimation(Node* current, MirageParameters& parameters, vector<VectorXd> &init, vector<VectorXd> &fd, vector<MatrixXd> &ns);
  double CalcPartialAlpha(int mixture_id, VectorXd& fd, MatrixXd& ns, MirageParameters& parameters);
  double CalcPartialGamma(int mixture_id, VectorXd& fd, MatrixXd& ns, MirageParameters& parameters);
  void GradientDescent(int id, vector<VectorXd>& fd, vector<MatrixXd>& ns, MirageParameters& parameters);
  double Qem(VectorXd& fd, MatrixXd& ns, double a, double g, double rate_factor);
  void SetOldParameter(Parameter& old_parameter, MirageParameters& m_parameter);
  void HistoryReconstruction(Node* current, MirageParameters& parameters);
  void HistoryTraceBack(ofstream& ofs, Node* current, int output_style);
  void OutputReconstruction(ofstream& ofs, Node* current, int output_style);
  MatrixXd SetTempSubstitutionRateMatrix(MirageParameters& parameters);
  void CheckInsideOutside(Node* current, int id);
  int CalcCashID(int* cash_index, int id);
  void FreeAndMalloc(Node* current);
  bool IsPatternMixture();
  void AddEpsilon(MatrixXd& matrix);

  void CalcOutsideValuesDPM(Node* current, MirageParameters& parameters, int id);
};
#endif
