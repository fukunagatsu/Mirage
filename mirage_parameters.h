#ifndef MIRAGE_PARAMETERS_H
#define MIRAGE_PARAMETERS_H

#define EIGEN_NO_DEBUG 
#define EIGEN_DONT_PARALLELIZE 
#define EIGEN_MPL2_ONLY

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cfloat>
#include <random>
#include "struct.h"

using namespace Eigen;
using namespace std;

class MirageParameters{
 public:
  MirageParameters(){
    _output_file_name = "";
    _init_grad_weight = 0.1;
    _weight_decay_rate = 0.1;
    _partial_par_threshold = 1e-6;
    _weight_threshold = 1.0/100000;
    _max_number = 3;
    _dim = 4;
    _model_id = 0;
    _mixture_method_id = 0;
    _number_of_mixtures = 5;
    _number_of_samples = 0;
    _min_init = 0.5;
    _max_init = 5.0;
    _max_rate = 10.0;
    _loop_threshold = 1.0;
    _gamma_distribution_parameter = 1.0;
    _gamma_dist_sample_size = 10000;
    _loop_max = 200;
  }
  void SetTrainingParameters(int argc, char* argv[]);
  void SetEstimationParameters(int argc, char* argv[]);
  void SetSubstitutionRateMatrix();
  vector<double> CalcGammaRate(double g);
  void CalcAllGammaRate();
  
  int GetMaxNumber(void);
  int GetModelID(void);
  int GetMixtureMethodID(void);
  int GetNumberOfMixtures(void);
  int GetDim(void);
  int GetNumberOfSamples(void);
  int GetLoopMax(void);
  string GetOutputFileName(void);
  double GetInitProb(int i, int j);
  double GetMixtureProbability(int i);
  double GetRateParameter(int i);
  double GetAlpha(int i);
  double GetBeta(int i);
  double GetGamma(int i);
  double GetParameter(int i, int j);
  double GetInitGradWeight(void);
  double GetPartialParThreshold(void);
  double GetWeightDecayRate(void);
  double GetWeightThreshold(void);
  double GetLoopThreshold(void);
  double GetGammaDistributionParameter(void);
  double GetGammaRate(int i, int j);
  Node* GetRoot(void);
  vector<MatrixXd> GetSubstitutionRateMatrix(void);
  void SetAlpha(int i, double alpha);
  void SetBeta(int i, double beta);
  void SetGamma(int i, double gamma);
  void SetParameter(int i, int j, double value);
  void SetMixtureProbability(int i, double value);
  void SetRateParameter(int i, double value);
  void SetInitProb(int i, int j, double value);
  void SetGammaDistributionParameter(double value);
  
 private:
  string _output_file_name;
  int _max_number;
  int _dim;
  int _model_id;
  int _mixture_method_id;
  int _number_of_mixtures;
  int _number_of_samples;
  int _loop_max;
  int _gamma_dist_sample_size;
  double _min_init;
  double _max_init;
  double _init_grad_weight;
  double _weight_decay_rate;
  double _weight_threshold;
  double _partial_par_threshold;
  double _loop_threshold;
  double _gamma_distribution_parameter;
  double _max_rate;
  Node* _root;
  vector<double> _mixture_probability;
  vector<double> _rate_parameter;
  vector<VectorXd> _init_prob;
  vector<double> _alpha;
  vector<double> _beta;
  vector<double> _gamma;
  vector<vector<double> > _parameter;
  vector<vector<double> > _gamma_rate;
  vector<MatrixXd> _substitution_rate_matrix;

  void ReadData(string file_name);
  void ReadParameter(string file_name);
  void ParameterInitialization();
 
  void ParseTree(string& newick, vector<Node*>& leaf_list);
  int GetTripleArrayId(int sample_id, int mixture_id, int element_id);
  string ParseName(string& newick, int& i);
  Node* MakeNode(void);
};
#endif
