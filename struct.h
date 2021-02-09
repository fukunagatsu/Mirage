#ifndef STRUCT_H
#define STRUCT_H

#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef struct lnode {
  char* name;
  double edge_length;
  double* inside_values;
  double* outside_values;
  double* fd_values;
  double* ns_values;
  double* logL;
  int* c;
  int* reconstruction;
  struct lnode *parent;
  struct lnode *right;
  struct lnode *left;
} Node;

typedef struct lparameter {
  vector<double> mixture_probability;
  vector<VectorXd> init_prob;
  vector<double> alpha;
  vector<double> beta;
  vector<double> gamma;
  vector<vector<double> > parameter;
  double gamma_distribution_parameter;
  vector<double> rate_parameter;
} Parameter;

#endif
