/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


#ifndef __SVM_HPP
#define __SVM_HPP

#include "classify.hpp"
#include "learn.hpp"

class TlibSVM {
public:
	struct svm_parameter
	{
		int svm_type;
		int kernel_type;
		double degree;	// for poly
		double gamma;	// for poly/rbf/sigmoid
		double coef0;	// for poly/sigmoid

		// these are for training only
		double cache_size; // in MB
		double eps;	// stopping criteria
		double C;	// for C_SVC and C_SVR (cost)
		double nu;	// for NU_SVC and ONE_CLASS
		double p;	// for C_SVR
		
		int max_iter;
		double iter_mult;
	};

	struct svm_node
	{
		int index;
		double value;
	};

	struct svm_problem
	{
		int l;
		double *y;
		struct svm_node **x;
	};

	//
	// svm_model
	//
	struct svm_model
	{
		int n;						// number of SVs
		double *sv_coef;			// sv_coef[i] is the coefficient of SV[i]
		struct svm_node **SV;		// SVs
    struct svm_node *x;
		double rho;					// the constant in the decision function

		struct svm_parameter param;	// parameter
	};

	struct svm_problem *problem;
	struct svm_model *model;

	void svm_train();
	void svm_classify(const struct svm_node *x, double *decision_value);

	TlibSVM();

	TlibSVM(struct svm_model *omodel) {
		model = omodel;
	};

	~TlibSVM() {
		if (model != NULL) {
			// mldelete the problem
			if (model->SV != NULL) {
        if (*(model->SV)!= NULL) {
				  free(*(model->SV));
				  *(model->SV) = NULL;
        }
				free(model->sv_coef);
				model->sv_coef = NULL;
				free(model->SV);
				model->SV = NULL;
			}
			mldelete model;
			model = NULL;
		}
	};
private:
	class Kernel;
	class Cache;
	class Solver;
	class C_SVC_Q;
	class NU_SVC_Q;
	class ONE_CLASS_Q;
	class C_SVR_Q;
	void solve_c_svr(const svm_problem *prob, const svm_parameter *param,double *alpha, double& obj, double& rho);
	void solve_c_svc(const svm_problem *prob, const svm_parameter* param,double *alpha, double& obj, double& rho);
	void solve_nu_svc(const svm_problem *prob, const svm_parameter* param,double *alpha, double& obj, double& rho);
	void solve_one_class(const svm_problem *prob, const svm_parameter* param,double *alpha, double& obj, double& rho);
};

// A wrapper for libsvm
class TSVMLearner : public TLearner {
public:
  __REGISTER_CLASS

  // model definition
	
  TSVMLearner();
  TlibSVM *svm;

  // definition of the parameters
	int svm_type; //P (>type) SVM type (C_SVC=0, NU_SVC, ONE_CLASS, C_SVR=3)
	int kernel_type; //P (>kernel) kernel type (LINEAR=0, POLY, RBF, SIGMOID=3)
	float degree;	//P polynomial kernel degree
	float gamma;	//P poly/rbf/sigm parameter
	float coef0;	//P poly/sigm parameter
	float cache_size; //P cache size in MB
	float eps;	//P stopping criteria
	float C;	//P for C_SVC and C_SVR
	float nu;	//P for NU_SVC and ONE_CLASS
	float p;	//P for C_SVR
	int max_iter; //P maximal number of iterations
	float iter_mult; //P epsilon multiplier

  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};


class TSVMClassifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  TlibSVM *svm;
  
	TSVMClassifier() {};

  ~TSVMClassifier() {
    mldelete svm;
  }
	
	TSVMClassifier(PDomain, TlibSVM *mod);

	virtual TValue operator ()(const TExample &);
};


//////////////////////////////////////////////////////////////




#endif

