/*
 
 Copyright (c) 2007-2010 The LIBLINEAR Project.
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 
 1. Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 
 3. Neither name of copyright holders nor the names of its contributors
 may be used to endorse or promote products derived from this software
 without specific prior written permission.
 
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
*/



#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif
	
	struct feature_node
	{
		int index;
		double value;
	};
	
	struct problem
	{
		int l, n;
		int *y;
		struct feature_node **x;
		double bias;            /* < 0 if no bias term */  
	};
	
	enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL }; /* solver_type */
	
	struct parameter
	{
		int solver_type;
		
		/* these are for training only */
		double eps;	        /* stopping criteria */
		double C;
		int nr_weight;
		int *weight_label;
		double* weight;
	};
	
	struct model
	{
		struct parameter param;
		int nr_class;		/* number of classes */
		int nr_feature;
		double *w;
		int *label;		/* label of each class */
		double bias;
	};
	
	struct model* train(const struct problem *prob, const struct parameter *param);
	void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, int *target);
	
	int predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
	int predict(const struct model *model_, const struct feature_node *x);
	int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);
	
	int save_model(const char *model_file_name, const struct model *model_);
	struct model *load_model(const char *model_file_name);
	
	int get_nr_feature(const struct model *model_);
	int get_nr_class(const struct model *model_);
	void get_labels(const struct model *model_, int* label);
	
	void free_model_content(struct model *model_ptr);
	void free_and_destroy_model(struct model **model_ptr_ptr);
	void destroy_param(struct parameter *param);
	
	const char *check_parameter(const struct problem *prob, const struct parameter *param);
	int check_probability_model(const struct model *model);
	void set_print_string_function(void (*print_func) (const char*));
	
#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

#ifndef _TRON_H
#define _TRON_H

class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;
	
	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
};

class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
	~TRON();
	
	void tron(double *w);
	void set_print_string(void (*i_print) (const char *buf));
	
private:
	int trcg(double delta, double *g, double *s, double *r);
	double norm_inf(int n, double *x);
	
	double eps;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif


#ifndef LINEAR_HPP
#define LINEAR_HPP

#include <map>
#include "classify.hpp"
#include "learn.hpp"
#include "orange.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "examples.hpp"

int linear_save_model_alt(string &, model *);
model *linear_load_model_alt(string &);

WRAPPER(ExampleTable)

class ORANGE_API TLinearLearner : public TLearner{
public:
	__REGISTER_CLASS
	
	CLASSCONSTANTS(Lossfunction1_old_) enum {L2_LR, L2Loss_SVM_Dual, L2Loss_SVM, L1Loss_SVM_Dual }; //For backwards compatibility with 1.4 version.
	CLASSCONSTANTS(Lossfunction1) enum {L2R_LR, L2R_L2Loss_SVC_Dual, L2R_L2Loss_SVC, L2R_L1Loss_SVC_Dual, MCSVM_CS, L1R_L2Loss_SVC, L1R_LR, L2R_LR_Dual};
	CLASSCONSTANTS(LIBLINEAR_VERSION: VERSION=170)
	
	int solver_type;	//P(&LinearLearner_Lossfunction1) Solver type (loss function1)
	float eps;			//P Stopping criteria
	float C;			//P Regularization parameter

	TLinearLearner();
	PClassifier operator()(PExampleGenerator, const int & = 0);
};

class ORANGE_API TLinearClassifier : public TClassifierFD{
public:
	__REGISTER_CLASS
	TLinearClassifier() {};
	TLinearClassifier(const PVariable &var, PExampleTable examples, model *_model, map<int, int> *indexMap=NULL);
	~TLinearClassifier();

	PDistribution classDistribution(const TExample &);
	TValue operator()(const TExample&);

	PFloatListList weights;	//P Computed feature weights
	PExampleTable examples;	//P Examples used to train the classifier

	model *getModel(){ return linmodel; }
private:
	model *linmodel;
	map<int, int> *indexMap;
};
/*
class O_RANGE_API TLinearClassifierSparse : public TLinearClassifier{
public:
	__R_EGISTER_CLASS
	TLinearClassifierSparse() {};
	TLinearClassifier(const PVariable &var, PExampleTable examples, model *_model, map<int, int> *indexMap=NULL);
	~TLinearClassifier();

	PDistribution classDistribution(const TExample &);
	TValue operator()(const TExample&);
};*/
WRAPPER(LinearLearner)
WRAPPER(LinearClassifier)

#endif /* LINEAR_HPP */
