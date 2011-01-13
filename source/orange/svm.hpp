/*
 
 Copyright (c) 2000-2010 Chih-Chung Chang and Chih-Jen Lin
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


#ifndef __SVM_HPP
#define __SVM_HPP

#include "table.hpp"
class TSVMLearner;
class TSVMClassifier;

/*##########################################
##########################################
 Changes to svm.h from LibSVM distribution:
	- added "TSVMLearner* learner;" and TSVMClassifier* classifier to struct svm_param 
 */

#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 300

#ifdef __cplusplus
extern "C" {
#endif
	
	extern int libsvm_version;
	
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
	
	enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
	enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED, CUSTOM}; /* kernel_type */
	
	struct svm_parameter
	{
		int svm_type;
		int kernel_type;
		int degree;	/* for poly */
		double gamma;	/* for poly/rbf/sigmoid */
		double coef0;	/* for poly/sigmoid */
		
		/* these are for training only */
		double cache_size; /* in MB */
		double eps;	/* stopping criteria */
		double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
		int nr_weight;		/* for C_SVC */
		int *weight_label;	/* for C_SVC */
		double* weight;		/* for C_SVC */
		double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
		double p;	/* for EPSILON_SVR */
		int shrinking;	/* use the shrinking heuristics */
		int probability; /* do probability estimates */
		
		TSVMLearner *learner;
		TSVMClassifier *classifier;
	};
	
	//
	// svm_model
	// 
	struct svm_model
	{
		struct svm_parameter param;	/* parameter */
		int nr_class;		/* number of classes, = 2 in regression/one class svm */
		int l;			/* total #SV */
		struct svm_node **SV;		/* SVs (SV[l]) */
		double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
		double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
		double *probA;		/* pariwise probability information */
		double *probB;
		
		/* for classification only */
		
		int *label;		/* label of each class (label[k]) */
		int *nSV;		/* number of SVs for each class (nSV[k]) */
		/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
		/* XXX */
		int free_sv;		/* 1 if svm_model is created by svm_load_model*/
		/* 0 if svm_model is created by svm_train */
	};
	
	struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
	void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);
	
	int svm_save_model(const char *model_file_name, const struct svm_model *model);
	struct svm_model *svm_load_model(const char *model_file_name);
	
	int svm_get_svm_type(const struct svm_model *model);
	int svm_get_nr_class(const struct svm_model *model);
	void svm_get_labels(const struct svm_model *model, int *label);
	double svm_get_svr_probability(const struct svm_model *model);
	
	double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
	double svm_predict(const struct svm_model *model, const struct svm_node *x);
	double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);
	
	void svm_free_model_content(struct svm_model *model_ptr);
	void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
	void svm_destroy_param(struct svm_parameter *param);
	
	const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
	int svm_check_probability_model(const struct svm_model *model);
	
	void svm_set_print_string_function(void (*print_func)(const char *));
	
	// deprecated
	// this function will be removed in future release
	void svm_destroy_model(struct svm_model *model_ptr); 
	
#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */


/*##########################################
##########################################*/

#include "classify.hpp"
#include "learn.hpp"
#include "orange.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "examples.hpp"
#include "distance.hpp"
#include "slist.hpp"

svm_model *svm_load_model_alt(string& buffer);
int svm_save_model_alt(string& buffer, const svm_model *model);

WRAPPER(ExampleGenerator)
WRAPPER(KernelFunc)
WRAPPER(SVMLearner)
WRAPPER(SVMClassifier)
WRAPPER(ExampleTable)

class ORANGE_API TKernelFunc: public TOrange{
public:
	__REGISTER_ABSTRACT_CLASS
	virtual float operator()(const TExample &, const TExample &)=0;
};

WRAPPER(KernelFunc)

//#include "callback.hpp"

class ORANGE_API TSVMLearner : public TLearner{
public:
	__REGISTER_CLASS

  CLASSCONSTANTS(SVMType: C_SVC=C_SVC; Nu_SVC=NU_SVC; OneClass=ONE_CLASS; Epsilon_SVR=EPSILON_SVR; Nu_SVR=NU_SVR)
  CLASSCONSTANTS(Kernel: Linear=LINEAR; Polynomial=POLY; RBF=RBF; Sigmoid=SIGMOID; Custom=CUSTOM)
  CLASSCONSTANTS(LIBSVM_VERSION: VERSION=LIBSVM_VERSION)

	//parameters
	int svm_type; //P(&SVMLearner_SVMType)  SVM type (C_SVC=0, NU_SVC, ONE_CLASS, EPSILON_SVR=3, NU_SVR=4)
	int kernel_type; //P(&SVMLearner_Kernel)  kernel type (LINEAR=0, POLY, RBF, SIGMOID, CUSTOM=4)
	float degree;	//P polynomial kernel degree
	float gamma;	//P poly/rbf/sigm parameter
	float coef0;	//P poly/sigm parameter
	float cache_size; //P cache size in MB
	float eps;	//P stopping criteria
	float C;	//P for C_SVC and C_SVR
	float nu;	//P for NU_SVC and ONE_CLASS
	float p;	//P for C_SVR
	int shrinking;	//P shrinking
	int probability;	//P probability
	bool verbose;		//P verbose

	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */

	PKernelFunc kernelFunc;	//P custom kernel function

	PExampleTable tempExamples;

	TSVMLearner();
	~TSVMLearner();

	PClassifier operator()(PExampleGenerator, const int & = 0);

protected:
	virtual svm_node* example_to_svm(const TExample &ex, svm_node* node, float last=0.0, int type=0);
	virtual int getNumOfElements(PExampleGenerator examples);
	virtual TSVMClassifier* createClassifier(PVariable var, PExampleTable ex, svm_model* model, svm_node* x_space);
};

class ORANGE_API TSVMLearnerSparse : public TSVMLearner{
public:
	__REGISTER_CLASS
	bool useNonMeta; //P include non meta attributes in the learning process
protected:
	virtual svm_node* example_to_svm(const TExample &ex, svm_node* node, float last=0.0, int type=0);
	virtual int getNumOfElements(PExampleGenerator examples);
	virtual TSVMClassifier* createClassifier(PVariable var, PExampleTable ex, svm_model* model, svm_node* x_space);
};


class ORANGE_API TSVMClassifier : public TClassifierFD{
public:
	__REGISTER_CLASS
	TSVMClassifier(){
		this->model = NULL;
		this->x_space = NULL;
	};

	TSVMClassifier(const PVariable & , PExampleTable, svm_model*, svm_node*);
	~TSVMClassifier();

	TValue operator()(const TExample&);
	PDistribution classDistribution(const TExample &);

	PFloatList getDecisionValues(const TExample &);

	PIntList nSV; //P nSV
	PFloatList rho;	//P rho
	PFloatListList coef; //P coef
	PFloatList probA; //P probA - pairwise probability information
	PFloatList probB; //P probB - pairwise probability information
	PExampleTable supportVectors; //P support vectors
	PExampleTable examples;	//P examples used to train the classifier
	PKernelFunc kernelFunc;	//P custom kernel function

	const TExample *currentExample;

    svm_model* getModel() {return model;}

protected:
	virtual svm_node* example_to_svm(const TExample &ex, svm_node* node, float last=0.0, int type=0);
	virtual int getNumOfElements(const TExample& example);

private:
	svm_model *model;
	svm_node *x_space;
};

class ORANGE_API TSVMClassifierSparse : public TSVMClassifier{
public:
	__REGISTER_CLASS
	TSVMClassifierSparse(){};
	TSVMClassifierSparse(PVariable var , PExampleTable ex, svm_model* model, svm_node* x_space, bool useNonMeta):TSVMClassifier(var, ex, model, x_space){
		this->useNonMeta=useNonMeta;
	}
	bool useNonMeta; //P include non meta attributes
protected:
	virtual svm_node* example_to_svm(const TExample &ex, svm_node* node, float last=0.0, int type=0);
	virtual int getNumOfElements(const TExample& example);
};

#endif

