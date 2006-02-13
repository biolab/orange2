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

#include "table.hpp"
class TSVMLearner;
class TSVMClassifier;

/*##########################################
##########################################*/

#ifndef _LIBSVM_H
#define _LIBSVM_H

#ifdef __cplusplus
extern "C" {
#endif

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
enum { LINEAR, POLY, RBF, SIGMOID, CUSTOM };	/* kernel_type */

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	double degree;	/* for poly */
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

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
double svm_get_svr_probability(const struct svm_model *model);

void svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_destroy_model(struct svm_model *model);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */

/*##########################################
##########################################*/

#include <iostream>
#include "classify.hpp"
#include "learn.hpp"
#include "orange.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "examples.hpp"
#include "distance.hpp"

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
	
	//parameters
	int svm_type; //P  SVM type (C_SVC=0, NU_SVC, ONE_CLASS, EPSILON_SVR=3, NU_SVR=4)
	int kernel_type; //P  kernel type (LINEAR=0, POLY, RBF, SIGMOID, CUSTOM=4)
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
	
	PKernelFunc kernelFunc;	//P custom kernel function

	PExampleTable tempExamples;

	TSVMLearner();

	PClassifier operator()(PExampleGenerator, const int & = 0);
};



class ORANGE_API TSVMClassifier : public TClassifier{
public:
	__REGISTER_CLASS

	TSVMClassifier(PVariable, PExampleTable, svm_model*, svm_node*);
	~TSVMClassifier();

	TValue operator()(const TExample&);
	PDistribution classDistribution(const TExample &);

	PExampleTable supportVectors; //P support vectors
	PExampleTable examples;	//P examples used to train the classifier
	PKernelFunc kernelFunc;	//P custom kernel function
	const TExample *currentExample;

private:
	svm_model *model;
	svm_node *x_space;
};


#endif

