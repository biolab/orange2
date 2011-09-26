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

#include "classify.hpp"
#include "learn.hpp"
#include "orange.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "examples.hpp"
#include "distance.hpp"
#include "slist.hpp"

#include "libsvm/svm.h"

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
  CLASSCONSTANTS(Kernel: Linear=LINEAR; Polynomial=POLY; RBF=RBF; Sigmoid=SIGMOID; Custom=PRECOMPUTED)
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

	TSVMClassifier(const PVariable & , PExampleTable, svm_model*, svm_node*, PKernelFunc);
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

	int svm_type; //P(&SVMLearner_SVMType)  SVM type (C_SVC=0, NU_SVC, ONE_CLASS, EPSILON_SVR=3, NU_SVR=4)
	int kernel_type; //P(&SVMLearner_Kernel)  kernel type (LINEAR=0, POLY, RBF, SIGMOID, CUSTOM=4)

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
	TSVMClassifierSparse(PVariable var , PExampleTable ex, svm_model* model,
			svm_node* x_space, bool useNonMeta, PKernelFunc kernelFunc)
	:TSVMClassifier(var, ex, model, x_space, kernelFunc){
		this->useNonMeta=useNonMeta;
	}
	bool useNonMeta; //P include non meta attributes
protected:
	virtual svm_node* example_to_svm(const TExample &ex, svm_node* node, float last=0.0, int type=0);
	virtual int getNumOfElements(const TExample& example);
};

#endif

