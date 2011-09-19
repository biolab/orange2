/*
    This file is part of Orange.

    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef LINEAR_HPP
#define LINEAR_HPP

// LIBLINEAR header

#include "linear.h"

#include <map>
#include "classify.hpp"
#include "learn.hpp"
#include "orange.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "examples.hpp"

// Alternative model save/load routines (using iostream, needed for in memory serialization)
int linear_save_model_alt(string &, model *);
model *linear_load_model_alt(string &);

WRAPPER(ExampleTable)

class ORANGE_API TLinearLearner : public TLearner{
public:
	__REGISTER_CLASS
	
	CLASSCONSTANTS(Lossfunction1_old_) enum {L2_LR, L2Loss_SVM_Dual, L2Loss_SVM, L1Loss_SVM_Dual }; //For backwards compatibility with 1.4 version.
	CLASSCONSTANTS(Lossfunction1) enum {L2R_LR, L2R_L2Loss_SVC_Dual, L2R_L2Loss_SVC, L2R_L1Loss_SVC_Dual, MCSVM_CS, L1R_L2Loss_SVC, L1R_LR, L2R_LR_Dual};
	CLASSCONSTANTS(LIBLINEAR_VERSION: VERSION=180)
	
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

WRAPPER(LinearLearner)
WRAPPER(LinearClassifier)

#endif /* LINEAR_HPP */
