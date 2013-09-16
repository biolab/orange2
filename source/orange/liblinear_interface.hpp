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
	float C;			//P Regularization parameter C
	float bias;         //P bias parameter (default -1.0 - no bias)

	TLinearLearner();
	PClassifier operator()(PExampleGenerator, const int & = 0);
};

class ORANGE_API TLinearClassifier : public TClassifierFD {
public:
	__REGISTER_CLASS

	TLinearClassifier() {};

	TLinearClassifier(PDomain domain, struct model * model);

	~TLinearClassifier();

	PDistribution classDistribution(const TExample &);
	TValue operator()(const TExample&);

	PFloatListList weights;	//P Computed feature weights
	float bias; //PR bias

	model *getModel(){ return linmodel; }

private:
	model *linmodel;
	// bias in double precision.
	double dbias;
	int get_nr_values();
};

WRAPPER(LinearLearner)
WRAPPER(LinearClassifier)

#endif /* LINEAR_HPP */
