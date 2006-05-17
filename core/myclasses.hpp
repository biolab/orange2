// We need this for MYMODULE_API and for REGISTER_CLASS
#include "mymodule_globals.hpp"

#include "learn.hpp"
#include "classify.hpp"
#include "random.hpp"
#include "filter.hpp"

#include "ftree.h"
//#include "estimator.h"
//#include "utils.h"
#include "rndforest.h"
//#include "rfUtil.h"
#include "options.h"
#include <stdio.h>
#include "frontend.h"
extern Options *opt ;
extern featureTree *gFT ;

// Class definitions as usual, except for the MYMODULE_API, __REGISTER_CLASS and //P

class CORE_API TRandomForestLearner : public TLearner {
public:
    __REGISTER_CLASS

    int randomSeed;  //P seed for the random generator
	featureTree *Forest;  
	Options *options; 

	// atributi RF
	int rfNoTrees; //P
	bool rfPredictClass;//P
	float rfSampleProp;//P
	int rfNoSelAttr;//P
	int rfkNearestEqual;//P

    TRandomForestLearner(const int &seed = 0);
    PClassifier operator()(PExampleGenerator, const int &weightID);
};


class CORE_API TRandomForest : public TClassifier {
public:
    __REGISTER_CLASS
	
	featureTree *Forest;  //P
    Options *options;  //P

    PRandomGenerator randomGenerator; //PR random generator

    TRandomForest(PVariable classVar, PRandomGenerator rgen);
    virtual TValue operator()(const TExample &ex);
	virtual PDistribution classDistribution(const TExample &);
};
////////////////////////////7

/*class RandomForestClassifier{
public:
	featureTree *Forest;  //*gFT
    Options *options;
	RandomForestClassifier(featureTree * inputForest);

	TClass Classify(TExample examp);
	TDistr Distribution(TExample examp);

};*/
/*
class RandomForestLearner{
public:
	int state;  // ?????????????
	featureTree *Forest;  //*gFT
	Options *options;

	// atributi RF
	int rfNoTrees;
	bool rfPredictClass;
	double rfSampleProp;
	int rfNoSelAttr;
	int rfkNearestEqual;

	RandomForestLearner();
	RandomForestLearner(int rfNoTrees);

	RandomForestClassifier* GetClassifier();
};
*/