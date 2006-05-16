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

#define TExample int
#define TValue int
#define TClass int
#define TDistr int

class RandomForestClassifier{
public:
	featureTree *Forest;  //*gFT
    Options *options;
	RandomForestClassifier(featureTree * inputForest);

	TClass Classify(TExample examp);
	TDistr Distribution(TExample examp);

};

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

