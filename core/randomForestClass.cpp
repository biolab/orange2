//#include "ftree.h"
//#include "estimator.h"
//#include "utils.h"
//#include "rndforest.h"
//#include "rfUtil.h"
//#include "options.h"
#include "randomForestClass.hpp"


RandomForestLearner::RandomForestLearner(){
	
}

RandomForestLearner::RandomForestLearner(int rfNoTrees){
	this->rfNoTrees=rfNoTrees;
}

RandomForestClassifier* RandomForestLearner::GetClassifier(void) {
	//RandomForestClassifier* a = new RandomForestClassifier();;
	gFT=NULL;
	opt=NULL;
	Forest = new featureTree();
	options = new Options();
	gFT = Forest;
	opt = options;
	opt->readConfig("credita.par") ;
	gFT->state = empty ;
	Tree->learnRF = TRUE ;
	gFT->buildForest();
    //
	// nekaj delat
	//
	RandomForestClassifier* rfcsfr = new RandomForestClassifier(gFT);
	rfcsfr->options = opt;
	gFT=NULL;
	opt=NULL;
	return rfcsfr;
}

RandomForestClassifier::RandomForestClassifier(featureTree * inputForest){
	Forest = inputForest;
}

