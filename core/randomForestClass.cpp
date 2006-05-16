#include "ftree.h"
//#include "estimator.h"
//#include "utils.h"
#include "rndforest.h"
//#include "rfUtil.h"
#include "options.h"
#include <stdio.h>
#include "frontend.h"

#include "myclasses.ppp"

#include "errors.hpp" // for raiseError

extern Options *opt ;
extern featureTree *gFT ;

TRandomForestLearner::TRandomForestLearner(const int &seed)
: randomSeed(seed)
{}

/*RandomForestLearner::RandomForestLearner(int rfNoTrees){
	this->rfNoTrees=rfNoTrees;
}
*/
PClassifier TRandomForestLearner::operator()(PExampleGenerator egen, const int &weightID)
{
  PRandomGenerator randGen = new TRandomGenerator(randomSeed);
	gFT=NULL;
	opt=NULL;
	Forest = new featureTree();
	options = new Options();
	gFT = Forest;
	opt = options;
	//opt->readConfig("credita.par") ;
	TExampleTable &table = dynamic_cast<TExampleTable &>(toExampleTable(egen).getReference());
	gFT->readDescription(table);
	gFT->readData(table);
	gFT->state = empty ;
	gFT->learnRF = TRUE ;
	gFT->buildForest();
    //
	// nekaj delat
	//
	TRandomForest* ranForest = new TRandomForest(egen->domain->classVar, randGen);
	ranForest->options = opt;
	ranForest->Forest = gFT;

	gFT=NULL;
	opt=NULL;
  
  return ranForest;
}


TRandomForest::TRandomForest(PVariable classVar, PRandomGenerator rgen)
: TClassifier(classVar),
  randomGenerator(rgen)
{
  if (classVar->varType != TValue::INTVAR)
    raiseError("MyClassifier cannot work with a non-discrete attribute '%s'", classVar->name.c_str());
}


TValue TRandomForest::operator()(const TExample &)
{
  return TValue(randomGenerator->randint(classVar->noOfValues()));
}

