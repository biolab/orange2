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
	// dodamo
	opt = options;
	//opt->readConfig("credita.par") ;
	TExampleTable &table = dynamic_cast<TExampleTable &>(toExampleTable(egen).getReference());
	gFT->readDescription(table);
	gFT->readData(table);
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

PDistribution TRandomForest::classDistribution(const TExample &origexam)
{ 
    float distr[4];
	TDiscDistribution *distribucija= new TDiscDistribution(distr,4);
	/*
	int contJ = 0, discJ = 1;
	int i = 1;
	gFT=Forest;
	opt = options;
	gFT->clearData();

    for(TExample::const_iterator eei(origexam.begin()), eee(origexam.end()-1); eei != eee; eei++)
      if ((*eei).varType == TValue::FLOATVAR)
		  gFT->ContData.Set(i, contJ, (*eei).isSpecial() ? NAcont : (*eei).floatV);
      else
        gFT->DiscData.Set(i, discJ, (*eei).isSpecial() ? NAdisc : (*eei).intV);
    if (origexam.getClass().isSpecial())
      throw "missing class value";
    gFT->DiscData.Set(i, 0, origexam.getClass().intV);
	//TExample exam = TExample(domain, origexam);
	//gFT->t
	marray<double> probDist(NoClasses+1) ;
	probDist.init(0.0) ;
	gFT->DTest[1]
	if (opt->rfkNearestEqual>0)
	  rfNearCheck(gFT->DTest[1], probDist) ;       
	else if (NoClasses==2 && opt->rfRegType==1)
	  rfCheckReg(gFT->DTest[1], probDist) ;
	else 
	  rfCheck(gFT->DTest[1], probDist) ;
*/
	return NULL;
}


