

#ifndef __LINREG_HPP
#define __LINREG_HPP

#include "tdidt_split.hpp"
#include "learn.hpp"
#include "classify.hpp"

WRAPPER(ExampleGenerator);
WRAPPER(LinRegLearner);

/* A spit constructor that builds the regression tree (with linear regression in leaves)
*/
class TTreeSplitConstructor_LR : public TTreeSplitConstructor {
public:
	__REGISTER_CLASS

	PLinRegLearner pLRLearner; //P node learner

	enum {SplitCandidates_ChooseFarthest, SplitCandidates_ChooseNearest};
	int ThresholdCandidates; //P determines whether nearest/farthest threshold candidates will be tested
	/*	determines which threshold candidates will be tested when searching for best split
		for each tested attribute: doesn't matter if 
		  NumOfThresholdCandidates > (# of different values of current attribute (among learning examples))

		SplitCandidates_ChooseFarthest: candidates farthest from regression function are used (using mean value for "distance")
		SplitCandidates_ChooseNearest: candidates nearest to regression function are used (using mean value for "distance")
	*/
	int NumOfThresholdCandidates; //P number of threshold candidates to test
	/*	determines how many threshold candidates will be tested when searching for best split
		"test" => calculates error if a split would be performed with pair (attribute, threshold)
	*/

	TTreeSplitConstructor_LR(const float & = 0);

	virtual PClassifier operator()( PStringList &descriptions,
									PDiscDistribution &subsetSizes,
									float &quality, int &spentAttribute,

									PExampleGenerator, const int &weightID = 0,
									PDomainContingency = PDomainContingency(), // there's nothing we coudld use in the contingency matrix
									PDistribution apriorClass = PDistribution(),
									const vector<bool> &candidates = vector<bool>(),
                                    PClassifier nodeClassifier = PClassifier()
								  );
};


class TLR_AtrRes {
public:
	float atrVal;
	float atrRes;
  
#ifdef _MSC_VER
	//TODO try using STLport.org's or some other STL implementation
	// => because if compiled under windows, with MSVC++ and MS's impl.,
	//	it will always use the default sort predicate (greater) when one (whichever) is specified in list::sort
	// Until then, the sortOrder member is neccesary :(
	int sortOrder;
	enum {ATR_ASC, RES_ASC, RES_DESC};
#endif

#ifndef _MSC_VER
	TLR_AtrRes(float atrV, float atrR) : atrVal(atrV), atrRes(atrR) {};
	TLR_AtrRes() : atrVal(0.0), atrRes(0.0) {};
#else
	TLR_AtrRes(float atrV, float atrR) : atrVal(atrV), atrRes(atrR), sortOrder(ATR_ASC) {};
	TLR_AtrRes() : atrVal(0.0), atrRes(0.0), sortOrder(ATR_ASC) {};
#endif
};

/*
	A pair Learner/Classifier for linear regression. For a "lrtree", this classifier is the node's nodeClassifier
*/

class TLinRegClassifier : public TClassifierFD {
	friend class TLinRegLearner;
public:
	__REGISTER_CLASS

	int numOfPoints;	// the number of points: == num. of examples from which we made/will make the regression function
	float squareError;
	float weightedError;
private:
	vector<float> lrCoeff;	// coefficients of the regression function; if constant model, all coefficients are 0.0
	bool bConstantModel;	// if constant model, TLinRegLearner will return a TMajorityClassifier instead of a TLinRegClassifier

	
	//obsolete	float constModelValue;	// the classifier returns this if model is constant
	inline void ZeroCoefficient(int attrNo)
	{ lrCoeff.at(attrNo)= 0.0; }
	int GetNonzeroCoeffCount();

	float LRFunc(const TExample &example);
	//use overload?? TValue LRFunc(const TExample &example);

public:
	TLinRegClassifier(PDomain = PDomain());
	TLinRegClassifier(const TLinRegClassifier &); // copy constructor
	
	virtual TValue operator ()(const TExample &);
};


class TLinRegLearner : public TLearner {
public:
	__REGISTER_CLASS

	PExampleGenerator examplegen; /* temporary ptr to our example generator; it's set at beginning of operator(),
		and set back to NULL at the end of said function
		*/

	enum {Simplify_None, Simplify_Node};
	int simplifyMode;	//P no simplification / node simplification
	/*	no simplification:	 the linear model in nodes will not be simplified (all attributes will be used when calculating regression function)
		node simplification: the linear model of node can be simplified, but the simplification (ie info about removed attributes)
							 does not propagate to descendants of the node
	*/

	int minAttrUniqueValues;	//P minimum number of unique values an attribute must have among examples to be used in linear model
	int minExamplesInNodeForLM;	//P minimum number of examples that must be in node
								//	=> if criteria not met, the node will use a constant model
	enum {LinRegDistance_Manhattan, LinRegDistance_Euclid};
	int distanceCalcMethod;	//P how the distance will be calculated when weighting examples (manhattan or euclid distance)
	float singEpsilon;		//P initial singular epsilon

	
	vector<bool> atrRemoved; // attributes that were removed from the lin. model
	vector<bool> atrUseForModel; // attributes that should be used when making linear model

	TLinRegLearner();
//	TLinRegLearner(const TLinRegLearner &); // copy constructor
	virtual PClassifier operator()(PExampleGenerator, const int & =0);

	inline void RemoveAtr(int attrNo)  // atrRemoved[attrNo] becomes true
		{ atrRemoved.at(attrNo)= true; }
	inline void RestoreAtr(int attrNo) // atrRemoved[attrNo] becomes false
		{ atrRemoved.at(attrNo)= false; }
	inline void DontUseAtr(int attrNo) // atrUseForModel[attrNo] becomes false
		{ atrUseForModel.at(attrNo)= false; }
	inline void UseAtr(int attrNo)     // atrUseForModel[attrNo] becomes true
		{ atrUseForModel.at(attrNo)= true; }

	int GetNumLRAtrs();
	int GetNumUsableExamples(TLinRegClassifier* lrClassifier = NULL);
		// returns number of examples that can be used to create regression func. with current setting (atrRemoved, atrUseForModel)

	enum {NotConstModel_ConstAtrsNotFound, NotConstModel_ConstAtrsFound, ConstModel_ConstAtrsNotFound, ConstModel_ConstAtrsFound};
	int RemoveConstAttributes(TLinRegClassifier* lrClassifier); // removes attributes with too few unique values among examples

	inline bool CanMakeLinearModel(int numExamples, int numAttrs)
		{ return ((numExamples >= 2*numAttrs) && (numExamples >= minExamplesInNodeForLM)) ? true:false; }
	inline bool CanMakeConstModel(int numExamples, int numAttrs)
		{ return((numExamples <= numAttrs || numAttrs <= 1) ? true:false); }

	// MakeConstantModel: examples will be approximated with a constant (class mean value from examples in current node)
	//  => the Learner will create and return a MajorityClassifier (which is a mean value cl. for regression trees)
	inline void MakeConstantModel(TLinRegClassifier* lrClassifier)
	{ lrClassifier->bConstantModel= true; }

	int RemoveAttribute(TLinRegClassifier* lrClassifier, float *newSqrError);
	int FindAttrToRemove(TLinRegClassifier* lrClassifier);
		// returns the attribute without which the error of the lin. model is lowest
		// (and lower than the original error)
		// returns -1 if no such attribute found
	bool SimplifyModel(TLinRegClassifier* lrClassifier, bool remove);
		// simplifies the classifier and/or linear model by removing attributes
		// if remove == false, doesn't change linear model
		// in any case it lets us know whether the model can be simplified
	void DoLinReg(TLinRegClassifier* lrClassifier); //ASK  should I wrap TLinRegClassifier and use PLinRegClassifier ??
	void DoLinReg_SimplifySmall(TLinRegClassifier* lrClassifier); // if not enough examples, simplifies linear model
	float CalcNonWeigthedError(TLinRegClassifier* lrClassifier);
	
	//TODO weights
};

#endif
