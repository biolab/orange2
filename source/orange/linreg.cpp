/*
When using MS Visual C++ 6:
	1) (reference: Q240014 (bug: "If a list has over 32K elements, list::sort removes the elements.")
	The bug is in the header file <list>.
	Also might be of interest: "If you are sorting through a list larger than 32K, increasing _MAXN from 15 to 25 would improve the performance."

	2) Q265109
	"The Standard Template Library (STL) list::sort function doesn't sort a list of pointers when a predicate function is defined for sorting."
	Cause:	"You can specify the predicate function for sorting, but it will always call the default (greater) function."


sigh..MS.. -_-


*/


#ifdef _MSC_VER
#pragma warning(disable: 4786 4244) // warning C4786: symbol greater than 255 characters
#endif

namespace nrutil {

float pythag(float a, float b);
void svdcmp(float **a, int m, int n, float w[], float **v);
void svbksb(float **u, float w[], float **v, int m, int n, float b[], float x[]);

#include "nrutil.h"
/*#include "svbksb.cpp"
#include "svdcmp.cpp"
#include "pythag.cpp"*/
}


#include "linreg.ppp"

#include <list>
#include <math.h>

//TODO remove unneeded
#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "distance.hpp"
#include "contingency.hpp"
#include "classify.hpp"
#include "classfromvar.hpp"
#include "discretize.hpp"


#ifdef _MSC_VER
/*
	(MS STL implementation problem: bug Q265109
*/
template<>
struct std::greater<TLR_AtrRes*>  : public binary_function<TLR_AtrRes* ,TLR_AtrRes*, bool> 
{
    bool operator()(const TLR_AtrRes* &_X, const TLR_AtrRes* &_Y) const
    {
		if (_X->sortOrder == TLR_AtrRes::RES_ASC)
			return (_X->atrRes < _Y->atrRes);
		else if (_X->sortOrder == TLR_AtrRes::RES_DESC)
			return (_X->atrRes > _Y->atrRes);
		else // _X->sortOrder == TLR_AtrRes::ATR_ASC
			return (_X->atrVal < _Y->atrVal);
    };
};

#endif // _MSC_VER

struct lrsort_atr : greater<TLR_AtrRes*> {
	bool operator()(const TLR_AtrRes* &_X, const TLR_AtrRes* &_Y) const
	{
		return (_X->atrVal < _Y->atrVal);
	}
};

struct lrsort_resasc : greater<TLR_AtrRes*> {
	bool operator()(const TLR_AtrRes* &_X, const TLR_AtrRes* &_Y) const
	{
		return (_X->atrRes < _Y->atrRes);
	}
};

struct lrsort_resdesc : greater<TLR_AtrRes*> {
	bool operator()(const TLR_AtrRes* &_X, const TLR_AtrRes* &_Y) const
	{
		return (_X->atrRes > _Y->atrRes);
	}
};




TTreeSplitConstructor_LR::TTreeSplitConstructor_LR(const float &aml)
: TTreeSplitConstructor(aml)
{}


PClassifier TTreeSplitConstructor_LR::operator()(PStringList &descriptions, PDiscDistribution &subsetSizes,
												 float &quality, int &spentAttribute, PExampleGenerator gen,
												 const int &weightID, PDomainContingency dcont,
												 PDistribution apriorClass, const vector<bool> &candidates,
                                                 PClassifier nodeClassifier
												 )
{
    PLearner wpLRLearner;
	if (!pLRLearner) //ASK must wrap TLinRegLearner into a garbage collector ?
	{
		pLRLearner= mlnew TLinRegLearner; // default options
        wpLRLearner = pLRLearner;
	}


	bool cse= candidates.size()==0;
	vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());
	if (!cse) {
		while(ci!=ce && !*ci) ci++;
		if (ci==ce) return returnNothing(descriptions, subsetSizes, quality, spentAttribute); // no candidates (attributes) left to use
		ci= candidates.begin();
	}

	vector<bool> discrete, continuous;
	TVarList::const_iterator vi(gen->domain->attributes->begin()), ve(gen->domain->attributes->end());

	for(; (cse || (ci!=ce)) && (vi!=ve); vi++)
	{
		if (cse || *(ci++))
		{
			discrete.push_back(  (*vi)->varType == TValue::INTVAR);
			continuous.push_back((*vi)->varType == TValue::FLOATVAR);
			continue;
		} else 
		{
			// don't use this attribute
			discrete.push_back(false);
			continuous.push_back(false);
		}
	}

	int thisAttr= 0, bestAttr= -1;
	float thisThresholdError= 0.0, bestThresholdError= numeric_limits<float>::max();
	quality= 0.0;
	float leftExamples, rightExamples;
	float bestThreshold= 0.0, thisThreshold;

	vi= gen->domain->attributes->begin();
	vector<bool>::const_iterator isDisc(discrete.begin()), isCont(continuous.begin());


	for(; (cse || (ci!=ce)) && (vi!=ve); vi++, isDisc++, isCont++, thisAttr++)
	{
		if (*isDisc)
		{
			//TODO also use discrete attributes when building the tree
			continue;
		}

		if (*isCont)
		{	// current attribute is continuous, and can be used (marked for use in list of candidates)
			
			/*ptc	prepare threshold candidates*/
			//TODO: use a vector instead of the list
			list<TLR_AtrRes*> tcandlist;

			float val, res;
			
			//FOLLOW UP: maybe a better idea: change TLR_AtrRes to store res & index of example, then calc
			//			 res only once (outside the attributes loop), sort it by accessing examples->(current attribute)

			{ PEITERATE(ei, gen)
  			    {
				    if ((*ei)[thisAttr].isRegular())
				    {
					    val= (*ei)[thisAttr].floatV;
                        TValue &rval = nodeClassifier->call(*ei);

                        if (rval.isRegular()) { // TODO
// TODO!!!!!!					      res= fabs((*ei).getClass().floatV - rval.floatV);
					      res= (*ei).getClass().floatV - rval.floatV;
                          tcandlist.push_back(mlnew TLR_AtrRes(val, res));
                        }
				    }
			    }
            }
			
			// sort by attribute value
			tcandlist.sort(lrsort_atr());
			
			// group same values, sum res
			list<TLR_AtrRes*>::iterator tcli(tcandlist.begin()), tcle(tcandlist.end());
			list<TLR_AtrRes*>::iterator tclc(tcandlist.begin()), tclt;
			float num;

			while (tcli != tcle)
			{
				(*tclc)->atrVal= (*tcli)->atrVal;
				(*tclc)->atrRes= (*tcli)->atrRes;
#ifdef _MSC_VER
				// prepare for the sort that will be performed after current loop
				(*tclc)->sortOrder= (ThresholdCandidates == SplitCandidates_ChooseFarthest) ? TLR_AtrRes::RES_DESC : TLR_AtrRes::RES_ASC;
#endif
				tclt = tcli; tclt++; num= 1.0;
				while (tclt != tcle && (*tclc)->atrVal == (*tclt)->atrVal)
				{
					(*tclc)->atrRes += (*tclt)->atrRes;
					num++;
					tclt++;
				}
				if (tclt != tcle)
				{
					(*tclc)->atrVal += (float)((*tclt)->atrVal - (*tclc)->atrVal) / 2.0; // split in middle
				}
				(*tclc)->atrRes /= num; // mean value f residuals
				
				tcli= tclt;
				tclc++;
			}
			// delete all elements from (including) tclc to last one
			tcandlist.erase(tclc, tcle);
			
			// re-sort the list
			if (ThresholdCandidates == SplitCandidates_ChooseNearest)
				tcandlist.sort(lrsort_resasc());
			else
				tcandlist.sort(lrsort_resdesc());

			// how many threshold candidates shall we try ?
			int candsToTry= (tcandlist.size() < NumOfThresholdCandidates) ? tcandlist.size() : NumOfThresholdCandidates;

			// truncate the cand. list if necessary
			if (tcandlist.size() > NumOfThresholdCandidates)
			{
				//TODO
			}

			// re-sort list, this time by attribute value, ascending
			tcandlist.sort(lrsort_atr());
			tcli= tcandlist.begin();
			tclc= tclt= tcli;
			tcle= tcandlist.end();

			/*ptc end*/

			/*ttc try threshold candidates for current attribute*/
			
			// prepare examples
			TExamplePointerTable *leftE=  mlnew TExamplePointerTable(gen->domain);
            PExampleGenerator wLeftE(leftE);
			TExamplePointerTable *rightE= mlnew TExamplePointerTable(gen->domain);
            PExampleGenerator wRightE(rightE);
			// probably could skip following loop with *rightE= mlnew TExamplePointerTable(gen);
			
			PEITERATE(ei, gen)
				if ((*ei)[thisAttr].isRegular() && (*ei).getClass().isRegular())
					rightE->addExample(*ei);
			
			//TODO sort the ExamplePointerTable

			bool atBegin;
			for (; tcli!=tcle;tcli++)
			{
				thisThreshold= (*tcli)->atrVal;

				// 1. move examples into left son
				TExampleIterator epi(rightE->begin()), eprev(rightE->begin());
				for (; epi;)
				{
					if (epi == rightE->begin()) atBegin= true;

					
					if ((*epi)[thisAttr].floatV <= thisThreshold)
					{
						leftE->addExample(*epi);
					}
					// atm, examples with attribute value == threshold go into both sons
					if ((*epi)[thisAttr].floatV < thisThreshold)
					{
						rightE->erase(epi); //ASK will this work ??
						
						if (atBegin)
							epi= wRightE->begin();
						else
							epi= eprev;
					}

					eprev= epi;
					++epi;
				}
				
				// 2. create a classifier over the examples in left & right; if error of split descreases,
				//	  keep this threshold as best

				//BAD if constant model returns MajorityClassifier; besides, weightedError and numOfPoints are private atm
				//TODO decrease singEpsilon (divide by ten) for each level down
				PClassifier leftClassifier =  pLRLearner->call(wLeftE);
				PClassifier rightClassifier = pLRLearner->call(wRightE);
				TLinRegClassifier *uleftClassifier = leftClassifier.AS(TLinRegClassifier);
				TLinRegClassifier *urightClassifier = rightClassifier.AS(TLinRegClassifier);

                // if (!uleftClassifier) --- this is not TLinRegClassifier!

				thisThresholdError= uleftClassifier->weightedError * uleftClassifier->numOfPoints + \
									urightClassifier->weightedError * urightClassifier->numOfPoints;

				//TODO if thisThresholdError == bestThresholdError, we should keep a list of them and,
				//	at the end, choose randomly one of them
				if (thisThresholdError < bestThresholdError)
				{
					bestAttr= thisAttr;
					bestThreshold= thisThreshold;
					bestThresholdError= thisThresholdError;
					leftExamples= leftE->size(); //ASK this ok ?
					rightExamples= rightE->size();
				}

			} // for (candidates list)

			/*ttc end*/			
			
			continue;
		}

	} // for vi (attributes)

	/* TODO something like this
	if (!wins)
		return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

	if (quality<worstAcceptable)
		return returnNothing(descriptions, subsetSizes, spentAttribute);
	*/

	subsetSizes= mlnew TDiscDistribution();
	subsetSizes->addint(0, leftExamples);
	subsetSizes->addint(1, rightExamples);

    TEnumVariable *ubvar = mlnew TEnumVariable();
    PVariable bvar;

	descriptions= mlnew TStringList();
	char str[128];
	sprintf(str, "<%3.3f", bestThreshold);
	descriptions->push_back(str);
    ubvar->values->push_back(str);
	sprintf(str, ">=%3.3f", bestThreshold);
	descriptions->push_back(str);
    ubvar->values->push_back(str);

	ubvar->name= gen->domain->attributes->at(bestAttr)->name;
	spentAttribute = -1;
	return mlnew TClassifierFromVarFD(bvar, gen->domain, bestAttr, subsetSizes, mlnew TThresholdDiscretizer(bestThreshold));

}



/********************************/
/* Lin. reg. learner/classifier */
/********************************/

TLinRegLearner::TLinRegLearner()
: TLearner(NeedsExampleGenerator), simplifyMode(Simplify_Node), minAttrUniqueValues(3), minExamplesInNodeForLM(2),
  distanceCalcMethod(LinRegDistance_Manhattan), singEpsilon(float(10e-8))
{
}

PClassifier TLinRegLearner::operator()(PExampleGenerator gen, const int &weight)
{
	examplegen= gen; // MUST be set back to NULL after learning is done

	atrRemoved= vector<bool>(examplegen->domain->attributes->size(), false);
	atrUseForModel= vector<bool>(examplegen->domain->attributes->size(), true);

	// create our classifier
	TLinRegClassifier* pLRClassifier= mlnew TLinRegClassifier(gen->domain);
    PClassifier wres(pLRClassifier);

	// create the initial regression function
	DoLinReg_SimplifySmall(pLRClassifier);

	// first refinement: remove pseudo-constant attributes
	int res= RemoveConstAttributes(pLRClassifier);
	bool constModel= (res == ConstModel_ConstAtrsFound || res == ConstModel_ConstAtrsNotFound);

	if (!constModel && simplifyMode != Simplify_None)
		SimplifyModel(pLRClassifier, false); // we'd use "true" only if simplification could be inherited

	examplegen = PExampleGenerator(); // DON'T FORGET to set this to NULL if a return is added before this line

	return wres;
}

int TLinRegLearner::GetNumLRAtrs()
{
	int i= 0;
	vector<bool>::iterator ui(atrUseForModel.begin()), ue(atrUseForModel.end());
	vector<bool>::iterator ri(atrRemoved.begin());
	while(ui!=ue) // atrUseForModel & atrRemoved are of same size
	{
		if (*ui && !*ri) i++;
		ui++; ri++;
	}
	return i;
}

int TLinRegLearner::GetNumUsableExamples(TLinRegClassifier* lrClassifier /* = NULL */)
{
	int i= 0, thisAttr;
	vector<bool>::const_iterator ui(atrUseForModel.begin()), ue(atrUseForModel.end());
	vector<bool>::const_iterator ri(atrRemoved.begin());

	PEITERATE(ei, examplegen)
	{
		if ((*ei).getClass().isSpecial()) continue; // if class value is special for current example, skip it

		// check if the example has all usable attributes defined
		for (ui= atrUseForModel.begin(), ri= atrRemoved.begin(), thisAttr= 0; ui!=ue; ui++, ri++, thisAttr++)
		{
			if (*ui && !*ri)
			{
				if ((*ei)[thisAttr].isSpecial()) break; // valueType != valueRegular(==known)
			}
		}
		if (ui!=ue) continue; // didn't get through -> encountered special value

		// example ok
		i++;
	}
	
	if (lrClassifier) lrClassifier->numOfPoints= i;
	return i;
}

int TLinRegLearner::RemoveConstAttributes(TLinRegClassifier* lrClassifier)
{
	/*	Removes attributes with too few unique values among examples;
		TODO: use contingency matrix to find these attributes.
	*/

	bool constModel= false;
	int constAtrs= 0, thisAttr= 0;

	vector<bool>::iterator ri(atrRemoved.begin()), re(atrRemoved.end());
	vector<bool>::iterator ui(atrUseForModel.begin()); //, ue(atrUseForModel.end());
	for(; ri!=re; ri++, ui++, thisAttr++)
	{
		if (GetNumLRAtrs() <= 1) break;
		if (!*ui || *ri) continue; // not to be used or already removed
		
		//TODO use cont. to find out how many unique values an attr has
		set<float> uniqueValuesSet;
		PEITERATE(ei, examplegen)
		{
			if ((*ei)[thisAttr].isRegular()) // unknowns don't count
			{
				uniqueValuesSet.insert((*ei)[thisAttr].floatV);
			}
			if (uniqueValuesSet.size() >= minAttrUniqueValues) break;
			//TODO: define and use a nearRange variable; if ABS(example1 - example2) < nearRange, treat the two as same value
		}
		if (uniqueValuesSet.size() < minAttrUniqueValues)
		{
			// "remove" constant attribute
			//RemoveAtr(attrNo);
			*ri= true;
			lrClassifier->ZeroCoefficient(thisAttr);
			constAtrs++;
		}
	} // for
	
	if (CanMakeConstModel(GetNumUsableExamples(lrClassifier), GetNumLRAtrs()))
	{
		// make a backup copy of the two vectors
		// (because it is so in LR; maybe we could drop this when alternate handling of discrete attributes is implemented)
		//FOLLOW UP atm it's not really needed, but let's leave it here until it's decided if 
		//			MajorityClassifier will be used when we have a constant model
		vector<bool> backupRemoved(atrRemoved), backupUseForModel(atrUseForModel);

		MakeConstantModel(lrClassifier);
		constModel= true;

		// restore
		atrRemoved= backupRemoved;
		atrUseForModel= backupUseForModel;
	}
	else if (constAtrs > 0)
	{
		DoLinReg(lrClassifier);
	}

	if (constAtrs > 0)
		return (constModel ? ConstModel_ConstAtrsFound : NotConstModel_ConstAtrsFound);
	else 
		return (constModel ? ConstModel_ConstAtrsNotFound : NotConstModel_ConstAtrsNotFound);
}

int TLinRegLearner::RemoveAttribute(TLinRegClassifier* lrClassifier, float *newSqrError)
{
	/*	Finds the most suitable attribute to be removed from the linear model,
		degenerating it into a constant model if neccesary.
		Returns -1 if no suitable attribute found.
	*/
	TLinRegClassifier tempClassifier(lrClassifier->domain);
	float minError= numeric_limits<float>::max(), curError;
	int bestAttr= -1, thisAttr;
	
	vector<bool>::iterator ui(atrUseForModel.begin()), ue(atrUseForModel.end());
	vector<bool>::iterator ri(atrRemoved.begin());

	for(thisAttr= 0; ui!=ue; ui++, ri++, thisAttr++)
	{
		if (!*ui || *ri) continue; // attribute unusable / removed
		
		DontUseAtr(thisAttr);
		if (GetNumLRAtrs() <= 1)
		{
			// make a backup copy of the two vectors
			vector<bool> backupRemoved(atrRemoved), backupUseForModel(atrUseForModel);

			MakeConstantModel(&tempClassifier);

			// restore
			atrRemoved= backupRemoved;
			atrUseForModel= backupUseForModel;

			//TODO	IF we are gonna use the MajorityClassifier for the constant model, create one here
			//		and get it's error
			// Will the error be comparable ???
			// until then, a little cheat: after removal the new error is same as the old error was
			curError= lrClassifier->squareError;
		} else {
			DoLinReg(&tempClassifier);
			curError= tempClassifier.squareError;
		}
		UseAtr(thisAttr);
		if (curError < minError)
		{
			bestAttr= thisAttr;
			minError= curError;
		}
	}
	
	*newSqrError= minError;
	return bestAttr;
}

int TLinRegLearner::FindAttrToRemove(TLinRegClassifier* lrClassifier)
{
	float oldError, newError;
	int attrNo;

	if(GetNumLRAtrs() <= 1) return -1;
	oldError= lrClassifier->squareError;
	attrNo= RemoveAttribute(lrClassifier, &newError);
	if(attrNo != -1 && newError <= oldError)
		return attrNo;
	else
		return -1;
}

bool TLinRegLearner::SimplifyModel(TLinRegClassifier* lrClassifier, bool remove)
{
	/*	Removes attributes from model until the removing wouldn't lower error.
		If remove == false, it only changes the classifier (regression function) and not the linear model itself;
		in any event it reports back whether the model can be simplified or not.

		for now, remove will always be false;
		SimplifyModel will be used with remove = true if/when inherited simplification will be added.
	*/

	int bestAttrToRemove;
	bool change= false;
	// make a backup copy of the two vectors that control linear model attributes
	vector<bool> backupRemoved(atrRemoved), backupUseForModel(atrUseForModel);

	do {
		bestAttrToRemove= FindAttrToRemove(lrClassifier) ;
		
		if(bestAttrToRemove != -1)
		{
			RemoveAtr(bestAttrToRemove);
			DoLinReg(lrClassifier);
			change= true ;
		}
	} while(bestAttrToRemove != -1);

	if(change && !remove)
	{
		atrRemoved= backupRemoved;
		atrUseForModel= backupUseForModel;
	}
	return change;
}

void TLinRegLearner::DoLinReg_SimplifySmall(TLinRegClassifier* lrClassifier)
{
	/*	If enough examples, calls DoLinReg;
		otherwise, first tries to simplify the model by removing attributes.
		In extremis, the result is a constant model.
	*/

	GetNumUsableExamples(lrClassifier);

	if(GetNumLRAtrs() > 1 && CanMakeLinearModel(lrClassifier->numOfPoints, GetNumLRAtrs()))
	{
		DoLinReg(lrClassifier);
		return;
	}


	switch(RemoveConstAttributes(lrClassifier))
	{
		// constant model:
		case ConstModel_ConstAtrsFound:
			return;
		case ConstModel_ConstAtrsNotFound:
			return;
		
		// linear model:
		case NotConstModel_ConstAtrsNotFound:
			DoLinReg(lrClassifier);
			break;
		case NotConstModel_ConstAtrsFound:
			// although we're still dealing with a linear model, some attributes were removed from it
			GetNumUsableExamples(lrClassifier);
			
			if(!CanMakeLinearModel(lrClassifier->numOfPoints, GetNumLRAtrs()))
			{
				if(simplifyMode != Simplify_None) // are we allowed to further simplify the linear model ?
				{
					if(!SimplifyModel(lrClassifier, false))
					{
						// linear model cannot be simplified
						MakeConstantModel(lrClassifier) ;
					}
				} else {
					// simplification forbidden
					MakeConstantModel(lrClassifier) ;
				}
			}
			break;
	}
}

void TLinRegLearner::DoLinReg(TLinRegClassifier* lrClassifier)
{
	/*	Calculates the coefficients of the linear model (regression function) based on given examples.
	*/
	
	int lrAtrs= GetNumLRAtrs(), lrPoints= GetNumUsableExamples(lrClassifier);
//	if (lrPoints < lrAtrs) raiseError("lin. reg.: more equations than unknowns");

	int k;
	float wmax, wmin, *w, *w0, *b;
	float **u,**v;
	w= nrutil::vector(1, lrAtrs);
	b= nrutil::vector(1, lrPoints); // here we will copy the class values of the examples
	u= nrutil::matrix(1, lrPoints, 1, lrAtrs);
	v= nrutil::matrix(1, lrAtrs, 1, lrAtrs);
	w0= nrutil::vector(1, lrAtrs) ;
	/* copy data into u */
	vector<bool>::const_iterator ui(atrUseForModel.begin()), ue(atrUseForModel.end());
	vector<bool>::const_iterator ri(atrRemoved.begin());
	int row= 1, col= 1, thisAttr;

	PEITERATE(ei, examplegen)
	{
		if ((*ei).getClass().isSpecial()) continue; // if class value is special for current example, skip it

		// check if the example has all usable attributes defined
		// we copy the attributes at the same time; if this example shouldn't be copied,
		// we will simply not increase the row index before going on to next example
		for (ui= atrUseForModel.begin(), ri= atrRemoved.begin(), thisAttr= 0, col= 1; ui!=ue; ui++, ri++, thisAttr++)
		{
			if (*ui && !*ri)
			{
				if ((*ei)[thisAttr].isSpecial()) break; // valueType != valueRegular(==known)
				u[row][col]= (*ei)[thisAttr].floatV;
				col++;
			}
		}
		if (ui!=ue) continue; // didn't get through -> encountered special value (or class value is special)

		// example ok => next example will be copied into next row
		b[row]= (*ei).getClass().floatV;
		row++;
	}

	/* decompose matrix a */
	nrutil::svdcmp(u, lrPoints, lrAtrs, w, v);

	/* find maximum singular value */
	wmax=0.0 ;
	for(k= 1;k<=lrAtrs;k++)
		if (w[k] > wmax) wmax= w[k];

	wmin= wmax * singEpsilon;    /* define "small" */
//	test(lrPoints >= lrAtrs, "LinReg: More eqations than Unknowns");
	// more eqations then unknowns(N>M) => zero (N-M) smallest w[k]
	// wmin <- (N-M)-th smallest w[k]
	/* zero the "small" singular values */
	for(k=1;k<=lrAtrs;k++)
		if(w[k] < wmin) w[k]= 0.0;

	nrutil::svbksb(u, w, v, lrPoints, lrAtrs, b, w0 );

	// set coefficients
	vector<float>::iterator ci(lrClassifier->lrCoeff.begin());
	for (ui= atrUseForModel.begin(), ri= atrRemoved.begin(), col= 1; ui!=ue; ui++, ri++, ci++)
	{
		if (*ui && !*ri)
		{
			*ci= w0[col++];
		} else {
			*ci= 0.0;
		}
	}

	nrutil::free_vector(w0, 1, lrAtrs);
	nrutil::free_matrix(v, 1, lrAtrs, 1, lrAtrs);
	nrutil::free_matrix(u, 1, lrPoints, 1, lrAtrs);
	nrutil::free_vector(w, 1, lrAtrs);
	nrutil::free_vector(b, 1, lrPoints);

	/* compute Error */
	//TODO
/*	if( Set.WeightData )
	{
		float *weight = new float[nCases()] ;
		setWeight( weight ) ;
		fn->setErr(lrErrW( weight, fn )) ;
		delete weight ;
	} else {
*/
		CalcNonWeigthedError(lrClassifier);
//	}


}

float TLinRegLearner::CalcNonWeigthedError(TLinRegClassifier* lrClassifier)
{
	float diff;
	float v= (float) lrClassifier->GetNonzeroCoeffCount();
	float n= (float) lrClassifier->numOfPoints;
	lrClassifier->squareError= 0.0;

	PEITERATE(ei, examplegen)
	{
		if ((*ei).getClass().isSpecial()) continue; // if class value is special for current example, skip it

		//ASK	should we calculate error over all examples (that have a regular class value),
		//		or only those that were used in creating the regression function ?
		/*
		for (ui= atrUseForModel.begin(), ri= atrRemoved.begin(), thisAttr= 0, col= 1; ui!=ue; ui++, ri++, thisAttr++)
		{
			if (*ui && !*ri)
			{
				(*ei)[thisAttr].floatV;
			}
		}
		if (ui!=ue) continue; // didn't get through -> encountered special value (or class value is special)
		*/

//TODO!!!!		diff= pow((*ei).getClass().floatV - lrClassifier->LRFunc(*ei), 2);
        diff = 42;
		lrClassifier->squareError+= diff;
	}

	lrClassifier->squareError/= n;

	lrClassifier->squareError*= (n+v)/(n-v);
	lrClassifier->weightedError= lrClassifier->squareError;
	
	return lrClassifier->squareError;
}


//TODO!!!! add domain
TLinRegClassifier::TLinRegClassifier(PDomain domain)
: TClassifierFD(domain, false), bConstantModel(false), squareError(0.0), weightedError(0.0)
{
}

TLinRegClassifier::TLinRegClassifier(const TLinRegClassifier &src)
: TClassifierFD(src), bConstantModel(src.bConstantModel), squareError(src.squareError),
  weightedError(src.weightedError)
{
	lrCoeff= src.lrCoeff;
}

int TLinRegClassifier::GetNonzeroCoeffCount()
{
	int v= 0;
	vector<float>::const_iterator ci(lrCoeff.begin()), ce(lrCoeff.end());

	for(; ci!=ce; ci++)
		if(*ci != 0.0) v++;
	
	return v;
}

float TLinRegClassifier::LRFunc(const TExample &exam)
{
    TExample example(domain, exam);
	float fnv= 0.0;
	TExample::const_iterator vi(exam.begin()), ve(exam.end());
	vector<float>::const_iterator ci(lrCoeff.begin());

	for(; (vi!=ve); vi++, ci++)
	{
		if ((*vi).isRegular())
		{
			fnv+= *ci * (*vi).floatV;
			//ASK what should we return if given example doesn't have all needed attributes ?
			//	(the needed attributes being those whose coefficient is not zero)
		}
	}

	return fnv;
}

TValue TLinRegClassifier::operator ()(const TExample &example)
{
	//TODO: return a valueDK or valueDC if given example doesn't have all needed attributes

	float tv= LRFunc(example);
	return TValue(tv);
}