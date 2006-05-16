/********************************************************************
*
*   Name:          module EXPR
*
*   Description:      deals with representation of
*                          boolean expressions
*
*********************************************************************/

#include <float.h>

#include "expr.h"
#include "bintree.h"
#include "dectree.h"
#include "ftree.h"
#include "error.h"
#include "utils.h"
#include "estimator.h"
#include "options.h"

extern Options *opt ;
extern featureTree *gFT ;

expr::expr(expr &Copy)
{
   root = 0 ;
   copy(Copy) ;
}

expr::~expr()
{
   if (root)
     destroy(root) ;
}


//**********************************************************************
//
//                      destroy
//                      -----------
//
//      deallocates space consumed by boolean expression
//
//**********************************************************************
void expr::destroy(void) 
{ 
	if (root) 
	  destroy(root);  
	root = 0 ;
}


void expr::destroy(exprNode *node)
{
    if (node->left)
      destroy(node->left) ;
    if (node->right)
      destroy(node->right) ;

    delete node ;

}

//**********************************************************************
//
//                      copy
//                      --------
//
//      copies source boolean expression to target
//
//**********************************************************************
void expr::copy(const expr &Source)
{
   modelType = Source.modelType ;

   if (root)
     destroy(root) ;
   if (Source.root)
     dup(Source.root,root) ;
   else
     root = 0 ;
 
   majorClass = Source.majorClass ;
   SBclAttrVal = Source.SBclAttrVal ;
   SBattrVal = Source.SBattrVal ;
   SBcl = Source.SBcl ;
   equalDistance = Source.equalDistance ;
   differentDistance = Source.differentDistance ;
   CAslope = Source.CAslope ;
}


//**********************************************************************
//
//                      operator =
//                      --------
//
//      copies source boolean expression to target
//
//**********************************************************************
void expr::operator= (const expr &Source)
{
    copy(Source) ;
}

//**********************************************************************
//
//                      dup
//                      -------
//
//      duplicates source boolean expression into target
//
//**********************************************************************
void expr::dup(exprNode *Source, PexprNode &Target)
{
    Target = new exprNode ;
	Target->iMain = Source->iMain ;
	Target->iAux = Source->iAux ;
//    Target->nodeType = Source->nodeType ;
//    Target->majorClass = Source->majorClass ;
 //   Target->instances = Source->instances ;

    if (Source->left)
      dup(Source->left, Target->left) ;
    else
      Target->left = 0 ;
    if (Source->right)
      dup(Source->right, Target->right ) ;
    else
      Target->right = 0 ;
}





void expr::createMajority(int Value)
{
    destroy() ;
    modelType = majority ;
    majorClass = Value ;
}



void expr::createKNN(void)
{
    destroy() ;
    modelType = kNN ;

    differentDistance.create(gFT->NoContinuous) ;
    equalDistance.create(gFT->NoContinuous) ;
    CAslope.create(gFT->NoContinuous) ;

    int i ;
    for (i=0 ; i < gFT->NoContinuous ; i++)
    {
        differentDistance[i] = gFT->AttrDesc[gFT->ContIdx[i]].DifferentDistance ;
        equalDistance[i] = gFT->AttrDesc[gFT->ContIdx[i]].EqualDistance ;
        if (differentDistance[i] != equalDistance[i])
            CAslope[i] = double(1.0)/(differentDistance[i] - equalDistance[i]) ;
         else
            CAslope[i] = FLT_MAX ;
    }
}


void expr::createKNNkernel(void)
{
    destroy() ;
    modelType = kNNkernel ;

    differentDistance.create(gFT->NoContinuous) ;
    equalDistance.create(gFT->NoContinuous) ;
    CAslope.create(gFT->NoContinuous) ;

    int i ;
    for (i=0 ; i < gFT->NoContinuous ; i++)
    {
        differentDistance[i] = gFT->AttrDesc[gFT->ContIdx[i]].DifferentDistance ;
        equalDistance[i] = gFT->AttrDesc[gFT->ContIdx[i]].EqualDistance ;
        if (differentDistance[i] != equalDistance[i])
            CAslope[i] = double(1.0)/(differentDistance[i] - equalDistance[i]) ;
         else
            CAslope[i] = FLT_MAX ;
    }
}


void expr::createSimpleBayes(estimation &Estimator, binnode *treeNode)
{
    destroy() ;
	modelType = simpleBayes ;

	int noAttr = gFT->NoAttr ;
	int iCont, iDisc, iClass, iAttr, iEx, iVal ;
    double contValue ;

	// discretize continuous attribute
    Boundary.create(gFT->NoContinuous) ;
	switch  (opt->bayesDiscretization)
	{
		case discrGreedy:
			 for (iCont = 0 ; iCont < Estimator.NoContinuous ; iCont++)
				Estimator.discretizeGreedy(iCont,Boundary[iCont], Estimator.NoDiscrete) ;
			 break ;
		case discrEqFreq:
			 for (iCont = 0 ; iCont < Estimator.NoContinuous ; iCont++)
				Estimator.discretizeEqualFrequency(iCont, opt->bayesEqFreqIntervals, Boundary[iCont]) ;
			 break ;

		default: error("expr::createSimpleBayes", "invalid discretization type for simple bayes") ;
	}

	// create appropriate data structures
    SBclAttrVal.create(gFT->NoClasses+1) ;
	for (iClass = 1 ; iClass <= gFT->NoClasses ; iClass++)
	{
    	iCont = 0 ; 
		SBclAttrVal[iClass].create(noAttr+1) ;
		for (iAttr = 1 ; iAttr <= noAttr ; iAttr++)
			if (gFT->AttrDesc[iAttr].continuous)
			{
	 			SBclAttrVal[iClass][iAttr].create(Boundary[iCont].filled()+2,0.0) ;
                iCont++ ;
			}
			else {
				SBclAttrVal[iClass][iAttr].create(gFT->AttrDesc[iAttr].NoValues+1, 0.0) ;
            }
	}

	// fill the data structure with the weights
    for (iDisc = 1 ; iDisc < Estimator.NoDiscrete ; iDisc++)
       for (iEx = 0 ; iEx < Estimator.TrainSize ; iEx++)
		    SBclAttrVal[Estimator.DiscValues(iEx,0)][gFT->DiscIdx[iDisc]][Estimator.DiscValues(iEx, iDisc)] ++ ;

    for (iCont = 0 ; iCont < Estimator.NoContinuous ; iCont++)
       for (iEx = 0 ; iEx < Estimator.TrainSize ; iEx++)
	   {
		   contValue = Estimator.ContValues(iEx, iCont)  ;
		   if (contValue != NAcont)
		      SBclAttrVal[Estimator.DiscValues(iEx, 0)][gFT->ContIdx[iCont]][Boundary[iCont].lessEqPlace(contValue)+1] ++ ;
           else
              SBclAttrVal[Estimator.DiscValues(iEx, 0 )][gFT->ContIdx[iCont]][0]++ ;
       }
    SBcl.create(gFT->NoClasses+1, 0.0) ;
	for (iClass = 1 ; iClass <=gFT->NoClasses ; iClass++)
		SBcl[iClass] = (treeNode->Classify[iClass]+1)/(treeNode->weight+gFT->NoClasses) ;

   
    SBattrVal.create(noAttr+1) ;
	iCont = 0 ;
	for (iAttr=1 ; iAttr <= noAttr ; iAttr++)
    {
	    if (gFT->AttrDesc[iAttr].continuous)
		{
	 		SBattrVal[iAttr].create(Boundary[iCont].filled()+2,0.0) ;
            iCont++ ;
		}
		else {
			SBattrVal[iAttr].create(gFT->AttrDesc[iAttr].NoValues+1, 0.0) ;
		}
			
		for (iVal = 0 ; iVal < SBattrVal[iAttr].len() ; iVal ++)
		   for (iClass = 1 ; iClass <= gFT->NoClasses ; iClass++)
                SBattrVal[iAttr][iVal] += SBclAttrVal[iClass][iAttr][iVal] ;
	}
}



void expr::predict(binnode *treeNode, int Case, marray<double> &probDist)
{
    switch(modelType)
    {
  
        case majority:
			{
				int i ;
                double m = opt->mEstPrediction ; 

				for (i=1 ; i < probDist.len() ; i++)
				{
                   probDist[i] = (treeNode->Classify[i] + m*gFT->AttrDesc[0].valueProbability[i] ) 
					              /  (treeNode->weight + m) ;
				}
				return ;
            }

        case kNN:
			{
				// find k nearest
									// find k nearest
				marray<sortRec> NN(treeNode->DTrain.filled()) ;
				int i ;
				for (i=0 ; i < treeNode->DTrain.filled() ; i++)
				{
					NN[i].value = treeNode->DTrain[i] ;
					NN[i].key = examplesDistance(treeNode, treeNode->DTrain[i], Case) ;
				}
				NN.setFilled(treeNode->DTrain.filled()) ;
				int k = Mmin(opt->kInNN, treeNode->DTrain.filled()) ;
				NN.sortKdsc(k) ;
                  
				probDist.init(0.0) ;
				for (i=NN.filled()-1 ; i > NN.filled()-1-k ; i--)
					probDist[gFT->DiscData(NN[i].value, 0)] += 1.0 ;
            
				for (i=1 ; i <= gFT->NoClasses ; i++)
				   probDist[i] /= double(k) ;
				
				return  ;
            } 
 
        case kNNkernel:
			{
                // for short description see e.g. Kukar et al(1999):Analysing and improving
                // the diagnosis of ishaemic heart disease with machine learning.
                // Artificial Intelligence in Medicine 16:25-50

                // find k nearest
               	marray<sortRec> NN(treeNode->DTrain.filled()) ;
				int i ;
				for (i=0 ; i < treeNode->DTrain.filled() ; i++)
				{
					NN[i].value = treeNode->DTrain[i] ;
					NN[i].key = examplesDistance(treeNode, treeNode->DTrain[i], Case) ;
				}
				NN.setFilled(treeNode->DTrain.filled()) ;
				int k = Mmin(opt->kInNN, treeNode->DTrain.filled()) ;
				NN.sortKdsc(k) ;
                  
				int noClasses = gFT->AttrDesc[0].NoValues ;
				probDist.init(0.0) ;
                double kr2 = 2*sqr(opt->nnKernelWidth) ;
				for (i=NN.filled()-1 ; i > NN.filled()-1-k ; i--)
					probDist[gFT->DiscData(NN[i].value, 0)] += exp(-sqr(NN[i].key)/kr2) ;
            
                double sumW = 0 ;
                kr2 = sqrt(2.0*Phi) * opt->nnKernelWidth ;
				for (i=1 ; i <= noClasses ; i++)
                {
                   probDist[i] /=  kr2 ; 
				   sumW += probDist[i] ;
                }
                for (i=1 ; i <= noClasses ; i++)
				   probDist[i] /= sumW ;
                
				return  ;
            } 
        case simpleBayes:
			{
				int noClasses = gFT->NoClasses ;
				int noAttr = gFT->NoAttr ;
                int iClass, iAttr, valueIdx, iCont, iDisc ;
				double contValue, denominator, factor ;
                double m = opt->mEstPrediction ; 
                double pAll = 0.0 ;

				for (iClass = 1 ; iClass <= noClasses ; iClass++)
				{
					probDist[iClass] = SBcl[iClass] ;
					
					iCont = 0 ;
					iDisc = 1 ;
					for(iAttr = 1 ; iAttr <= noAttr ; iAttr++)
					{
            			if (gFT->AttrDesc[iAttr].continuous)
						{
                           contValue = gFT->ContData(Case, iCont) ;
						   if (contValue == NAcont) 
							  valueIdx = 0 ;
						   else
						      valueIdx = Boundary[iCont].lessEqPlace(contValue)+1 ;
						   iCont++ ;
						}
						else {
							valueIdx = gFT->DiscData(Case, iDisc) ;
							iDisc ++ ;
						}
						denominator =  (SBattrVal[iAttr][valueIdx] + m) * SBcl[iClass] ;
						if (denominator > 0)
						   factor = (SBclAttrVal[iClass][iAttr][valueIdx] + m * SBcl[iClass]) / denominator ;
                        else factor = 0 ;
						if (factor > 0)
						  probDist[iClass] *= factor ;
					}
					pAll += probDist[iClass] ;
				}
				// normalization to probabilities
				for (iClass = 1 ; iClass <= noClasses ; iClass++)
     				probDist[iClass] /= pAll ;


				return ;
  			}

        default:  error("expr::predict","Cannot evaluate nonexistent model") ;

    }
}



/*
void expr::predict(binnode *treeNode, int Case, exprNode* Node)
{
    #ifdef DEBUG
       if (!Node)
          error("expr::predict", "Invalid structure of model") ;
    #endif
}

*/


char* expr::descriptionString(void)
{
    char *result ;
    switch(modelType)
    {
        case majority:
                    strcpy(result=new char[strlen(gFT->AttrDesc[0].ValueName[majorClass-1])+1],
                           gFT->AttrDesc[0].ValueName[majorClass - 1] ) ;
                    return result ;

        case kNN:
                    result = new char[5] ;
                    sprintf(result,"k-NN") ;
                    return result ;

        case kNNkernel:
                    result = new char[20] ;
                    sprintf(result,"k-NN with kernel") ;
                    return result ;
        case simpleBayes:
                    result = new char[16] ;
                    sprintf(result,"simple Bayes") ;
                    return result ;
			
        default:    error("expr::descriptionString","Cannot print  nonexistent model") ;
					return 0 ;

    }
//   if (root)
//      return descriptionString(root) ;
   return 0 ;
}


/*
char* expr::descriptionString(exprNode* Node)
{
    #ifdef DEBUG
       if (!Node)
          error("expr::descriptionString", "Invalid structure of model") ;
    #endif
	return 0 ;
}
*/

//************************************************************
//
//                        examplesDistance
//                        ----------------
//
//     finds the distance between two examples in attribute space
//
//************************************************************
double expr::examplesDistance(binnode *treeNode, int I1, int I2) 
{
    int i ;
    double distance = 0.0;

    for (i=1 ; i < gFT->NoDiscrete ; i++)
       distance += DAdiff(treeNode, i, I1, I2) ;

    for (i=0 ; i < gFT->NoContinuous ; i++)
       distance += CAdiff(treeNode, i, I1, I2) ;

    return distance ;
}


// ***************************************************************************
//
//                   CAdiff
//              diff function for continuous attribute
//
// ***************************************************************************
double expr::CAdiff(binnode *treeNode, int AttrIdx, int I1, int I2)
{
   double cV1 = gFT->ContData(I1, AttrIdx) ;
   double cV2 = gFT->ContData(I2, AttrIdx) ;
   if (cV1 == NAcont)
      cV1 = treeNode->NAcontValue[AttrIdx] ;
   if (cV2 == NAcont)
      cV2 = treeNode->NAcontValue[AttrIdx] ;
   #if defined(RAMP_FUNCTION)
       return CARamp(AttrIdx, fabs(cV2 - cV1) ) ;
   #else
      return  fabs(cV2 - cV1) / GlobalRegressionTree->valueInterval[AttrIdx] ;
   #endif
}



// ***************************************************************************
//
//                   DAdiff
//              diff function of discrete attribute
//
// ***************************************************************************
double expr::DAdiff(binnode *treeNode, int AttrIdx, int I1, int I2)
{

  // we assume that missing value has value 0
  int dV1 = gFT->DiscData(I1, AttrIdx) ;
  int dV2 = gFT->DiscData(I2, AttrIdx) ;
  if (dV1 == NAdisc)
     dV1 = treeNode->NAdiscValue[AttrIdx] ;
  if (dV2 == NAdisc)
     dV2 = treeNode->NAdiscValue[AttrIdx] ;
   if (dV1 == dV2)
      return  0.0 ;
   else
      return 1.0 ;
}

// ***************************************************************************
//
//                    CARamp
//          ramp function of continuous attribute (or class)
//
// ***************************************************************************
#if defined(RAMP_FUNCTION)
inline double expr::CARamp(int AttrIdx, double distance)
{
  if (distance >= differentDistance[AttrIdx])
     return 1.0 ;
 
  if (distance <= equalDistance[AttrIdx])
     return 0.0 ;

  return  (distance - equalDistance[AttrIdx]) * CAslope[AttrIdx] ;
}
#endif



