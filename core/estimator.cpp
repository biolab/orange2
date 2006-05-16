#include <stdlib.h>
#include <float.h>

#include "estimator.h"
#include "general.h"
#include "contain.h"
#include "utils.h"
#include "binpart.h"
#include "options.h"

extern Options *opt ;
extern featureTree *gFT ;

//const double epsilon = 1e-7 ;   // computational error

// ***************************************************************************
//
//                     estimate
//     estimate selected attributes with choosen measure
//     and returns the index and type of the best estimated attribute
//
// ***************************************************************************
int estimation::estimate(int selectedEstimator, int contAttrFrom, int contAttrTo, 
                         int discAttrFrom, int discAttrTo, attributeCount &bestType) {
   if (opt->binaryAttributes) {
      opt->binaryAttributes = FALSE ;
      estBinarized(selectedEstimator, contAttrFrom, contAttrTo, discAttrFrom, discAttrTo, discAttrTo) ;
	  opt->binaryAttributes = TRUE ;
   }
   else {

	   switch (selectedEstimator)  // for discrete attributes
	   {
		   case estReliefFkEqual:
		   case estReliefFexpRank: 
		   case estReliefFdistance:
		   case estReliefFsqrDistance:
				   ReliefF(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo, selectedEstimator) ;
				   break ;

		   case estReliefFbestK: 
				   ReliefFbestK(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo,selectedEstimator) ;
				   break ;

		   case estRelief: 
				   Relief(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo) ;
				   break ;
       
		   case estInfGain: 
				   infGain(discAttrFrom,discAttrTo) ;
				   break ;

		   case  estGainRatio:
				   gainRatio(discAttrFrom,discAttrTo) ;
				   break ;

		   case  estMdl:
				   mdl(discAttrFrom,discAttrTo) ;
				   break ;
       
		   case  estGini:
				   Gini(discAttrFrom,discAttrTo) ;
				   break ;
  
		   case  estReliefFmyopic:
				   ReliefMyopic(discAttrFrom,discAttrTo) ;
				   break ;

		   case estAccuracy:
				   Accuracy(discAttrFrom,discAttrTo) ;
				   break ;

		  case estBinAccuracy:
				   BinAccuracy(discAttrFrom,discAttrTo) ;
				   break ;

		  case estReliefFmerit:  
				   ReliefFmerit(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo,2) ;
				   break ;
 
  
          case estDKM:
				   DKM(discAttrFrom,discAttrTo) ;
				   break ;

  		  case estReliefFexpC:
				   ReliefFexpC(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo, selectedEstimator) ;
                   break ;
          case estReliefFavgC:
				   ReliefFavgC(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo, selectedEstimator) ;
				   break ;

          case estReliefFpe:
				   ReliefFpe(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo, selectedEstimator) ;
				   break ;

	      case estReliefFpa: 
				   ReliefFpa(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo, selectedEstimator) ;
				   break ;
	      case estReliefFsmp: 
				   ReliefFsmp(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo, selectedEstimator) ;
				   break ;
		   case  estGainRatioC:
				   gainRatioC(discAttrFrom,discAttrTo) ;
				   break ;
          case estDKMc:
				   DKMc(discAttrFrom,discAttrTo) ;
				   break ;
	      case estReliefFcostKukar: 
				   ReliefFcostKukar(contAttrFrom,contAttrTo,discAttrFrom,discAttrTo) ;
				   break ;
          case estMDLsmp:
				   MDLsmp(discAttrFrom,discAttrTo) ;
				   break ;
		  default:  error("estimation::estimate", "selected estimator is out of range") ;

	   }
	   // some measures demand discretized attributes
	   if (selectedEstimator == estInfGain || selectedEstimator == estGainRatio ||
		   selectedEstimator == estMdl || selectedEstimator == estGini ||
		   selectedEstimator == estReliefFmyopic || selectedEstimator == estAccuracy ||
		   selectedEstimator == estBinAccuracy || selectedEstimator == estDKM ||
           selectedEstimator == estDKMc || selectedEstimator == estGainRatioC ||
           selectedEstimator == estMDLsmp)
	   {
		 int beforeEstimator = opt->selectionEstimator ;
		 opt->selectionEstimator = selectedEstimator ;
		 int idx ;
		 // binarize continuous attributes and the estimate of the best split 
    	 // is the estimate of the attribute
	     double result ;
		 if (opt->binarySplitNumericAttributes)
		 {
			construct contAttrib ;
			for (idx=contAttrFrom ; idx < contAttrTo ; idx++) 
			{
			  contAttrib.createSingle(idx, aCONTINUOUS) ;
			  contAttrib.splitValue = bestSplitGeneral(contAttrib, result, discAttrTo) ;
			  ContEstimation[idx] = result ;
			}
		 } 
		 else {
			marray<double> Bounds ;
			for (idx=contAttrFrom ; idx < contAttrTo ; idx++) 
			{
			   ContEstimation[idx] = discretizeGreedy(idx, Bounds, discAttrTo) ;
			}
		 }
		 opt->selectionEstimator = beforeEstimator ;
	   }
   }
   // find best attribute
   double bestContEst = - FLT_MAX, bestDiscEst = - FLT_MAX ;
   int i, bestContIdx = -1, bestDiscIdx = -1 ;
   for (i=contAttrFrom ; i < contAttrTo; i++)
   {
      if (ContEstimation[i] > bestContEst)
      {
          bestContEst =  ContEstimation[i] ;
          bestContIdx = i ;
      }
   }
   for (i=discAttrFrom ; i < discAttrTo; i++)
   {
      if (DiscEstimation[i] > bestDiscEst)
      {
         bestDiscEst =  DiscEstimation[i] ;
         bestDiscIdx = i ;
      }
   }
   if (bestContEst > bestDiscEst)
   {
      bestType = aCONTINUOUS ; // continuous
      return bestContIdx ;
   }
   else
   {
      bestType = aDISCRETE ; // discrete
      return bestDiscIdx ;
   }
}


// ***************************************************************************
//
//                     estimateConstruct
//     estimate selected constructs with choosen measure
//     and returns the index and type of the best estimated attribute
//     the description of constructs serves for possible complexity based measures
//
// ***************************************************************************
int estimation::estimateConstruct(int selectedEstimator, 
    int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, 
    attributeCount &bestType, marray<construct> &DiscConstruct, 
    marray<construct> &ContConstruct)
{
   return estimate(selectedEstimator, contAttrFrom,contAttrTo,
                       discAttrFrom, discAttrTo, bestType) ;
}




// ***************************************************************************
//
//                     adjustTables
//        prepare tables for increased number of attributes
//
// ***************************************************************************
void estimation::adjustTables(int newContSize, int newDiscSize)
{
   if (newContSize > currentContSize)
   {
      ContValues.addColumns(newContSize) ;
      ContEstimation.enlarge(newContSize) ;
      ContDistance.addColumns(newContSize) ;

      minValue.enlarge(newContSize) ;
      maxValue.enlarge(newContSize) ;
      valueInterval.enlarge(newContSize) ;
      step.enlarge(newContSize) ;
      NAcontValue.addColumns(newContSize) ;

#ifdef RAMP_FUNCTION
      DifferentDistance.enlarge(newContSize) ;
      EqualDistance.enlarge(newContSize) ;
      CAslope.enlarge(newContSize) ;
#endif

      currentContSize = newContSize ;
 
   }

   if (newDiscSize > currentDiscSize)
   {
      DiscValues.addColumns(newDiscSize) ;
      DiscEstimation.enlarge(newDiscSize) ;
      DiscDistance.addColumns(newDiscSize) ;
      discNoValues.enlarge(newDiscSize) ;

      NAdiscValue.addColumns(newDiscSize) ;

      currentDiscSize = newDiscSize ;
   }
}



// ***************************************************************************
//
//                      prepareContAttr  
//                      ----------------
// 
//        creating continuous data representation of feature
//
// ***************************************************************************
void estimation::prepareContAttr(int attrIdx) 
{

    // min, max, interval
    int j=0 ; 
    while (ContValues(j,attrIdx) == NAcont && j < TrainSize)
       j++ ;
    if (j >= TrainSize)
    {
      minValue[attrIdx] = maxValue[attrIdx] = NAcont ;
      // error("estimation::prepareContAttr", "all values of the attribute are missing") ;
    }
     else
        minValue[attrIdx] = maxValue[attrIdx] = ContValues(j, attrIdx) ;

    for (j=j+1 ; j < TrainSize ; j++)
       if (ContValues(j, attrIdx) != NAcont)
       {
         if (ContValues(j, attrIdx) < minValue[attrIdx])
            minValue[attrIdx] = ContValues(j, attrIdx) ;
         else
           if (ContValues(j, attrIdx) > maxValue[attrIdx])
             maxValue[attrIdx] = ContValues(j, attrIdx) ;
       }

    valueInterval[attrIdx] = maxValue[attrIdx] - minValue[attrIdx] ;
     
    if (valueInterval[attrIdx] < epsilon)
      valueInterval[attrIdx] = epsilon ;

   // step
   step[attrIdx] =  valueInterval[attrIdx]/noNAdiscretizationIntervals*double(1.000001) ; // 1.000001 - to avoid overflows due to numerical aproximation
   
   int k ;
   // missing values probabilities
   for (k=1 ; k <= NoClasses ; k++)
      NAcontValue(k,attrIdx).create(noNAdiscretizationIntervals+1, 0.0) ;
    
   for (j=0 ; j < TrainSize ; j++)
     if (ContValues(j,attrIdx) != NAcont)
       NAcontValue(DiscValues(j,0),attrIdx)[int((ContValues(j,attrIdx)-minValue[attrIdx])/step[attrIdx])+1] += 1 ;

   double denominator, valueProb ;
   for (k=1 ; k <= NoClasses ; k++)
   {
       denominator = noNAdiscretizationIntervals;
       for (j=1; j < NAcontValue(k, attrIdx).len() ; j++)
          denominator += NAcontValue(k, attrIdx)[j] ;
  
       NAcontValue(k, attrIdx)[0] = 0.0 ;
       for (j=1; j < NAcontValue(k, attrIdx).len() ; j++)
       {
          valueProb = (NAcontValue(k, attrIdx)[j] + double(1.0)) / denominator ;
          NAcontValue(k, attrIdx)[j] =  double(1.0) - valueProb ;
          // both are missing - compute same value probability
          NAcontValue(k, attrIdx)[0] += valueProb * valueProb  ;
       }
       NAcontValue(k, attrIdx)[0] = double(1.0) - NAcontValue(k, attrIdx)[0] ;
   }

#ifdef RAMP_FUNCTION
   // differemt, equal, slope
   DifferentDistance[attrIdx] = valueInterval[attrIdx] * opt->numAttrProportionEqual ;
   EqualDistance[attrIdx] = valueInterval[attrIdx] * opt->numAttrProportionDifferent  ;
   if (DifferentDistance[attrIdx] > EqualDistance[attrIdx])
      CAslope[attrIdx] = double(1.0)/(DifferentDistance[attrIdx] - EqualDistance[attrIdx]) ;
    else
      CAslope[attrIdx] = FLT_MAX ;
#endif

}



// ***************************************************************************
//
//                      prepareDiscAttr  
//                      ----------------
// 
//        creating discrete data representation of feature
//
// ***************************************************************************
void estimation::prepareDiscAttr(int attrIdx, int noValues) 
{
    
     discNoValues[attrIdx] = noValues ;

    // diff for missing values
    double denominator, valueProb ;    
    int j, k ;
    for (k=1 ; k <= NoClasses ; k++)
       NAdiscValue(k,attrIdx).create(discNoValues[attrIdx] +1, 0.0) ;

    for (j=0 ; j < TrainSize ; j++)
      NAdiscValue(DiscValues(j,0),attrIdx)[DiscValues(j,attrIdx)] += 1.0 ;

    for (k=1 ; k <= NoClasses ; k++)
    {
      denominator = discNoValues[attrIdx]  ;
      for (j=1; j < NAdiscValue(k,attrIdx).len() ; j++)
         denominator += NAdiscValue(k,attrIdx)[j] ;
     
      NAdiscValue(k,attrIdx)[0] = 0.0 ;
      for (j=1; j < NAdiscValue(k, attrIdx).len() ; j++)
      {
         valueProb = (NAdiscValue(k,attrIdx)[j]+double(1.0))/denominator ;
         NAdiscValue(k, attrIdx)[j] =  double(1.0) - valueProb ;
         // both are missing - compute same value probability
         NAdiscValue(k,attrIdx)[0] += valueProb * valueProb  ;
      }
      NAdiscValue(k, attrIdx)[0] = double(1.0) - NAdiscValue(k, attrIdx)[0] ;
    }
 }



// ***************************************************************************
//
//                      ReliefMyopic  
//                      ------------
// 
//        estimator myopic Relief (Gini corelated)
//
// ***************************************************************************
void estimation::ReliefMyopic(int discAttrFrom, int discAttrTo)
{

   // prepare estimations array
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // int NoDiscEstimated = discAttrTo - discAttrFrom ;
  

   marray<int> noExInClass(NoClasses+1, 0) ;

   // number of examples belonging to each of the classes
   int i;
   for (i=0 ; i < TrainSize ; i++)
      noExInClass[ DiscValues(i,0) ]++ ;
 
   // probabilities of the classes
   double pEqualC = 0.0 ;
   int classIdx ;
   for (classIdx=1 ; classIdx <= NoClasses ;classIdx++)
      pEqualC += sqr(double(noExInClass[classIdx]) / double(TrainSize)) ;

   if (pEqualC == 0.0 || pEqualC == 1.0)
   {
      DiscEstimation.init(discAttrFrom,discAttrTo,-1.0) ;
      return ;
   }

   double pEqualA, GiniR, condSum ;
   int valIdx, noOK ;
   int discIdx ;
   mmatrix<int> noClassAttrVal ;
   marray<int> valNo ;
   for (discIdx = discAttrFrom ; discIdx < discAttrTo ; discIdx++)
   {
	  noClassAttrVal.create(NoClasses+1, discNoValues[discIdx]+1, 0) ;
     valNo.create(discNoValues[discIdx]+1, 0) ;
      

	  // compute number of examples with each value of attribute and class
	  for (i=0 ; i < TrainSize ; i++)
        noClassAttrVal(DiscValues(i, 0), DiscValues(i, discIdx) ) ++ ;

	  // compute number of examples with each value of attribute 
	  for (valIdx = 0 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
        for (classIdx = 1 ; classIdx <= NoClasses ; classIdx ++)
       {
          valNo[valIdx] += noClassAttrVal(classIdx, valIdx) ;
       }
     }
     noOK = TrainSize - valNo[0] ;  // we do not take missing values into account
     if (noOK <= 0 )
     {
        DiscEstimation[discIdx] = -1.0 ;
        continue ;
     }

     // probability of equal attribute values
     pEqualA = 0.0 ;
	  for (valIdx = 1 ; valIdx <= discNoValues[discIdx]  ; valIdx++)
     {
        pEqualA +=   sqr(valNo[valIdx]/double(noOK)) ;
     }

     // computation of Gini'
     GiniR = 0.0 ;
     for (valIdx = 1 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
         
        condSum = 0.0 ;
        if (valNo[valIdx] > 0)
        {
          for (classIdx = 1 ; classIdx <= NoClasses ; classIdx++)
             condSum += sqr(double(noClassAttrVal(classIdx,valIdx))/double(valNo[valIdx])) ;
        }

        GiniR += sqr(valNo[valIdx]/double(noOK)) * condSum ;

     }
     GiniR = GiniR / pEqualA - pEqualC ;

     DiscEstimation[discIdx] = pEqualA/pEqualC/(1.0 - pEqualC) * GiniR ;

   }

}


// ***************************************************************************
//
//                      Gini  
//                      ----
// 
//        estimator Gini index
//
// ***************************************************************************
void estimation::Gini(int discAttrFrom, int discAttrTo)
{

   // prepare estimations arrays
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;
   // int NoDiscEstimated = discAttrTo - discAttrFrom ;
  

   marray<int> noExInClass(NoClasses+1, 0) ;

   // number of examples belonging to each of the classes
   int i;
   for (i=0 ; i < TrainSize ; i++)
      noExInClass[ DiscValues(i,0) ]++ ;
 
   // probabilities of the classes
   double pEqualC = 0.0 ;
   int classIdx ;
   for (classIdx=1 ; classIdx <= NoClasses ;classIdx++)
      pEqualC += sqr(double(noExInClass[classIdx]) / double(TrainSize)) ;

   if (pEqualC == 0.0 || pEqualC == 1.0)
   {
      DiscEstimation.init(discAttrFrom,discAttrTo,-1.0) ;
      return ;
   }

   double Gini, condSum ;
   int valIdx, noOK ;
   int discIdx ;
   mmatrix<int> noClassAttrVal ;
   marray<int> valNo ;
   for (discIdx = discAttrFrom ; discIdx < discAttrTo ; discIdx++)
   {
	  noClassAttrVal.create(NoClasses+1, discNoValues[discIdx]+1, 0) ;
      valNo.create(discNoValues[discIdx]+1, 0) ;
      
	  // compute number of examples with each value of attribute and class
	  for (i=0 ; i < TrainSize ; i++)
        noClassAttrVal(DiscValues(i, 0), DiscValues(i, discIdx) ) ++ ;

	  // compute number of examples with each value of attribute 
	  for (valIdx = 0 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
        for (classIdx = 1 ; classIdx <= NoClasses ; classIdx ++)
       {
          valNo[valIdx] += noClassAttrVal(classIdx, valIdx) ;
       }
     }
     noOK = TrainSize - valNo[0] ;  // we do not take missing values into account
     if (noOK <= 0 )
     {
        DiscEstimation[discIdx] = -1.0 ;
        continue ;
     }


     // computation of Gini
     Gini = 0.0 ;
     for (valIdx = 1 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
         
        condSum = 0.0 ;
        if (valNo[valIdx] > 0)
        {
          for (classIdx = 1 ; classIdx <= NoClasses ; classIdx++)
             condSum += sqr(double(noClassAttrVal(classIdx,valIdx))/double(valNo[valIdx])) ;
        }

        Gini += valNo[valIdx]/double(noOK) * condSum ;

     }
     Gini -= pEqualC ;

     DiscEstimation[discIdx] = Gini ;

   }

}



void estimation::mdl(int discAttrFrom, int discAttrTo)
{
   // prepare estimations arrays
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;
   // int NoDiscEstimated = discAttrTo - discAttrFrom ;
  

   marray<int> noExInClass(NoClasses+1, 0) ;

   // number of examples belonging to each of the classes
   int i, classIdx;
   for (i=0 ; i < TrainSize ; i++)
      noExInClass[ DiscValues(i,0) ]++ ;
 
   marray<double> Multinom(NoClasses) ;
   
   // encoding prior number of examples in each class
   for (classIdx=1 ; classIdx <= NoClasses ;classIdx++)
      Multinom[classIdx-1] = noExInClass[classIdx] ;
   Multinom.setFilled(NoClasses) ;
   double priorMDL = multinomLog2(Multinom) ;

   // encoding prior decoder 
   Multinom[0] = NoClasses  -1 ;
   Multinom[1] = TrainSize ;
   Multinom.setFilled(2) ;
   priorMDL += multinomLog2(Multinom) ;


   // compute postMDL
   int valIdx, noOK ;
   int discIdx ;
   mmatrix<int> noClassAttrVal ;
   marray<int> valNo ;
   double postClass, postDecoder ;
   for (discIdx = discAttrFrom ; discIdx < discAttrTo ; discIdx++)
   {
      noClassAttrVal.create(NoClasses+1, discNoValues[discIdx]+1, 0) ;
      valNo.create(discNoValues[discIdx]+1, 0) ;
      
	   // compute number of examples with each value of attribute and class
	   for (i=0 ; i < TrainSize ; i++)
         noClassAttrVal(DiscValues(i, 0), DiscValues(i, discIdx) ) ++ ;

	  // compute number of examples with each value of attribute 
	  for (valIdx = 0 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
        for (classIdx = 1 ; classIdx <= NoClasses ; classIdx ++)
       {
          valNo[valIdx] += noClassAttrVal(classIdx, valIdx) ;
       }
     }
     noOK = TrainSize - valNo[0] ;  // we do not take missing values into account
     if (noOK <= 0 )
     {
        DiscEstimation[discIdx] = -1.0 ;
        continue ;
     }
     // computation of postMDL
     postClass = postDecoder = 0.0 ;
     for (valIdx = 1 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
         
        if (valNo[valIdx] > 0)
        {
          for (classIdx = 1 ; classIdx <= NoClasses ; classIdx++)
             Multinom[classIdx-1] = noClassAttrVal(classIdx,valIdx) ;
          Multinom.setFilled(NoClasses) ;
          postClass += multinomLog2(Multinom) ;

          Multinom[0] = NoClasses - 1 ;
          Multinom[1] = valNo[valIdx] ;
          Multinom.setFilled(2) ;
          postDecoder += multinomLog2(Multinom) ;
        }
     }

     DiscEstimation[discIdx] = (priorMDL - postClass - postDecoder) / double(TrainSize) ;

   }
}

void estimation::gainRatio(int discAttrFrom, int discAttrTo)
{
     // prepare estimations arrays
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;
  
   marray<int> noExInClass(NoClasses+1, 0) ;

   // number of examples belonging to each of the classes
   int i;
   for (i=0 ; i < TrainSize ; i++)
      noExInClass[ DiscValues(i,0) ]++ ;
 
   // probabilities of the classes
   double Hc = 0.0 ;
   double tempP ;
   int classIdx ;
   for (classIdx=1 ; classIdx <= NoClasses ;classIdx++)
   {
      if (noExInClass[classIdx] > 0)
      {
         tempP = double(noExInClass[classIdx]) / double(TrainSize) ;                 
         Hc -= tempP * log2(tempP) ;
      }
   }

   double Hca, Ha ;
   int valIdx, noOK ;
   int discIdx ;
   mmatrix<int> noClassAttrVal ;
   marray<int> valNo ;
   for (discIdx = discAttrFrom ; discIdx < discAttrTo ; discIdx++)
   {
	   noClassAttrVal.create(NoClasses+1, discNoValues[discIdx]+1, 0) ;
      valNo.create(discNoValues[discIdx]+1, 0) ;
      
	  // compute number of examples with each value of attribute and class
	  for (i=0 ; i < TrainSize ; i++)
        noClassAttrVal(DiscValues(i, 0), DiscValues(i, discIdx) ) ++ ;

	  // compute number of examples with each value of attribute 
	  for (valIdx = 0 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
        for (classIdx = 1 ; classIdx <= NoClasses ; classIdx ++)
       {
          valNo[valIdx] += noClassAttrVal(classIdx, valIdx) ;
       }
     }
     noOK = TrainSize - valNo[0] ;  // we do not take missing values into account
     if (noOK <= 0 )
     {
        DiscEstimation[discIdx] = -1.0 ;
        continue ;
     }
      
     // computation of Informaion gain
     Hca = Ha = 0.0 ;
     for (valIdx = 1 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
         
        if (valNo[valIdx] > 0)
        {
          for (classIdx = 1 ; classIdx <= NoClasses ; classIdx++)
             if (noClassAttrVal(classIdx,valIdx) > 0)
             {
                tempP = double(noClassAttrVal(classIdx,valIdx))/double(noOK) ; 
                Hca -= tempP * log2(tempP) ;
             }
        
          if (valNo[valIdx] != noOK)
          {
             tempP = double(valNo[valIdx]) / double(noOK) ;
             Ha -= tempP * log2(tempP) ;
          }
        }
     }
     if (Ha > 0.0)
       DiscEstimation[discIdx] = (Hc + Ha - Hca) / Ha ;
     else
       DiscEstimation[discIdx] = -1.0 ;
   }

}

void estimation::infGain(int discAttrFrom, int discAttrTo)
{
   // prepare estimations arrays
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;
   // int NoDiscEstimated = discAttrTo - discAttrFrom ;
  

   marray<int> noExInClass(NoClasses+1, 0) ;

   // number of examples belonging to each of the classes
   int i;
   for (i=0 ; i < TrainSize ; i++)
      noExInClass[ DiscValues(i,0) ]++ ;
 
   // probabilities of the classes
   double Hc = 0.0 ;
   double tempP ;
   int classIdx ;
   for (classIdx=1 ; classIdx <= NoClasses ;classIdx++)
   {
      if (noExInClass[classIdx] > 0)
      {
         tempP = double(noExInClass[classIdx]) / double(TrainSize) ;                 
         Hc -= tempP * log2(tempP) ;
      }
   }

   double Hca, Ha ;
   int valIdx, noOK ;
   int discIdx ;
   mmatrix<int> noClassAttrVal ;
   marray<int> valNo ;
   for (discIdx = discAttrFrom ; discIdx < discAttrTo ; discIdx++)
   {
	   noClassAttrVal.create(NoClasses+1, discNoValues[discIdx]+1, 0) ;
      valNo.create(discNoValues[discIdx]+1, 0) ;
      
	  // compute number of examples with each value of attribute and class
	  for (i=0 ; i < TrainSize ; i++)
        noClassAttrVal(DiscValues(i, 0), DiscValues(i, discIdx) ) ++ ;

	  // compute number of examples with each value of attribute 
	  for (valIdx = 0 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
        for (classIdx = 1 ; classIdx <= NoClasses ; classIdx ++)
       {
          valNo[valIdx] += noClassAttrVal(classIdx, valIdx) ;
       }
     }
     noOK = TrainSize - valNo[0] ;  // we do not take missing values into account
     if (noOK <= 0 )
     {
        DiscEstimation[discIdx] = -1.0 ;
        continue ;
     }
      
     // computation of Informaion gain
     Hca = Ha = 0.0 ;
     for (valIdx = 1 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
         
        if (valNo[valIdx] > 0)
        {
          for (classIdx = 1 ; classIdx <= NoClasses ; classIdx++)
             if (noClassAttrVal(classIdx,valIdx) > 0)
             {
                tempP = double(noClassAttrVal(classIdx,valIdx))/double(noOK) ; 
                Hca -= tempP * log2(tempP) ;
             }
        
          tempP = double(valNo[valIdx]) / double(noOK) ;
          Ha -= tempP * log2(tempP) ;
        }
     }
     DiscEstimation[discIdx] = Hc + Ha - Hca ;

   }

}


    
// ***************************************************************************
//
//                      Accuracy 
//                      --------
// 
//        estimator Accuracy
//
// ***************************************************************************
void estimation::Accuracy(int discAttrFrom, int discAttrTo) 
{

   // prepare estimations arrays
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;
   // int NoDiscEstimated = discAttrTo - discAttrFrom ;
  
   double Accuracy ;
   int discIdx, classIdx, maxClassIdx, valIdx, i, noOK ;
   mmatrix<int> noClassAttrVal ;
   marray<int> valNo ;
   for (discIdx = discAttrFrom ; discIdx < discAttrTo ; discIdx++)
   {
      noClassAttrVal.create(NoClasses+1, discNoValues[discIdx]+1, 0) ;
      valNo.create(discNoValues[discIdx]+1, 0) ;
      
	  // compute number of examples with each value of attribute and class
	  for (i=0 ; i < TrainSize ; i++)
        noClassAttrVal(DiscValues(i, 0), DiscValues(i, discIdx) ) ++ ;

	  // compute number of examples with each value of attribute 
	  for (valIdx = 0 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
        for (classIdx = 1 ; classIdx <= NoClasses ; classIdx ++)
       {
          valNo[valIdx] += noClassAttrVal(classIdx, valIdx) ;
       }
     }
     noOK = TrainSize - valNo[0] ;  // we do not take missing values into account
     if (noOK <= 0 )
     {
        DiscEstimation[discIdx] = -1.0 ;
        continue ;
     }


     // computation of Accuracy
     Accuracy = 0.0 ;
     for (valIdx = 1 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {
        
        if (valNo[valIdx] > 0)
        {
          maxClassIdx = 1 ; 
          for (classIdx = 2 ; classIdx <= NoClasses ; classIdx++)
             if (noClassAttrVal(classIdx,valIdx) > noClassAttrVal(maxClassIdx,valIdx) )
                maxClassIdx = classIdx ;

          Accuracy += double(noClassAttrVal(maxClassIdx,valIdx))/double(noOK) ;
        }

     }
     
     DiscEstimation[discIdx] = Accuracy ;

   }

}


// ***************************************************************************
//
//                      BinAccuracy 
//                      -----------
// 
//        estimator binarized accuracy (accuracy on a binary split)
//
// ***************************************************************************
void estimation::BinAccuracy(int discAttrFrom, int discAttrTo) 
{

   // prepare estimations arrays
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;
   // int NoDiscEstimated = discAttrTo - discAttrFrom ;
  
   int discIdx, i, j, maxSample ;
   mmatrix<int> noClassAttrVal(NoClasses+1, 3) ;
       
   if (opt->discretizationSample==0)
     maxSample = TrainSize -1;
   else
      maxSample = opt->discretizationSample ;
  

   double est, maxEst, maxRound ;
   int greedyPositions, exhaustivePositions, idxRound, filled ;
   int attrValue ;
   marray<int> dataColumn(TrainSize) ;
   for (discIdx = discAttrFrom ; discIdx < discAttrTo ; discIdx++)
   {
      if (discNoValues[discIdx] > 2)  // demands binarization
      {
         // binarize attributes and the estimate of the best binarization is the estimate of the attribute
         binPartition Generator( discNoValues[discIdx]) ;
         greedyPositions = discNoValues[discIdx] * (discNoValues[discIdx]+1)/2 ;
         if ( discNoValues[discIdx] < maxVal4ExhDisc)
			 exhaustivePositions = Generator.noPositions() ;
		 else 
			 exhaustivePositions = -1 ; // invalid value
         if ( (discNoValues[discIdx] < maxVal4ExhDisc) && (exhaustivePositions * 0.8 <= greedyPositions || exhaustivePositions < maxSample))
         {
            // exhaustive search
            maxEst = -1.0 ;
            while (Generator.increment() )
            {
               // compute data column
               for (i=0 ; i < TrainSize ; i++)
               {
                  attrValue = DiscValues(i, discIdx) ;
                  if (attrValue == NAdisc)
                     dataColumn[i] = NAdisc ;
                  else
                     if (Generator.leftPartition[attrValue])
                        dataColumn[i] = 1 ;
                      else
                        dataColumn[i] = 2 ;
               }
               noClassAttrVal.init(0) ;

          	  // compute number of examples with each value of attribute and class
      	     for (i=0 ; i < TrainSize ; i++)
                noClassAttrVal(DiscValues(i, 0), dataColumn[i] ) ++ ;

              est = binAccEst(noClassAttrVal, 2) ;
              if (est > maxEst)
                 maxEst = est ;
            }
         }
         else { // greedy search
      
            marray<boolean> currentBest(discNoValues[discIdx]+1, FALSE) ;
            maxEst = -1.0 ;
            for (filled=1 ; filled < discNoValues[discIdx] ; filled++)
            {
               maxRound = -1.0 ;
               idxRound = -1 ;
               for (j=1 ; j <= discNoValues[discIdx]; j++)
               if (currentBest[j] == FALSE)
               {
                  currentBest[j] = TRUE ;
    
                  // compute data column
                  for (i=0 ; i < TrainSize ; i++)
                  {
                     attrValue = DiscValues(i, discIdx) ;
                    if (attrValue == NAdisc)
                       dataColumn[i] = NAdisc ;
                    else
                     if (currentBest[attrValue])
                        dataColumn[i] = 1 ;
                      else
                        dataColumn[i] = 2 ;
                  }
                  noClassAttrVal.init(0) ;

                  // compute number of examples with each value of attribute and class
      	         for (i=0 ; i < TrainSize ; i++)
                    noClassAttrVal(DiscValues(i, 0), dataColumn[i] ) ++ ;

                  est = binAccEst(noClassAttrVal, 2) ;
                  if (est > maxRound)
                  {
                     maxRound = est ;
                     idxRound = j ;
                  }
                  currentBest[j] = FALSE ;
               }
               if (maxRound > maxEst)
                    maxEst = maxRound ;
               currentBest[idxRound] = TRUE ;
            }
         }
         DiscEstimation[discIdx] = maxEst ; 

      }
      else
      {
        noClassAttrVal.init(0) ;
        
   	  // compute number of examples with each value of attribute and class
	     for (i=0 ; i < TrainSize ; i++)
          noClassAttrVal(DiscValues(i, 0), DiscValues(i, discIdx) ) ++ ;
	     
        DiscEstimation[discIdx] = binAccEst(noClassAttrVal, 2) ;
      }
   }

}


double estimation::binAccEst(mmatrix<int> &noClassAttrVal, int noValues) 
{
   marray<int> valNo(noValues+1, 0) ;
   int valIdx, classIdx ;
   // compute number of examples with each value of attribute 
   for (valIdx = 0 ; valIdx <= noValues ; valIdx++)
       for (classIdx = 1 ; classIdx <= NoClasses ; classIdx ++)
          valNo[valIdx] += noClassAttrVal(classIdx, valIdx) ;

   int noOK = TrainSize - valNo[0] ;  // we do not take missing values into account
   if (noOK <= 0 )
      return -1.0 ;
   
   // computation of accuracy
    double AccuracyEst = 0.0 ;
    int maxClassIdx ;
    for (valIdx = 1 ; valIdx <= noValues ; valIdx++)
    {
       if (valNo[valIdx] > 0)
       {
         maxClassIdx = 1 ; 
         for (classIdx = 2 ; classIdx <= NoClasses ; classIdx++)
            if (noClassAttrVal(classIdx,valIdx) > noClassAttrVal(maxClassIdx,valIdx) )
               maxClassIdx = classIdx ;

         AccuracyEst += double(noClassAttrVal(maxClassIdx,valIdx))/double(noOK) ;
       }
    }
    return AccuracyEst ;
}

// ***************************************************************************
//
//                      DKM 
//                      ----
// 
//        estimator Dietterich, Kearns, Mansour (DKM): theoretically justified
//        in authors ICML'96 paper; basically it is  impurity function
//        G(q)=2*sqrt(q*(1-q)) where q represents probabillity of the majority class
//
// ***************************************************************************
void estimation::DKM(int discAttrFrom, int discAttrTo)
{

   // prepare estimations arrays
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;
   // int NoDiscEstimated = discAttrTo - discAttrFrom ;
  

   marray<int> noExInClass(NoClasses+1, 0) ;

   // number of examples belonging to each of the classes
   int i;
   for (i=0 ; i < TrainSize ; i++)
      noExInClass[ DiscValues(i,0) ]++ ;
 
   // probability of the majority class
   double q, DKMprior ;
   int maxEx = noExInClass[1] ;
   int classIdx ;
   for (classIdx=2 ; classIdx <= NoClasses ; classIdx++)
      if (noExInClass[classIdx] > maxEx)
          maxEx = noExInClass[classIdx] ; 
   q = double(maxEx)/TrainSize ;

   if (q == 0.0 || q == 1.0)
   {
      DiscEstimation.init(discAttrFrom,discAttrTo,-1.0) ;
      return ;
   }
   else DKMprior = 2.0 * sqrt(q*(1.0-q)) ;

   double DKMpost, qCond ;
   int valIdx, noOK ;
   int discIdx ;
   mmatrix<int> noClassAttrVal ;
   marray<int> valNo ;
   for (discIdx = discAttrFrom ; discIdx < discAttrTo ; discIdx++)
   {
	  noClassAttrVal.create(NoClasses+1, discNoValues[discIdx]+1, 0) ;
      valNo.create(discNoValues[discIdx]+1, 0) ;
      
	  // compute number of examples with each value of attribute and class
	  for (i=0 ; i < TrainSize ; i++)
        noClassAttrVal(DiscValues(i, 0), DiscValues(i, discIdx) ) ++ ;

	  // compute number of examples with each value of attribute 
	  for (valIdx = 0 ; valIdx <= discNoValues[discIdx] ; valIdx++)
        for (classIdx = 1 ; classIdx <= NoClasses ; classIdx ++)
          valNo[valIdx] += noClassAttrVal(classIdx, valIdx) ;

      noOK = TrainSize - valNo[0] ;  // we do not take missing values into account
     if (noOK <= 0 )
     {
        DiscEstimation[discIdx] = -1.0 ;
        continue ;
     }

     // computation of DKM   
     DKMpost = 0.0 ;
     for (valIdx = 1 ; valIdx <= discNoValues[discIdx] ; valIdx++)  
        if (valNo[valIdx] > 0) {
          // find majority value
          maxEx = noClassAttrVal(1,valIdx) ;
          for (classIdx = 2 ; classIdx <= NoClasses ; classIdx++)
             if (noClassAttrVal(classIdx,valIdx)>maxEx) 
                 maxEx = noClassAttrVal(classIdx,valIdx) ;
           qCond = double(maxEx)/ valNo[valIdx] ;
           if (qCond > 0.0 && qCond < 1.0)
             DKMpost += double(valNo[valIdx])/noOK * sqrt(qCond*(1.0-qCond)) ;
        }
     DiscEstimation[discIdx] = DKMprior-2*DKMpost ;

   }

}



// ***************************************************************************
//
//                       CVVilalta
//                       ---------
//
//         computes measure of problem difficulty called concept variation
//              (Vilalta, R., 1999)
//                   
//
// ***************************************************************************
double estimation::CVVilalta(int contAttrFrom, int contAttrTo,
                                    int discAttrFrom, int discAttrTo)
{

   const double alpha = 2.0 ;


   double NoUsed = contAttrTo - contAttrFrom + discAttrTo - discAttrFrom;
 
   
   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   double weightSum, weight, sigma, ConVar = 0.0 , distance, denominator ;
   int current, m ;
   

   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)
   {
   
       if (NoIterations == TrainSize)
          current = iterIdx ;
       else
           current =  randBetween(0, TrainSize) ;
 
       // first we compute distances of all other examples to current
      computeDistances(current) ;

     weightSum = 0.0 ;
     sigma = 0.0 ;
     for (m=0 ; m < NoIterations ; m++)
     {
        if (m==current)
           continue ;

        distance = CaseDistance(m) ;
        denominator = NoUsed - distance ; 
        if (denominator > epsilon)
          weight = 1.0 / pow(2.0, alpha * distance/denominator) ;
        else 
          weight = 0.0 ;

        weightSum += weight ;

        sigma += weight * DiscDistance(m, 0) ;
     }

     ConVar += sigma/weightSum ;

  }
   
  return ConVar/double(NoIterations) ;

}



// ***************************************************************************
//
//                       CVmodified
//                       ---------
//
//         computes measure of problem difficulty called concept variation
//          based on our modification
//                   
//
// ***************************************************************************
double estimation::CVmodified(int contAttrFrom, int contAttrTo,
                                    int discAttrFrom, int discAttrTo)
{

  
   int NoUsed = contAttrTo - contAttrFrom + discAttrTo - discAttrFrom;
 
   
   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   double ConVar = 0.0, incConVar ;
   int current, i, iDisc, iCont, k ;
   sortRec tempSort ;
   marray<sortRec> distSort(TrainSize) ;
   

   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)
   {
   
       if (NoIterations == TrainSize)
          current = iterIdx ;
       else
           current =  randBetween(0, TrainSize) ;
 
       // first we compute distances of all other examples to current
      computeDistances(current) ;

      //  sort all the examples with descending distance
      distSort.clear() ;
      for (i=0 ; i < TrainSize; i++)
      {
        if (i==current)  // we skip current example
          continue ;
        tempSort.key =  CaseDistance(i) ;
        tempSort.value = i ;
        distSort.addEnd(tempSort) ;
      }

      distSort.sort(ascSortComp) ;
    
      for (iDisc=discAttrFrom ; iDisc < discAttrTo ; iDisc++)
      {
         incConVar = 0.0 ; 
         k = 0 ;
         for (i=0 ; i < distSort.filled() ; i++)
            if (DiscDistance(distSort[i].value,iDisc) > 0)
            {
                incConVar += DiscDistance(distSort[i].value, 0) ;
                k++ ;
                if (k >= kNearestEqual)
                  break ;
            }
         if (k > 0)
             ConVar += incConVar / double(k) ;
      }

      for (iCont=contAttrFrom ; iCont < contAttrTo ; iCont++)
      {
         incConVar = 0.0 ; 
         k = 0 ;
         for (i=0 ; i < distSort.filled() ; i++)
            if (ContDistance(distSort[i].value, iCont) > 0)
            {
                incConVar += DiscDistance(distSort[i].value, 0) ;
                k++ ;
                if (k >= kNearestEqual)
                   break ;
            }
         if (k>0)
             ConVar += incConVar / double(k) ;
       }
   }  

  return ConVar/double(NoIterations)/double(NoUsed) ;
}



// ***************************************************************************
//
//                     estimateSelected
//     estimate selected attributes with choosen measure
//     and returns the index and type of the best estimated attribute
//        intended only for fast estimation with RF and  with non-Releif measures
//
// ***************************************************************************
int estimation::estimateSelected(int selectedEstimator, marray<boolean> &mask, attributeCount &bestType) {
   attributeCount bT ; // dummy
   double bestEst = - FLT_MAX;
   int bestIdx = -1, iA ;
   	for (iA=1; iA < mask.filled(); iA++) 
	   if (mask[iA]) { // evaluate that attribute
		   if (gFT->AttrDesc[iA].continuous) { 
			   estimate(selectedEstimator, gFT->AttrDesc[iA].tablePlace, gFT->AttrDesc[iA].tablePlace +1, 0, 0, bT) ;
			   if (ContEstimation[gFT->AttrDesc[iA].tablePlace] > bestEst){
				   bestEst = ContEstimation[gFT->AttrDesc[iA].tablePlace] ;
			       bestType = aCONTINUOUS ;
				   bestIdx = gFT->AttrDesc[iA].tablePlace ;
			   }
		   }
		   else {
			   estimate(selectedEstimator, 0, 0, gFT->AttrDesc[iA].tablePlace, gFT->AttrDesc[iA].tablePlace +1,  bT) ;
			   if (DiscEstimation[gFT->AttrDesc[iA].tablePlace] > bestEst){
				   bestEst = DiscEstimation[gFT->AttrDesc[iA].tablePlace] ;
			       bestType = aDISCRETE ;
				   bestIdx = gFT->AttrDesc[iA].tablePlace ;
			   }
		   }
	   }
   return bestIdx ;
}
