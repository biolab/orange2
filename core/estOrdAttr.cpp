
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "general.h"
#include "contain.h"
#include "estimator.h"                
#include "utils.h"
#include "options.h"

extern Options *opt ;

// ***************************************************************************
//
//                       avReliefF
//                       -------
//
//   contains the version of ReliefF for attribute values
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//
// ***************************************************************************
// ***************************************************************************
void estimation::aVReliefF(int discAttrFrom, int discAttrTo, marray<marray<double> > &result,
						   int distanceType) {

   int iA, iV ;
   // empty the results arrays
   for (iA=discAttrFrom ; iA < discAttrTo ; iA++) 
	   result[iA].init(0.0) ;
	   
   // number of examples belonging to each of the classes
   marray<int> noExInClass(NoClasses+1) ;
   marray<double> probClass(NoClasses+1) ;
   noExInClass.init(0) ;
   probClass.init(0.0) ;
   int i, idx, aVal ;
   for (i=0 ; i < TrainSize ; i++)
   {
      noExInClass[ DiscValues(i,0) ]++ ;
      probClass[ DiscValues(i,0) ] += weight[i] ;
   }

   // obtain the greatest sensible k (nubmer of nearest hits/misses)
   // and the total weight of examples
   int maxK = noExInClass[1] ;
   double wAll = probClass[1] ;
   for (idx=2 ; idx <= NoClasses ; idx++)
   {
      if (noExInClass[idx] > maxK)
         maxK = noExInClass[idx] ;
      wAll += probClass[idx] ;
   }

   // compute estimations of class value probabilities with their relative frequencies
   for (idx=1 ; idx <= NoClasses ; idx++)
      probClass[idx] = probClass[idx] / wAll ;

   // data structure to hold nearest hits/misses
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)  {
      distanceArray[iClss].create(noExInClass[iClss]) ;
      diffSorted[iClss].create(noExInClass[iClss]) ;
   }

   // normalization of contribution of misses
   mmatrix<double> clNorm(NoClasses+1,NoClasses+1) ;
   for (int j=1 ; j<=NoClasses ; j++)
     for (i=1 ; i<=NoClasses ; i++)
        clNorm.Set(j,i, probClass[j]/(1.0-probClass[i]) ) ;

   // we have to compute distances up to the folowing attributes
   discUpper = NoDiscrete ;
   contUpper = NoContinuous ;

   double distanceSum, normDistance, Adiff ;
   int current, neighbourIdx, cl, iAttr, currentClass ;
   
   marray<marray<double> > incDiscDiffA(NoDiscrete+1) ;
   for (iA=1 ; iA < NoDiscrete ; iA++) 
	   incDiscDiffA[iA].create(discNoValues[iA]+1, 0.0) ;
       
   // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
  
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)  {

	   current = sampleIdx[iterIdx] ;
 
       // initialize (optimization reasons)
      currentClass =  DiscValues(current, 0) ;
      
        
      // first we compute distances of  all other examples to current
      computeDistances(current) ;

      // compute distance factors
      prepareDistanceFactors(current, distanceType) ;

      for (cl=1 ; cl<=NoClasses ; cl++) {
         // compute sum of diffs
        for (iA=discAttrFrom ; iA < discAttrTo ; ++iA) 
           incDiscDiffA[iA].init(0.0) ;
         distanceSum = 0.0 ;
         for (i=0 ; i < distanceArray[cl].filled() ; i++) {
            neighbourIdx = distanceArray[cl][i].value ;
            normDistance = distanceArray[cl][i].key ;
            distanceSum += normDistance ;
                 
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++) {
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
			   aVal = DiscValues(neighbourIdx, iAttr) ;
			   if (aVal != NAdisc) {
			      incDiscDiffA[iAttr][aVal] +=  Adiff * normDistance  ;
			      incDiscDiffA[iAttr][0] +=  Adiff * normDistance  ;
			   }
            }
         }
		 if (cl == currentClass) { // hit or miss
            // hit
            // normalization of increments
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; ++iAttr)
	           for (iV=0 ; iV <= discNoValues[iAttr] ; iV++)
                  if (incDiscDiffA[iAttr][iV] > epsilon)
                    result[iAttr][iV] -= incDiscDiffA[iAttr][iV]/distanceSum ;
          }
          else {
             // miss
             // normalization of increments
             for (iAttr=discAttrFrom ; iAttr < discAttrTo ; ++iAttr)
	           for (iV=0 ; iV <= discNoValues[iAttr] ; iV++)
                  if (incDiscDiffA[iAttr][iV] > epsilon)
                      result[iAttr][iV] += clNorm(cl, currentClass) * incDiscDiffA[iAttr][iV]/distanceSum ;
          }
      }
   }  
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
	   for (iV=0 ; iV <= discNoValues[iAttr] ; iV++) {
          result[iAttr][iV] /= double(NoIterations) ;
          #ifdef DEBUG
          if (result[iAttr][iV] > 1.00001 || result[iAttr][iV] < -1.00001)
             error("estimation::avReliefF", "computed discrete weights are out of scope") ;
          #endif
       }
 
}

   
// ***************************************************************************
//
//                       aVordReliefF
//                       -------
//
//                   
//
// ***************************************************************************
// ***************************************************************************
void estimation::ordAvReliefF(int discAttrFrom, int discAttrTo, 
	        marray<marray<double> > &resultCpAp, marray<marray<double> > &resultCpAn,
			marray<marray<double> > &resultCpAe, 
			marray<marray<double> > &resultCnAp, marray<marray<double> > &resultCnAn,
			marray<marray<double> > &resultCnAe, 
			marray<marray<double> > &resultCeAp, marray<marray<double> > &resultCeAn,
			marray<marray<double> > &resultCeAe, 
			int distanceType) {

   int iA, iV ;
   // empty the results arrays
   for (iA=discAttrFrom ; iA < discAttrTo ; iA++) {
	   resultCpAe[iA].init(0.0) ;
	   resultCpAp[iA].init(0.0) ;
	   resultCpAn[iA].init(0.0) ;
 	   resultCnAe[iA].init(0.0) ;
	   resultCnAp[iA].init(0.0) ;
	   resultCnAe[iA].init(0.0) ;
 	   resultCeAe[iA].init(0.0) ;
	   resultCeAp[iA].init(0.0) ;
	   resultCeAe[iA].init(0.0) ;
  }
	   
   // number of examples belonging to each of the classes
   marray<int> noExInClass(NoClasses+1) ;
   marray<double> probClass(NoClasses+1) ;
   noExInClass.init(0) ;
   probClass.init(0.0) ;
   int i, idx, aVal ;
   for (i=0 ; i < TrainSize ; i++) {
      noExInClass[ DiscValues(i,0) ]++ ;
      probClass[ DiscValues(i,0) ] += weight[i] ;
   }

   // obtain the greatest sensible k (nubmer of nearest hits/misses)
   // and the total weight of examples
   int maxK = noExInClass[1] ;
   double wAll = probClass[1] ;
   for (idx=2 ; idx <= NoClasses ; idx++)
   {
      if (noExInClass[idx] > maxK)
         maxK = noExInClass[idx] ;
      wAll += probClass[idx] ;
   }

   // compute estimations of class value probabilities with their relative frequencies
   for (idx=1 ; idx <= NoClasses ; idx++)
      probClass[idx] = probClass[idx] / wAll ;

   // data structure to hold nearest hits/misses
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)  {
      distanceArray[iClss].create(noExInClass[iClss]) ;
      diffSorted[iClss].create(noExInClass[iClss]) ;
   }

   // normalization of contribution of misses
   mmatrix<double> clNorm(NoClasses+1,NoClasses+1) ;
   for (int j=1 ; j<=NoClasses ; j++)
     for (i=1 ; i<=NoClasses ; i++)
		if (i==j)
          clNorm.Set(j,i, 1.0) ; //hit
		else
          clNorm.Set(j,i, probClass[j]/(1.0-probClass[i]) ) ;

   // we have to compute distances up to the folowing attributes
   discUpper = NoDiscrete ;
   contUpper = NoContinuous ;

   double distanceSum, normDistance, Adiff, clDiff ;
   int current, neighbourIdx, cl, iAttr, currentClass ;
												   
   marray<marray<double> > incCpAe(NoDiscrete+1), incCpAn(NoDiscrete+1), incCpAp(NoDiscrete+1),
	                       incCnAe(NoDiscrete+1), incCnAn(NoDiscrete+1), incCnAp(NoDiscrete+1),
	                       incCeAe(NoDiscrete+1), incCeAn(NoDiscrete+1), incCeAp(NoDiscrete+1)  ;

   for (iA=0 ; iA < NoDiscrete ; iA++) {
	   incCpAp[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCpAn[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCpAe[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCnAp[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCnAn[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCnAe[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCeAp[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCeAn[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCeAe[iA].create(discNoValues[iA]+1, 0.0) ;
   }
       
   // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
  
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)  {

	   current = sampleIdx[iterIdx] ;
       currentClass =  DiscValues(current, 0) ;
      // first we compute distances of  all other examples to current
      computeDistancesOrd(current) ;

      // compute distance factors
      prepareDistanceFactors(current, distanceType) ;

      for (cl=1 ; cl<=NoClasses ; cl++) {
         // compute sum of diffs
		 for (iA=discAttrFrom ; iA < discAttrTo ; ++iA) {
             incCpAp[iA].init(0.0) ;
             incCpAn[iA].init(0.0) ;
			 incCpAe[iA].init(0.0) ;
			 incCnAp[iA].init(0.0) ;
             incCnAn[iA].init(0.0) ;
			 incCnAe[iA].init(0.0) ;
			 incCeAp[iA].init(0.0) ;
             incCeAn[iA].init(0.0) ;
			 incCeAe[iA].init(0.0) ;
		 }
         distanceSum = 0.0 ;
         clDiff = DAdiffSign(0, current, distanceArray[cl][0].value) ;     

         for (i=0 ; i < distanceArray[cl].filled() ; i++) {
            neighbourIdx = distanceArray[cl][i].value ;
            normDistance = distanceArray[cl][i].key ;
            distanceSum += normDistance ;

            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++) {
               Adiff = DAdiffSign(iAttr, current, neighbourIdx) ;
			   //aVal = DiscValues(neighbourIdx, iAttr) ;	  
			   aVal = DiscValues(current, iAttr) ;
			   if (aVal != NAdisc) {
				   if (clDiff==0) { //hit				   
					   if (Adiff==0) {
					      incCeAe[iAttr][aVal] +=  normDistance  ;
			              incCeAe[iAttr][0] +=  normDistance  ;
				       }
				       else if (Adiff > 0.0) {	 
					      incCeAp[iAttr][aVal] +=  Adiff * normDistance  ;
			              incCeAp[iAttr][0] +=  Adiff * normDistance  ;
				       }
				       else {
					      incCeAn[iAttr][aVal] -=  Adiff * normDistance  ;
			              incCeAn[iAttr][0] -=  Adiff * normDistance  ;
				       }
				   }
				   else if (clDiff>0) {
					   if (Adiff==0) {
					      incCpAe[iAttr][aVal] +=  normDistance  ;
			              incCpAe[iAttr][0] +=  normDistance  ;
				       }
				       else if (Adiff > 0.0) {	 
					      incCpAp[iAttr][aVal] +=  Adiff * normDistance  ;
			              incCpAp[iAttr][0] +=  Adiff * normDistance  ;
				       }
				       else {
					      incCpAn[iAttr][aVal] -=  Adiff * normDistance  ;
			              incCpAn[iAttr][0] -=  Adiff * normDistance  ;
				       }
				   }
				   else {   // clDiff < 0
					   if (Adiff==0) {
					      incCnAe[iAttr][aVal] +=  normDistance  ;
			              incCnAe[iAttr][0] +=  normDistance  ;
				       }
				       else if (Adiff > 0.0) {	 
					      incCnAp[iAttr][aVal] +=  Adiff * normDistance  ;
			              incCnAp[iAttr][0] +=  Adiff * normDistance  ;
				       }
				       else {
					      incCnAn[iAttr][aVal] -=  Adiff * normDistance  ;
			              incCnAn[iAttr][0] -=  Adiff * normDistance  ;
				       }
				   
				   }
               }
			}  // for all attributes
		 }  // for all nearest  
		 // normalization of increments
		 //for (iAttr=discAttrFrom ; iAttr < discAttrTo ; ++iAttr) {
			// for (iV=0 ; iV <= discNoValues[iAttr] ; iV++) {
   //              resultCpAp[iAttr][iV] += clNorm(cl, currentClass) * incCpAp[iAttr][iV]/distanceSum ;
   //              resultCpAn[iAttr][iV] += clNorm(cl, currentClass) * incCpAn[iAttr][iV]/distanceSum ;
   //              resultCpA0[iAttr][iV] += clNorm(cl, currentClass) * incCpA0[iAttr][iV]/distanceSum ;
   //              resultCnAp[iAttr][iV] += clNorm(cl, currentClass) * incCnAp[iAttr][iV]/distanceSum ;
   //              resultCnAn[iAttr][iV] += clNorm(cl, currentClass) * incCnAn[iAttr][iV]/distanceSum ;
   //              resultCnA0[iAttr][iV] += clNorm(cl, currentClass) * incCnA0[iAttr][iV]/distanceSum ;
			// }
		 //}
		 if (distanceSum > 0) {
			for (iAttr=discAttrFrom ; iAttr < discAttrTo ; ++iAttr) {
			     iV = DiscValues(current, iAttr) ;
                 resultCpAp[iAttr][iV] += clNorm(cl, currentClass) * incCpAp[iAttr][iV]/distanceSum ;
                 resultCpAn[iAttr][iV] += clNorm(cl, currentClass) * incCpAn[iAttr][iV]/distanceSum ;
                 resultCpAe[iAttr][iV] += clNorm(cl, currentClass) * incCpAe[iAttr][iV]/distanceSum ;
                 resultCnAp[iAttr][iV] += clNorm(cl, currentClass) * incCnAp[iAttr][iV]/distanceSum ;
                 resultCnAn[iAttr][iV] += clNorm(cl, currentClass) * incCnAn[iAttr][iV]/distanceSum ;
                 resultCnAe[iAttr][iV] += clNorm(cl, currentClass) * incCnAe[iAttr][iV]/distanceSum ;
                 resultCeAp[iAttr][iV] += clNorm(cl, currentClass) * incCeAp[iAttr][iV]/distanceSum ;
                 resultCeAn[iAttr][iV] += clNorm(cl, currentClass) * incCeAn[iAttr][iV]/distanceSum ;
                 resultCeAe[iAttr][iV] += clNorm(cl, currentClass) * incCeAe[iAttr][iV]/distanceSum ;
	
				 iV = 0 ;  // for averaging
                 resultCpAp[iAttr][iV] += clNorm(cl, currentClass) * incCpAp[iAttr][iV]/distanceSum ;
                 resultCpAn[iAttr][iV] += clNorm(cl, currentClass) * incCpAn[iAttr][iV]/distanceSum ;
                 resultCpAe[iAttr][iV] += clNorm(cl, currentClass) * incCpAe[iAttr][iV]/distanceSum ;
                 resultCnAp[iAttr][iV] += clNorm(cl, currentClass) * incCnAp[iAttr][iV]/distanceSum ;
                 resultCnAn[iAttr][iV] += clNorm(cl, currentClass) * incCnAn[iAttr][iV]/distanceSum ;
                 resultCnAe[iAttr][iV] += clNorm(cl, currentClass) * incCnAe[iAttr][iV]/distanceSum ;			 
                 resultCeAp[iAttr][iV] += clNorm(cl, currentClass) * incCeAp[iAttr][iV]/distanceSum ;
                 resultCeAn[iAttr][iV] += clNorm(cl, currentClass) * incCeAn[iAttr][iV]/distanceSum ;
                 resultCeAe[iAttr][iV] += clNorm(cl, currentClass) * incCeAe[iAttr][iV]/distanceSum ;			 
			}
		 }
      } // for all classes
   }  
   //for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
   //   for (iV=0 ; iV <= discNoValues[iAttr] ; iV++) {
          // result[iAttr][iV] /= double(NoIterations) ;
          // result[iAttr][iV] =  resultP[iAttr][iV] - resultN[iAttr][iV] ;

   //    }
 
}

  
 // ***************************************************************************
//
//                   DAdiffSign
//              diff function of discrete attribute
//
// ***************************************************************************
inline double estimation::DAdiffSign(int AttrIdx, int I1, int I2) {

  // we assume that missing value has value 0
  int dV1 = DiscValues(I1, AttrIdx) ;
  int dV2 = DiscValues(I2, AttrIdx) ;
  if (dV1 == NAdisc)
     return 0 ; //NAdiscValue(DiscValues(I1,0),AttrIdx)[int(dV2)] ;
  else
    if (dV2 == NAdisc)
      return 0 ; // NAdiscValue(DiscValues(I2,0),AttrIdx)[int(dV1)] ;
     else
       // return double(dV2-dV1)/double(discNoValues[AttrIdx]-1) ;
       return sign(dV2-dV1) ;
}

inline double estimation::DAdiffOrd(int AttrIdx, int I1, int I2) {

  // we assume that missing value has value 0
  int dV1 = DiscValues(I1, AttrIdx) ;
  int dV2 = DiscValues(I2, AttrIdx) ;
  if (dV1 == NAdisc)
     return NAdiscValue(DiscValues(I1,0),AttrIdx)[int(dV2)] ;
  else
    if (dV2 == NAdisc)
      return NAdiscValue(DiscValues(I2,0),AttrIdx)[int(dV1)] ;
     else
       return double(dV2-dV1)/double(discNoValues[AttrIdx]-1) ;
}
void estimation::computeDistancesOrd(int Example) {
   int i ;
   for (int j=0 ; j < TrainSize ; j++)
   {
      if (Example == j)
      {
         for (i=0; i<contUpper; i++)
           ContDistance.Set(j, i, 0.0) ;
         for (i=0 ; i < discUpper ; i++)
           DiscDistance.Set(j, i, 0.0) ;
      }
      else {
        for (i=0; i<contUpper; i++)
          ContDistance.Set(j, i, CAdiff(i,Example,j)) ;
        for (i=0 ; i < discUpper ; i++)
          DiscDistance.Set(j, i, fabs(DAdiffOrd(i,Example,j))) ;
      }
   }
}

// ***************************************************************************
//
//                       ordAV3clReliefF
//                       -------
//
//                   
//
// ***************************************************************************
// ***************************************************************************
void estimation::ordAV3clReliefF(int discAttrFrom, int discAttrTo, 
	        marray<marray<double> > &resultCpAp, marray<marray<double> > &resultCpAn,
			marray<marray<double> > &resultCpAe, 
			marray<marray<double> > &resultCnAp, marray<marray<double> > &resultCnAn,
			marray<marray<double> > &resultCnAe,
			marray<marray<double> > &resultCeAp, marray<marray<double> > &resultCeAn,
			marray<marray<double> > &resultCeAe, 
			int distanceType) {

   int iA, iV ;
   // empty the results arrays
   for (iA=discAttrFrom ; iA < discAttrTo ; iA++) {
	   resultCpAe[iA].init(0.0) ;
	   resultCpAp[iA].init(0.0) ;
	   resultCpAn[iA].init(0.0) ;
 	   resultCnAe[iA].init(0.0) ;
	   resultCnAp[iA].init(0.0) ;
	   resultCnAn[iA].init(0.0) ;
 	   resultCeAe[iA].init(0.0) ;
	   resultCeAp[iA].init(0.0) ;
	   resultCeAn[iA].init(0.0) ;
  }
	   
   // number of examples belonging to each of the classes
   marray<double> probClass(NoClasses+1) ;
   probClass.init(0.0) ;
   int i, j, idx, aVal ;
   for (i=0 ; i < TrainSize ; i++) 
      probClass[ DiscValues(i,0) ] += weight[i] ;

   // compute estimations of class value probabilities with their relative frequencies
   for (idx=1 ; idx <= NoClasses ; idx++)
      probClass[idx] = probClass[idx] / double(TrainSize)  ;

   // data structure to hold nearest hits, + and - misses
   for (int iClss = 0 ; iClss <= NoClasses; iClss++)  {
      distanceArray[iClss].create(TrainSize) ;
      diffSorted[iClss].create(TrainSize) ;
   }

   // normalization of contribution of misses
   double pLower, pHigher ;
   mmatrix<double> clNorm(NoClasses+1,3) ;
   for (i=1 ; i<=NoClasses ; i++)  {
	 pLower = pHigher = 0.0 ;
	 for (j=1 ; j < i ; j++)
		 pLower +=   probClass[j] ;
	 for (j=i+1; j<=NoClasses ; j++)
		 pHigher +=   probClass[j] ;
     clNorm(i, 0) = probClass[i] ; // 1.0 ; //hit
     clNorm(i, 1) = pLower ; // /(1.0-probClass[i])  ;
	 clNorm(i, 2) = pHigher ;// /(1.0-probClass[i]) ;
   }
   // we have to compute distances up to the folowing attributes
   discUpper = NoDiscrete ;
   contUpper = NoContinuous ;

   double distanceSum, normDistance, Adiff ;
   int current, neighbourIdx, cl, iAttr, currentClass ;
												   
   marray<marray<double> > incCpAe(NoDiscrete+1), incCpAn(NoDiscrete+1), incCpAp(NoDiscrete+1),
	                       incCnAe(NoDiscrete+1), incCnAn(NoDiscrete+1), incCnAp(NoDiscrete+1),
	                       incCeAe(NoDiscrete+1), incCeAn(NoDiscrete+1), incCeAp(NoDiscrete+1)  ;

   for (iA=0 ; iA < NoDiscrete ; iA++) {
	   incCpAp[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCpAn[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCpAe[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCnAp[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCnAn[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCnAe[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCeAp[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCeAn[iA].create(discNoValues[iA]+1, 0.0) ;
	   incCeAe[iA].create(discNoValues[iA]+1, 0.0) ;
   }
       
   // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
  
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)  {

	   current = sampleIdx[iterIdx] ;
       currentClass =  DiscValues(current, 0) ;
      // first we compute distances of  all other examples to current
      computeDistancesOrd(current) ;

      // compute distance factors
      prepare3clDistanceFactors(current, distanceType) ;

      for (cl=0 ; cl<=2 ; cl++) {
         // compute sum of diffs
		 for (iA=discAttrFrom ; iA < discAttrTo ; ++iA) {
             incCpAp[iA].init(0.0) ;
             incCpAn[iA].init(0.0) ;
			 incCpAe[iA].init(0.0) ;
			 incCnAp[iA].init(0.0) ;
             incCnAn[iA].init(0.0) ;
			 incCnAe[iA].init(0.0) ;
			 incCeAp[iA].init(0.0) ;
             incCeAn[iA].init(0.0) ;
			 incCeAe[iA].init(0.0) ;
		 }
         distanceSum = 0.0 ;
		 // clDiff = DAdiffSign(0, current, distanceArray[cl][0].value) ;     

         for (i=0 ; i < distanceArray[cl].filled() ; i++) {
            neighbourIdx = distanceArray[cl][i].value ;
            normDistance = distanceArray[cl][i].key ;
            distanceSum += normDistance ;

            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++) {
               Adiff = DAdiffSign(iAttr, current, neighbourIdx) ;
			   //aVal = DiscValues(neighbourIdx, iAttr) ;	  
			   aVal = DiscValues(current, iAttr) ;
			   if (aVal != NAdisc) {
  			      if (cl==0) { //hit				   
				      if (Adiff==0) {
					      incCeAe[iAttr][aVal] +=  normDistance  ;
			              incCeAe[iAttr][0] +=  normDistance  ;
				       }
				       else if (Adiff > 0.0) {	 
					      incCeAp[iAttr][aVal] +=  Adiff * normDistance  ;
			              incCeAp[iAttr][0] +=  Adiff * normDistance  ;
				       }
				       else {
					      incCeAn[iAttr][aVal] -=  Adiff * normDistance  ;
			              incCeAn[iAttr][0] -=  Adiff * normDistance  ;
				       }
				   }
				   else if (cl==2) { // larger class
					   if (Adiff==0) {
					      incCpAe[iAttr][aVal] +=  normDistance  ;
			              incCpAe[iAttr][0] +=  normDistance  ;
				       }
				       else if (Adiff > 0.0) {	 
					      incCpAp[iAttr][aVal] +=  Adiff * normDistance  ;
			              incCpAp[iAttr][0] +=  Adiff * normDistance  ;
				       }
				       else {
					      incCpAn[iAttr][aVal] -=  Adiff * normDistance  ;
			              incCpAn[iAttr][0] -=  Adiff * normDistance  ;
				       }
				   }
				   else {   // cl == 1, clDiff < 0, lower class
					   if (Adiff==0) {
					      incCnAe[iAttr][aVal] +=  normDistance  ;
			              incCnAe[iAttr][0] +=  normDistance  ;
				       }
				       else if (Adiff > 0.0) {	 
					      incCnAp[iAttr][aVal] +=  Adiff * normDistance  ;
			              incCnAp[iAttr][0] +=  Adiff * normDistance  ;
				       }
				       else {
					      incCnAn[iAttr][aVal] -=  Adiff * normDistance  ;
			              incCnAn[iAttr][0] -=  Adiff * normDistance  ;
				       }
				   
				   }
               }
			}  // for all attributes
		 }  // for all nearest  
		 // normalization of increments
		 if (distanceSum > 0) {
			for (iAttr=discAttrFrom ; iAttr < discAttrTo ; ++iAttr) {
			    iV = DiscValues(current, iAttr) ;
                resultCpAp[iAttr][iV] += clNorm(currentClass,cl) * incCpAp[iAttr][iV]/distanceSum ;
	            resultCpAn[iAttr][iV] += clNorm(currentClass,cl) * incCpAn[iAttr][iV]/distanceSum ;
		        resultCpAe[iAttr][iV] += clNorm(currentClass,cl) * incCpAe[iAttr][iV]/distanceSum ;
			    resultCnAp[iAttr][iV] += clNorm(currentClass,cl) * incCnAp[iAttr][iV]/distanceSum ;
			    resultCnAn[iAttr][iV] += clNorm(currentClass,cl) * incCnAn[iAttr][iV]/distanceSum ;
                resultCnAe[iAttr][iV] += clNorm(currentClass,cl) * incCnAe[iAttr][iV]/distanceSum ;
			    resultCeAp[iAttr][iV] += clNorm(currentClass,cl) * incCeAp[iAttr][iV]/distanceSum ;
			    resultCeAn[iAttr][iV] += clNorm(currentClass,cl) * incCeAn[iAttr][iV]/distanceSum ;
                resultCeAe[iAttr][iV] += clNorm(currentClass,cl) * incCeAe[iAttr][iV]/distanceSum ;
	  			iV = 0 ;  // for averaging
                resultCpAp[iAttr][iV] += clNorm(currentClass,cl) * incCpAp[iAttr][iV]/distanceSum ;
	            resultCpAn[iAttr][iV] += clNorm(currentClass,cl) * incCpAn[iAttr][iV]/distanceSum ;
		        resultCpAe[iAttr][iV] += clNorm(currentClass,cl) * incCpAe[iAttr][iV]/distanceSum ;
			    resultCnAp[iAttr][iV] += clNorm(currentClass,cl) * incCnAp[iAttr][iV]/distanceSum ;
			    resultCnAn[iAttr][iV] += clNorm(currentClass,cl) * incCnAn[iAttr][iV]/distanceSum ;
                resultCnAe[iAttr][iV] += clNorm(currentClass,cl) * incCnAe[iAttr][iV]/distanceSum ;			 
			    resultCeAp[iAttr][iV] += clNorm(currentClass,cl) * incCeAp[iAttr][iV]/distanceSum ;
			    resultCeAn[iAttr][iV] += clNorm(currentClass,cl) * incCeAn[iAttr][iV]/distanceSum ;
                resultCeAe[iAttr][iV] += clNorm(currentClass,cl) * incCeAe[iAttr][iV]/distanceSum ;			 
			}
	     } 	
      } // for all classes
   }  
   //for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
   //   for (iV=0 ; iV <= discNoValues[iAttr] ; iV++) {
          // result[iAttr][iV] /= double(NoIterations) ;
          // result[iAttr][iV] =  resultP[iAttr][iV] - resultN[iAttr][iV] ;

   //    }
 
}

 // ***************************************************************************
//
//                          prepare3clDistanceFactors
// computation of distance probability weight factors for given example
//
// ***************************************************************************
void estimation::prepare3clDistanceFactors(int current, int distanceType) {
// we use only original attributes to obtain distance in attribute space

   int kSelected = 0 ;
   switch (distanceType)   {
      case estReliefFkEqual:
              kSelected = kNearestEqual ;
              break ;
      case estReliefFexpRank: 
              kSelected = kDensity ;
              break ;
      case estReliefFbestK:
              kSelected = TrainSize ;  // we have to consider all neighbours
      default: error("estimation::prepare3clDistanceFactors","invalid distance type") ;
   }
   int i, cl ;
   sortRec tempSort ;
   for (cl = 0 ; cl <= 2; cl++) {
      // empty data structures
      distanceArray[cl].clear() ;
      diffSorted[cl].clear() ;
   }

   // distances in attributes space
   int bunch, currentCl = DiscValues(current, 0) ;
   for (i=0 ; i < TrainSize; i++)  {
      if (i==current)  // we skip current example
         continue ;
      tempSort.key =  CaseDistance(i) ;
      tempSort.value = i ;
	  if (DiscValues(i,0) < currentCl)   // lower
		  bunch = 1 ;
	  else if (DiscValues(i,0) > currentCl)	// higher
		  bunch = 2 ;
	  else bunch = 0 ; // hits
      
      diffSorted[bunch].addEnd(tempSort) ;
   }

   // sort examples 
   for (cl=0 ; cl <= 2 ; cl++)    {
      // we sort groups of examples according to ascending distance from current
      if (diffSorted[cl].filled() > 1)
         diffSorted[cl].sortKdsc(Mmin(kSelected, diffSorted[cl].filled())) ;
   }

   int upper, idx ;
   double factor ;
   // depending on tpe of distance, copy the nearest cases
   // and their distance factors into resulting array
   switch (distanceType)   {      
        case estReliefFkEqual: 
        case estReliefFbestK:
          {
            for (cl=0; cl <= 2 ; cl++)    {
               idx =  diffSorted[cl].filled() -1;
               upper = Mmin(kSelected, diffSorted[cl].filled()) ;
               for (i=0 ; i < upper ; i++) {
                  distanceArray[cl][i].value = diffSorted[cl][idx].value ;
                  idx -- ;
                  distanceArray[cl][i].key = 1.0  ;
               }
               distanceArray[cl].setFilled(upper) ;
            }
          }
          break ;
        case estReliefFexpRank: 
          {
            for (cl=0; cl <= 2 ; cl++)
            {
               upper = Mmin(kSelected, diffSorted[cl].filled()) ;
               distanceArray[cl].setFilled(upper) ;
               if (upper < 1)  // are there any elements
                  continue ;
               idx =  diffSorted[cl].filled() -1;
               factor = 1.0  ;
               distanceArray[cl][0].key =  factor ;
               distanceArray[cl][0].value = diffSorted[cl][idx].value ;
               idx -- ;
               for (i=1 ; i < upper ; i++) {
                  if (diffSorted[cl][idx].key != diffSorted[cl][idx+1].key)
                     factor = double(exp(-sqr(double(i))/varianceDistanceDensity)) ;
                  distanceArray[cl][i].key =  factor ;
                  distanceArray[cl][i].value = diffSorted[cl][idx].value ;
                  idx -- ;
               }
            }
          }
          break ;
        case estReliefFdistance:
          {
            double minNonZero = FLT_MAX ; // minimal non zero distance
            for (cl=1; cl <= NoClasses ; cl++)
               for (i= diffSorted[cl].filled() -1 ; i >= 0 ; i--)
                  if (diffSorted[cl][i].key > 0.0) {
                     if (diffSorted[cl][i].key < minNonZero)
                        minNonZero = diffSorted[cl][i].key ;
                     break;
                  }
            if (minNonZero == FLT_MAX)
               minNonZero = 1.0 ;

            for (cl=1; cl <= NoClasses ; cl++) {
               idx =  diffSorted[cl].filled() -1;
               upper = Mmin(kSelected, diffSorted[cl].filled()) ;
               for (i=0 ; i < upper ; i++)
               {
                  if (diffSorted[cl][idx].key > 0)
                     factor = 1.0 / diffSorted[cl][idx].key ;
                  else 
                     factor = 2.0 / minNonZero ;
                  distanceArray[cl][i].value = diffSorted[cl][idx].value ;
                  distanceArray[cl][i].key = factor  ;
                  idx -- ;
               }
               distanceArray[cl].setFilled(upper) ;
            }
          }
          break ;
        default: error("estimation::prepare3clDistanceFactors","invalid distanceType detected") ;
   }
}
