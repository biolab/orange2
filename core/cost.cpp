
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include "general.h"
#include "estimator.h"                
#include "contain.h"
#include "utils.h"


// ***************************************************************************
//
//                       ReliefFpa
//                       -------
//
//   contains two versions of ReliefFcost:
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//    average cost of missclassification
// ***************************************************************************
void estimation::ReliefFpa(int contAttrFrom, int contAttrTo,
                  int discAttrFrom, int discAttrTo, int distanceType)
{

   ContEstimation.init(contAttrFrom,contAttrTo,0.0) ;
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // prepare estimations arrays
   int NoContEstimated = contAttrTo - contAttrFrom ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
 
   // number of examples belonging to each of the classes
   marray<int> noExInClass(NoClasses+1) ;
   marray<double> probClass(NoClasses+1) ;
   noExInClass.init(0) ;
   probClass.init(0.0) ;
   int i, j, idx ;
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

   // compute estimations of class value probabilities with their 
   // relative frequencies
   for (idx=1 ; idx <= NoClasses ; idx++)
      probClass[idx] = probClass[idx] / wAll ;

   // initialize weights for all the attributes and all the k
   marray<double> PhitDisc(NoDiscEstimated, 0.0) ;
   marray<double> PmissDisc(NoDiscEstimated, 0.0) ;
   marray<double> PhitCont(NoContEstimated, 0.0) ;
   marray<double> PmissCont(NoContEstimated, 0.0) ;
 
   // data structure to hold nearest hits/misses
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)
   {
      distanceArray[iClss].create(noExInClass[iClss]) ;
      diffSorted[iClss].create(noExInClass[iClss]) ;
   }

   marray<double> avgCost(NoClasses+1, 0.0) ;
   double avgCostSum = 0.0 ;
   for (i=1 ; i<=NoClasses ; i++) {
     for (j=1 ; j<=NoClasses ; j++)
         if (j != i) {
           avgCost[i] += fTree->CostMatrix(i, j) ;
         }
     avgCost[i] /= (NoClasses - 1.0) ;
     avgCostSum += avgCost[i] ;
   }
   
   // normalization of contributions for hit/misses
   mmatrix<double> clNorm(NoClasses+1,NoClasses+1) ;
   for (i=1 ; i<=NoClasses ; i++)
     for (j=1 ; j<=NoClasses ; j++)
         if (i==j) clNorm(i,i) =  1 ;
         else clNorm(i,j) =  (avgCost[j]/avgCostSum)/(1.0- avgCost[i]/avgCostSum) ;
                             //fTree->CostMatrix(i,j)/(NoClasses - 1.0); // /  avgCost[i] ;
                            // probClass[j]* fTree->CostMatrix(i,j)/(expCost[i]*(1-probClass[i])) ;
   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   double distanceSum, normDistance, Adiff ;
   int current, neighbourIdx, cl, iAttr, currentClass ;
   
   marray<double> incContDiffA(NoContEstimated), incDiscDiffA(NoDiscEstimated) ;
              
   // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;

   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++) {
       current = sampleIdx[iterIdx] ;
       // initialize (optimization reasons)
       currentClass =  DiscValues(current, 0) ;
       // first we compute distances of  all other examples to current
       computeDistances(current) ;
       // compute distance factors
       prepareDistanceFactors(current, distanceType) ;

      for (cl=1 ; cl<=NoClasses ; cl++)
      {
         // compute sum of diffs
         incContDiffA.init(0.0) ;
         incDiscDiffA.init(0.0) ;
         distanceSum = 0.0 ;
         for (i=0 ; i < distanceArray[cl].filled() ; i++)
         {
            neighbourIdx = distanceArray[cl][i].value ;
            normDistance = distanceArray[cl][i].key ;
            distanceSum += normDistance ;
                 
            // adjust the weights for all the estimated attributes and values
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
            {
               idx = iAttr - contAttrFrom ;
               Adiff = ContDistance(neighbourIdx, iAttr) ;
               incContDiffA[idx] += Adiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
            {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscDiffA[idx] +=  Adiff * normDistance  ;
            }
         }
         if (cl == currentClass) // hit or miss
         {
            // hit
            // normalization of increments
            for (idx=0 ; idx < NoContEstimated ; idx++)
              if (incContDiffA[idx] > epsilon)
                 PhitCont[idx] += clNorm(cl, cl) * incContDiffA[idx]/distanceSum ;
            for (idx=0 ; idx < NoDiscEstimated ; idx++)
              if (incDiscDiffA[idx] > epsilon)
                PhitDisc[idx] += clNorm(cl, cl) * incDiscDiffA[idx]/distanceSum ;
          }
          else
          {
             // miss
             // normalization of increments
             for (idx=0 ; idx < NoContEstimated ; idx++)
               if (incContDiffA[idx] > epsilon)
                 PmissCont[idx] += clNorm(currentClass, cl) * incContDiffA[idx]/distanceSum ;
             for (idx=0 ; idx < NoDiscEstimated ; idx++)
               if (incDiscDiffA[idx] > epsilon)
                 PmissDisc[idx] += clNorm(currentClass, cl) * incDiscDiffA[idx]/distanceSum ;
          }
      }
   }  
   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
   {
      idx = iAttr - contAttrFrom ;
      ContEstimation[iAttr] = (PmissCont[idx] - PhitCont[idx])/double(NoIterations) ;
  }
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
   {
      idx = iAttr - discAttrFrom ;
      DiscEstimation[iAttr] = (PmissDisc[idx] - PhitDisc[idx])/double(NoIterations) ;
  }
}


// ***************************************************************************
//
//                       ReliefFpe
//                       -------
//
//   contains two versions of ReliefFeCt, but different normalization
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//    probabilities by expected cost of missclassification
// ***************************************************************************
void estimation::ReliefFpe(int contAttrFrom, int contAttrTo,
                  int discAttrFrom, int discAttrTo, int distanceType)
{
   ContEstimation.init(contAttrFrom,contAttrTo,0.0) ;
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // prepare estimations arrays
   int NoContEstimated = contAttrTo - contAttrFrom ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
 
   // number of examples belonging to each of the classes
   marray<int> noExInClass(NoClasses+1) ;
   marray<double> probClass(NoClasses+1) ;
   noExInClass.init(0) ;
   probClass.init(0.0) ;
   int i, j, idx ;
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

   // compute estimations of class value probabilities with their 
   // relative frequencies
   for (idx=1 ; idx <= NoClasses ; idx++)
      probClass[idx] = probClass[idx] / wAll ;

   // initialize weights for all the attributes and all the k
   marray<double> PDisc(NoDiscEstimated, 0.0) ;
   marray<double> PCont(NoContEstimated, 0.0) ;
   
   // data structure to hold nearest hits/misses
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)
   {
      distanceArray[iClss].create(noExInClass[iClss]) ;
      diffSorted[iClss].create(noExInClass[iClss]) ;
   }

   // normalization of contribution of misses
   
   mmatrix<double> clNorm(NoClasses+1,NoClasses+1) ;
   marray<double> expCost(NoClasses+1, 0.0) ;
   double  eSum =0.0;
   for (i=1 ; i<=NoClasses ; i++) {
      for (j=1 ; j<=NoClasses ; j++) {
         if (j!=i) 
            expCost[i] += probClass[j] * fTree->CostMatrix(i,j) ;
      }
      expCost[i] /= (1.0 - probClass[i]) ;
      eSum += probClass[i]*expCost[i] ;
   }
   for (i=1 ; i<=NoClasses ; i++) 
      for (j=1 ; j<=NoClasses ; j++)
         if (j==i)
            clNorm(i,i) = - 1 ; // expCost[i] ;
         else clNorm(i,j) = (probClass[j] * expCost[j]/eSum /(1.0 - probClass[i]*expCost[i]/eSum)) ;
             // probClass[j] * fTree->CostMatrix(i,j) / (1.0 - probClass[i]) ;
   

   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   double distanceSum, normDistance, Adiff ;
   int current, neighbourIdx, cl, iAttr, currentClass ;
   
   marray<double> incContDiffA(NoContEstimated), incDiscDiffA(NoDiscEstimated) ;
       
   // prepare order of iterations, select them according to expected cost
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
        
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)
   {
       current = sampleIdx[iterIdx] ;
 
       // initialize (optimization reasons)
       currentClass =  DiscValues(current, 0) ;

       // first we compute distances of  all other examples to current
       computeDistances(current) ;

      // compute distance factors
      prepareDistanceFactors(current, distanceType) ;

      for (cl=1 ; cl<=NoClasses ; cl++)
      {
         // compute sum of diffs
         incContDiffA.init(0.0) ;
         incDiscDiffA.init(0.0) ;
         distanceSum = 0.0 ;
         for (i=0 ; i < distanceArray[cl].filled() ; i++)
         {
            neighbourIdx = distanceArray[cl][i].value ;
            normDistance = distanceArray[cl][i].key ;
            distanceSum += normDistance ;
                 
            // adjust the weights for all the estimated attributes and values
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
            {
               idx = iAttr - contAttrFrom ;
               Adiff = ContDistance(neighbourIdx, iAttr) ;
               incContDiffA[idx] += Adiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
            {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscDiffA[idx] +=  Adiff * normDistance  ;
            }
         }
 
         // normalization of increments for hits amd misses
         for (idx=0 ; idx < NoContEstimated ; idx++)
            if (incContDiffA[idx] > epsilon)
               PCont[idx] += clNorm(currentClass, cl) * incContDiffA[idx]/distanceSum ;
         for (idx=0 ; idx < NoDiscEstimated ; idx++)
            if (incDiscDiffA[idx] > epsilon)
              PDisc[idx] += clNorm(currentClass, cl) * incDiscDiffA[idx]/distanceSum ;
      }
   }  
   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
   {
      idx = iAttr - contAttrFrom ;
      ContEstimation[iAttr] = PCont[idx]/double(NoIterations) ;
  }
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
   {
      idx = iAttr - discAttrFrom ;
      DiscEstimation[iAttr] = PDisc[idx]/double(NoIterations) ;
  }
}

/*
// ***************************************************************************
//
//                       ReliefRcost
//                       -------
//
//   R: all near instances, disregarding the class
//   contains two versions of ReliefRcost:
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//
// ***************************************************************************
void estimation::ReliefRcost(int contAttrFrom, int contAttrTo,
                  int discAttrFrom, int discAttrTo, int distanceType)
{

   ContEstimation.init(contAttrFrom,contAttrTo,0.0) ;
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // prepare estimations arrays
   int NoContEstimated = contAttrTo - contAttrFrom ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
  
   // initialize weights for all the attributes and all the k
   marray<double> PDisc(NoDiscEstimated, 0.0) ;
   marray<double> PCont(NoContEstimated, 0.0) ;

   diffRsorted.create(TrainSize) ;
   distanceRarray.create(TrainSize) ;
   // marray<sortRec> diffSort(TrainSize) ;

   // normalization of contribution of misses
   mmatrix<double> clNorm(NoClasses+1,NoClasses+1) ;
   double sumCost ;
   int i,j ;
   for (i=1 ; i<=NoClasses ; i++) {
      sumCost = 0.0 ;
      for (j=1 ; j<=NoClasses ; j++)
         if (j!=i) 
            sumCost += fTree->CostMatrix(j,i) ;
      for (j=1 ; j<=NoClasses ; j++)
         if (j==i)
            clNorm(i,i) = sumCost / (NoClasses-1.0) ;
         else clNorm(j,i) = fTree->CostMatrix(j,i) ;
   }
   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   double distanceSum, normDistance, Adiff ;
   int current, neighbourIdx, cl, iAttr, currentClass, idx ;
   
   marray<double> incContA(NoContEstimated), incDiscA(NoDiscEstimated) ;
       
   // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
 
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)
   {
       current = sampleIdx[iterIdx] ;
       currentClass =  DiscValues(current, 0) ;
       
      // first we compute distances of  all other examples to current
      computeDistances(current) ;

      // compute distance factors
      RprepareDistanceFactors(current, distanceType) ;

      incContA.init(0.0) ;
      incDiscA.init(0.0) ;
      distanceSum = 0.0 ;

      for (i=0 ; i < distanceRarray.filled() ; i++)
      {
          neighbourIdx = distanceRarray[i].value ;
          normDistance = distanceRarray[i].key ;
          distanceSum += normDistance ;
          cl = DiscValues(neighbourIdx, 0) ;

         if (cl == currentClass) {
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++) {
              idx = iAttr - contAttrFrom ;
              Adiff = ContDistance(neighbourIdx, iAttr) ;
              incContA[idx] -= clNorm(cl, currentClass) * Adiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++) {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscA[idx] -= clNorm(cl, currentClass) *  Adiff * normDistance  ;
            }
          }
          else
          {
             // miss
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++) {
              idx = iAttr - contAttrFrom ;
              Adiff = ContDistance(neighbourIdx, iAttr) ;
              incContA[idx] += clNorm(cl, currentClass) * Adiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++) {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscA[idx] += clNorm(cl, currentClass) *  Adiff * normDistance  ;
            }
          }
      } 
      //  normalize contribution of this example
      for (idx=0 ; idx < NoContEstimated ; idx++) 
           PCont[idx] += incContA[idx]/distanceSum ;
      for (idx=0 ; idx < NoDiscEstimated ; idx++)
           PDisc[idx] += incDiscA[idx]/distanceSum ;
   }  
   // final averaging
   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)   {
      idx = iAttr - contAttrFrom ;
      ContEstimation[iAttr] /= PCont[idx] /double(NoIterations) ;
  }
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)  {
      idx = iAttr - discAttrFrom ;
      DiscEstimation[iAttr] = PDisc[idx]/double(NoIterations) ;
  }
}


// ***************************************************************************
//
//                          RprepareDistanceFactors
// computation of distance probability weight factors for given example
// instances are not grouped by class
//
// ***************************************************************************
void estimation::RprepareDistanceFactors(int current, int distanceType)
{

// we use only original attributes to obtain distance in attribute space

   int kSelected = 0 ;
   switch (distanceType)
   {
       case estReliefRcostEqualK:
              kSelected = kNearestEqual ;
              break ;

      case estReliefRcostExpRank:
              kSelected = kDensity ;
              break ;
      default: error("estimation::prepareDistanceFactors","invalid distance type") ;
   }

   int i ;
   sortRec tempSort ;

   // empty data structures
   distanceRarray.clear() ;
   diffRsorted.clear() ;
   
   // distances in attributes space
   for (i=0 ; i < TrainSize; i++)
   {
      if (i==current)  // we skip current example
         continue ;
      tempSort.key =  CaseDistance(i) ;
      tempSort.value = i ;
      diffRsorted.addEnd(tempSort) ;
   }

   // sort examples according to ascending distance from current
      if (diffRsorted.filled() > 1)
         diffRsorted.sortKdsc(Mmin(kSelected, diffRsorted.filled())) ;
   

   int upper, idx ;
   double factor ;
   // depending on type of distance, copy the nearest cases
   // and their distance factors into resulting array
   switch (distanceType)
   {
        case estReliefRcostEqualK:
          {
               idx =  diffRsorted.filled() -1;
               upper = Mmin(kSelected, diffRsorted.filled()) ;
               for (i=0 ; i < upper ; i++)
               {
                  distanceRarray[i].value = diffRsorted[idx].value ;
                  idx -- ;
                  distanceRarray[i].key = 1.0  ;
               }
               distanceRarray.setFilled(upper) ;
          }
          break ;
        case estReliefRcostExpRank:
          {
               upper = Mmin(kSelected, diffRsorted.filled()) ;
               distanceRarray.setFilled(upper) ;
               if (upper < 1)  // are there any elements
                  break ;
               idx =  diffRsorted.filled() -1;
               factor = 1.0  ;
               distanceRarray[0].key =  factor ;
               distanceRarray[0].value = diffRsorted[idx].value ;
               idx -- ;
               for (i=1 ; i < upper ; i++)
               {
                  if (diffRsorted[idx].key != diffRsorted[idx+1].key)
                     factor = double(exp(-sqr(double(i))/varianceDistanceDensity)) ;
                  distanceRarray[i].key =  factor ;
                  distanceRarray[i].value = diffRsorted[idx].value ;
                  idx -- ;
               }
          }
          break ;
        default: error("estimation::RprepareDistanceFactors","invalid distanceType detected") ;
   }
}

// ***************************************************************************
//
//                          EprepareDistanceFactors
// computation of distance probability weight factors for given example
// instances are not grouped by class
//
// ***************************************************************************
void estimation::EprepareDistanceFactors(int current, int distanceType)
{
  // we use only original attributes to obtain distance in attribute space
   int kSelected = 0 ;
   switch (distanceType)
   {
       case estReliefEcostEqualK:
              kSelected = kNearestEqual ;
              break ;

      case estReliefEcostExpRank:
              kSelected = kDensity ;
              break ;
      default: error("estimation::EprepareDistanceFactors","invalid distance type") ;
   }

   int i ;
   sortRec tempSort ;

   // empty data structures
   distanceEMarray.clear() ;
   distanceEHarray.clear() ;
   diffEMsorted.clear() ;
   diffEHsorted.clear() ;
   
   int cl=DiscValues(current,0) ;

   // distances in attributes space
   for (i=0 ; i < TrainSize; i++)
   {
      if (i==current)  // we skip current example
         continue ;
      tempSort.key =  CaseDistance(i) ;
      tempSort.value = i ;
      if (DiscValues(i,0)==cl) 
          diffEHsorted.addEnd(tempSort) ; // add to nearest misses
      else
          diffEMsorted.addEnd(tempSort) ; // add to nearest hits
   }

   // sort examples 
   // we sort groups of examples according to ascending distance from current
   if (diffEHsorted.filled() > 1)
        diffEHsorted.sortKdsc(Mmin(kSelected, diffEHsorted.filled())) ;
   if (diffEMsorted.filled() > 1)
        diffEMsorted.sortKdsc(Mmin(kSelected, diffEMsorted.filled())) ;

   int upper, idx ;
   double factor ;
   // depending on tpe of distance, copy the nearest cases
   // and their distance factors into resulting array
   switch (distanceType)
   {
        case estReliefEcostEqualK: 
          {
               // hits
               idx =  diffEHsorted.filled() -1;
               upper = Mmin(kSelected, diffEHsorted.filled()) ;
               for (i=0 ; i < upper ; i++)
               {
                  distanceEHarray[i].value = diffEHsorted[idx].value ;
                  idx -- ;
                  distanceEHarray[i].key = 1.0  ;
               }
               distanceEHarray.setFilled(upper) ;

               // misses
               idx =  diffEMsorted.filled() -1;
               upper = Mmin(kSelected, diffEMsorted.filled()) ;
               for (i=0 ; i < upper ; i++)
               {
                  distanceEMarray[i].value = diffEMsorted[idx].value ;
                  idx -- ;
                  distanceEMarray[i].key = 1.0  ;
               }
               distanceEMarray.setFilled(upper) ;
          }
          break ;
        case estReliefEcostExpRank:
          {
               // hits
               upper = Mmin(kSelected, diffEHsorted.filled()) ;
               distanceEHarray.setFilled(upper) ;
               if (upper > 0) { // are there any elements
                 idx =  diffEHsorted.filled() -1;
                 factor = 1.0  ;
                 distanceEHarray[0].key =  factor ;
                 distanceEHarray[0].value = diffEHsorted[idx].value ;
                 idx -- ;
                 for (i=1 ; i < upper ; i++)
                 {
                    if (diffEHsorted[idx].key != diffEHsorted[idx+1].key)
                     factor = double(exp(-sqr(double(i))/varianceDistanceDensity)) ;
                    distanceEHarray[i].key =  factor ;
                    distanceEHarray[i].value = diffEHsorted[idx].value ;
                    idx -- ;
                 }
               }
               // misses
               upper = Mmin(kSelected, diffEMsorted.filled()) ;
               distanceEMarray.setFilled(upper) ;
               if (upper > 0) { // are there any elements
                 idx =  diffEMsorted.filled() -1;
                 factor = 1.0  ;
                 distanceEMarray[0].key =  factor ;
                 distanceEMarray[0].value = diffEMsorted[idx].value ;
                 idx -- ;
                 for (i=1 ; i < upper ; i++)
                 {
                    if (diffEMsorted[idx].key != diffEMsorted[idx+1].key)
                     factor = double(exp(-sqr(double(i))/varianceDistanceDensity)) ;
                    distanceEMarray[i].key =  factor ;
                    distanceEMarray[i].value = diffEMsorted[idx].value ;
                    idx -- ;
                 }
               }
          }
          break ;
        default: error("estimation::EprepareDistanceFactors","invalid distanceType detected") ;
   }
}

// ***************************************************************************
//
//                       ReliefEcost
//                       -------
//
//   E: all near instances, hits and misses separately
//   contains two versions of ReliefEcost:
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//
// ***************************************************************************
void estimation::ReliefEcost(int contAttrFrom, int contAttrTo,
                  int discAttrFrom, int discAttrTo, int distanceType)
{
   ContEstimation.init(contAttrFrom,contAttrTo,0.0) ;
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // prepare estimations arrays
   int NoContEstimated = contAttrTo - contAttrFrom ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
  
   // initialize weights for all the attributes and all the k
   marray<double> PDisc(NoDiscEstimated, 0.0) ;
   marray<double> PCont(NoContEstimated, 0.0) ;

   diffEHsorted.create(TrainSize) ;
   diffEMsorted.create(TrainSize) ;
   distanceEHarray.create(TrainSize) ;
   distanceEMarray.create(TrainSize) ;

   // normalization of contribution of misses
   mmatrix<double> clNorm(NoClasses+1,NoClasses+1) ;
   double sumCost ;
   int i,j ;
   for (i=1 ; i<=NoClasses ; i++) {
      sumCost = 0.0 ;
      for (j=1 ; j<=NoClasses ; j++)
         if (j!=i) 
            sumCost += fTree->CostMatrix(j,i) ;
      for (j=1 ; j<=NoClasses ; j++)
         if (j==i)
            clNorm(i,i) = sumCost / (NoClasses-1.0) ;
         else clNorm(j,i) = fTree->CostMatrix(j,i) ;
   }
   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   double distanceSum, normDistance, Adiff ;
   int current, neighbourIdx, cl, iAttr, currentClass, idx ;
   
   marray<double> incContA(NoContEstimated), incDiscA(NoDiscEstimated) ;
       
   // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
 
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)
   {
       current = sampleIdx[iterIdx] ;
       currentClass =  DiscValues(current, 0) ;
       
      // first we compute distances of  all other examples to current
      computeDistances(current) ;

      // compute distance factors
      EprepareDistanceFactors(current, distanceType) ;

      incContA.init(0.0) ;
      incDiscA.init(0.0) ;
      distanceSum = 0.0 ;

      for (i=0 ; i < distanceRarray.filled() ; i++)
      {
          neighbourIdx = distanceRarray[i].value ;
          normDistance = distanceRarray[i].key ;
          distanceSum += normDistance ;
          cl = DiscValues(neighbourIdx, 0) ;

         if (cl == currentClass) {
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++) {
              idx = iAttr - contAttrFrom ;
              Adiff = ContDistance(neighbourIdx, iAttr) ;
              incContA[idx] -= clNorm(cl, currentClass) * Adiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++) {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscA[idx] -= clNorm(cl, currentClass) *  Adiff * normDistance  ;
            }
          }
          else
          {
             // miss
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++) {
              idx = iAttr - contAttrFrom ;
              Adiff = ContDistance(neighbourIdx, iAttr) ;
              incContA[idx] += clNorm(cl, currentClass) * Adiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++) {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscA[idx] += clNorm(cl, currentClass) *  Adiff * normDistance  ;
            }
          }
      } 
      //  normalize contribution of this example
      for (idx=0 ; idx < NoContEstimated ; idx++) 
           PCont[idx] += incContA[idx]/distanceSum ;
      for (idx=0 ; idx < NoDiscEstimated ; idx++)
           PDisc[idx] += incDiscA[idx]/distanceSum ;
   }  
   // final averaging
   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)   {
      idx = iAttr - contAttrFrom ;
      ContEstimation[iAttr] = PCont[idx] /double(NoIterations) ;
  }
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)  {
      idx = iAttr - discAttrFrom ;
      DiscEstimation[iAttr] = PDisc[idx]/double(NoIterations) ;
  }
}
*/

// ***************************************************************************
//
//                       ReliefFcostKukar
//                       -------
//
// ***************************************************************************
void estimation::ReliefFcostKukar(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo)
{

   ContEstimation.init(contAttrFrom,contAttrTo,0.0) ;
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // prepare estimations arrays
   int NoContEstimated = contAttrTo - contAttrFrom ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
  
   // initialize weights for all the attributes and all the k
   marray<double> PDisc(NoDiscEstimated, 0.0) ;
   marray<double> PCont(NoContEstimated, 0.0) ;
 
   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   marray<double> costVector(NoClasses+1,0.0), costWeight(NoClasses+1,0.0) ;
   double sumCost, sumVector=0.0 ;
   int i, j ;
   for (i=1 ; i<=NoClasses ; i++) {
      sumCost = 0.0 ;
      for (j=1 ; j<=NoClasses ; j++)
         if (j!=i) 
            sumCost += fTree->AttrDesc[0].valueProbability[j] * fTree->CostMatrix(i,j) ;
      costVector[i] = sumCost/(1.0 - fTree->AttrDesc[0].valueProbability[i])  ;
      sumVector += fTree->AttrDesc[0].valueProbability[i] * costVector[i] ;
   }
   for (i=1 ; i<=NoClasses ; i++) 
       costWeight[i] = costVector[i] / sumVector ;

   int current, idx, iAttr, hit, miss, currentClass ;
   
   // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
  
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)
   {
      current = sampleIdx[iterIdx] ;
      currentClass = DiscValues(current, 0) ;
  
	   // first we compute distances of  all other examples to current
      computeDistances(current) ;

      findHitMiss(current, hit, miss) ;

      // adjust the weights for all the estimated attributes and values
      for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)   {
           idx = iAttr - contAttrFrom ;
           PCont[idx] += costWeight[currentClass] * ContDistance(miss, iAttr) - ContDistance(hit, iAttr) ;
      }
      for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)  {
         idx = iAttr - discAttrFrom ;
         PDisc[idx] += costWeight[currentClass] * DiscDistance(miss, iAttr) - DiscDistance(hit, iAttr) ;
      }
   }  
   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
   {
      idx = iAttr - contAttrFrom ;
      ContEstimation[iAttr] = PCont[idx]/double(NoIterations) ;
      #ifdef DEBUG
      if (ContEstimation[iAttr] > 1.00001 || ContEstimation[iAttr] < -1.00001)
        error("estimation::ReliefCostKukar", "computed continuous weights are out of scope") ;
      #endif
   }
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
   {
      idx = iAttr - discAttrFrom ;
      DiscEstimation[iAttr] = PDisc[idx]/double(NoIterations) ;
      #ifdef DEBUG
      if (DiscEstimation[iAttr] > 1.00001 || DiscEstimation[iAttr] < -1.00001)
        error("estimation::ReliefCostKukar", "computed discrete weights are out of scope") ;
      #endif
   } 
}


// ***************************************************************************
//
//                       ReliefFavgC
//                       -------
//
//   contains two versions of ReliefFcost:
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//    average cost of missclassification
// ***************************************************************************
void estimation::ReliefFavgC(int contAttrFrom, int contAttrTo,
                  int discAttrFrom, int discAttrTo, int distanceType)
{

   ContEstimation.init(contAttrFrom,contAttrTo,0.0) ;
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // prepare estimations arrays
   int NoContEstimated = contAttrTo - contAttrFrom ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
 
   // number of examples belonging to each of the classes
   marray<int> noExInClass(NoClasses+1) ;
   marray<double> probClass(NoClasses+1) ;
   noExInClass.init(0) ;
   probClass.init(0.0) ;
   int i, j, idx ;
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

   // compute estimations of class value probabilities with their 
   // relative frequencies
   for (idx=1 ; idx <= NoClasses ; idx++)
      probClass[idx] = probClass[idx] / wAll ;

   // initialize weights for all the attributes and all the k
   marray<double> PhitDisc(NoDiscEstimated, 0.0) ;
   marray<double> PmissDisc(NoDiscEstimated, 0.0) ;
   marray<double> PhitCont(NoContEstimated, 0.0) ;
   marray<double> PmissCont(NoContEstimated, 0.0) ;
 
   // data structure to hold nearest hits/misses
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)
   {
      distanceArray[iClss].create(noExInClass[iClss]) ;
      diffSorted[iClss].create(noExInClass[iClss]) ;
   }

   marray<double> avgCost(NoClasses+1, 0.0) ;
   for (i=1 ; i<=NoClasses ; i++) {
     for (j=1 ; j<=NoClasses ; j++)
         if (j != i) {
            avgCost[i] += fTree->CostMatrix(i, j) ;           
         }
     avgCost[i] /= (NoClasses - 1.0) ;
   }
   
   // normalization of contributions for hit/misses
   mmatrix<double> clNorm(NoClasses+1,NoClasses+1) ;
   for (i=1 ; i<=NoClasses ; i++)
     for (j=1 ; j<=NoClasses ; j++)
         if (i==j) clNorm(i,i) =  avgCost[i] ;
         else clNorm(i,j) =  fTree->CostMatrix(i,j)/(NoClasses - 1.0); // /  avgCost[i] ;
                            // probClass[j]* fTree->CostMatrix(i,j)/(expCost[i]*(1-probClass[i])) ;
   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   double distanceSum, normDistance, Adiff ;
   int current, neighbourIdx, cl, iAttr, currentClass ;
   
   marray<double> incContDiffA(NoContEstimated), incDiscDiffA(NoDiscEstimated) ;
       
   // prepare order of iterations by the expected cost of missclassification
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
  
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++) {
       current = sampleIdx[iterIdx] ;
       // initialize (optimization reasons)
       currentClass =  DiscValues(current, 0) ;
       // first we compute distances of  all other examples to current
       computeDistances(current) ;
       // compute distance factors
       prepareDistanceFactors(current, distanceType) ;

      for (cl=1 ; cl<=NoClasses ; cl++)
      {
         // compute sum of diffs
         incContDiffA.init(0.0) ;
         incDiscDiffA.init(0.0) ;
         distanceSum = 0.0 ;
         for (i=0 ; i < distanceArray[cl].filled() ; i++)
         {
            neighbourIdx = distanceArray[cl][i].value ;
            normDistance = distanceArray[cl][i].key ;
            distanceSum += normDistance ;
                 
            // adjust the weights for all the estimated attributes and values
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
            {
               idx = iAttr - contAttrFrom ;
               Adiff = ContDistance(neighbourIdx, iAttr) ;
               incContDiffA[idx] += Adiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
            {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscDiffA[idx] +=  Adiff * normDistance  ;
            }
         }
         if (cl == currentClass) // hit or miss
         {
            // hit
            // normalization of increments
            for (idx=0 ; idx < NoContEstimated ; idx++)
              if (incContDiffA[idx] > epsilon)
                 PhitCont[idx] += clNorm(cl, cl) * incContDiffA[idx]/distanceSum ;
            for (idx=0 ; idx < NoDiscEstimated ; idx++)
              if (incDiscDiffA[idx] > epsilon)
                PhitDisc[idx] += clNorm(cl, cl) * incDiscDiffA[idx]/distanceSum ;
          }
          else
          {
             // miss
             // normalization of increments
             for (idx=0 ; idx < NoContEstimated ; idx++)
               if (incContDiffA[idx] > epsilon)
                 PmissCont[idx] += clNorm(currentClass, cl) * incContDiffA[idx]/distanceSum ;
             for (idx=0 ; idx < NoDiscEstimated ; idx++)
               if (incDiscDiffA[idx] > epsilon)
                 PmissDisc[idx] += clNorm(currentClass, cl) * incDiscDiffA[idx]/distanceSum ;
          }
      }
   }  
   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
   {
      idx = iAttr - contAttrFrom ;
      ContEstimation[iAttr] = (PmissCont[idx] - PhitCont[idx])/double(NoIterations) ;
  }
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
   {
      idx = iAttr - discAttrFrom ;
      DiscEstimation[iAttr] = (PmissDisc[idx] - PhitDisc[idx])/double(NoIterations) ;
  }
}


// ***************************************************************************
//
//                       ReliefFexpC
//                       -------
//
//   contains two versions of ReliefFeCt, but different normalization
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//    expected cost of missclassification
// ***************************************************************************
void estimation::ReliefFexpC(int contAttrFrom, int contAttrTo,
                  int discAttrFrom, int discAttrTo, int distanceType)
{
   ContEstimation.init(contAttrFrom,contAttrTo,0.0) ;
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // prepare estimations arrays
   int NoContEstimated = contAttrTo - contAttrFrom ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
 
   // number of examples belonging to each of the classes
   marray<int> noExInClass(NoClasses+1) ;
   marray<double> probClass(NoClasses+1) ;
   noExInClass.init(0) ;
   probClass.init(0.0) ;
   int i, j, idx ;
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

   // compute estimations of class value probabilities with their 
   // relative frequencies
   for (idx=1 ; idx <= NoClasses ; idx++)
      probClass[idx] = probClass[idx] / wAll ;

   // initialize weights for all the attributes and all the k
   marray<double> PDisc(NoDiscEstimated, 0.0) ;
   marray<double> PCont(NoContEstimated, 0.0) ;
   
   // data structure to hold nearest hits/misses
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)
   {
      distanceArray[iClss].create(noExInClass[iClss]) ;
      diffSorted[iClss].create(noExInClass[iClss]) ;
   }

   // normalization of contribution of misses
   
   mmatrix<double> clNorm(NoClasses+1,NoClasses+1) ;
   double expCost ;
   for (i=1 ; i<=NoClasses ; i++) {
      expCost = 0.0 ; 
      for (j=1 ; j<=NoClasses ; j++) {
         if (j!=i) 
            expCost += probClass[j] * fTree->CostMatrix(i,j) ;
      }
      expCost /= (1.0 - probClass[i]) ;
      for (j=1 ; j<=NoClasses ; j++)
         if (j==i)
            clNorm(i,i) = - expCost  ;
         else clNorm(i,j) = probClass[j] * fTree->CostMatrix(i,j) / (1.0 - probClass[i]) ;
   }

   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   double distanceSum, normDistance, Adiff ;
   int current, neighbourIdx, cl, iAttr, currentClass ;
   
   marray<double> incContDiffA(NoContEstimated), incDiscDiffA(NoDiscEstimated) ;
       
   // prepare order of iterations, select them according to expected cost
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
   
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)
   {
       current = sampleIdx[iterIdx] ;
 
       // initialize (optimization reasons)
       currentClass =  DiscValues(current, 0) ;

       // first we compute distances of  all other examples to current
       computeDistances(current) ;

      // compute distance factors
      prepareDistanceFactors(current, distanceType) ;

      for (cl=1 ; cl<=NoClasses ; cl++)
      {
         // compute sum of diffs
         incContDiffA.init(0.0) ;
         incDiscDiffA.init(0.0) ;
         distanceSum = 0.0 ;
         for (i=0 ; i < distanceArray[cl].filled() ; i++)
         {
            neighbourIdx = distanceArray[cl][i].value ;
            normDistance = distanceArray[cl][i].key ;
            distanceSum += normDistance ;
                 
            // adjust the weights for all the estimated attributes and values
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
            {
               idx = iAttr - contAttrFrom ;
               Adiff = ContDistance(neighbourIdx, iAttr) ;
               incContDiffA[idx] += Adiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
            {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscDiffA[idx] +=  Adiff * normDistance  ;
            }
         }
 
         // normalization of increments for hits amd misses
         for (idx=0 ; idx < NoContEstimated ; idx++)
            if (incContDiffA[idx] > epsilon)
               PCont[idx] += clNorm(currentClass, cl) * incContDiffA[idx]/distanceSum ;
         for (idx=0 ; idx < NoDiscEstimated ; idx++)
            if (incDiscDiffA[idx] > epsilon)
              PDisc[idx] += clNorm(currentClass, cl) * incDiscDiffA[idx]/distanceSum ;
      }
   }  
   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
   {
      idx = iAttr - contAttrFrom ;
      ContEstimation[iAttr] = PCont[idx]/double(NoIterations) ;
  }
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
   {
      idx = iAttr - discAttrFrom ;
      DiscEstimation[iAttr] = PDisc[idx]/double(NoIterations) ;
  }
}

// ***************************************************************************
//
//                       ReliefFsmp
//                       -------
//
//   contains two versions of ReliefF:
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//
// ***************************************************************************
void estimation::ReliefFsmp(int contAttrFrom, int contAttrTo,
                  int discAttrFrom, int discAttrTo, int distanceType)
{

   ContEstimation.init(contAttrFrom,contAttrTo,0.0) ;
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // prepare estimations arrays
   int NoContEstimated = contAttrTo - contAttrFrom ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
  

   // number of examples belonging to each of the classes
   marray<int> noExInClass(NoClasses+1) ;
   marray<double> probClass(NoClasses+1) ;
   noExInClass.init(0) ;
   probClass.init(0.0) ;
   int i, idx ;
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

   // compute estimations of class value probabilities with their 
   // relative frequencies
   for (idx=1 ; idx <= NoClasses ; idx++)
      probClass[idx] = probClass[idx] / wAll ;

   // initialize weights for all the attributes and all the k
   marray<double> PhitDisc(NoDiscEstimated, 0.0) ;
   marray<double> PmissDisc(NoDiscEstimated, 0.0) ;
   marray<double> PhitCont(NoContEstimated, 0.0) ;
   marray<double> PmissCont(NoContEstimated, 0.0) ;
 
   // data structure to hold nearest hits/misses
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)   {
      distanceArray[iClss].create(noExInClass[iClss]) ;
      diffSorted[iClss].create(noExInClass[iClss]) ;
   }

   // normalization of contribution of misses
   mmatrix<double> clNorm(NoClasses+1,NoClasses+1) ;
   for (int j=1 ; j<=NoClasses ; j++)
     for (i=1 ; i<=NoClasses ; i++)
        clNorm.Set(j,i, probClass[j]/(1.0-probClass[i]) ) ;

   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   double distanceSum, normDistance, Adiff ;
   int current, neighbourIdx, cl, iAttr, currentClass ;
   
   marray<double> incContDiffA(NoContEstimated), incDiscDiffA(NoDiscEstimated) ;
       
   // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   stratifiedExpCostSample(sampleIdx, NoIterations, TrainSize, probClass, noExInClass) ;
   
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)
   {
       current = sampleIdx[iterIdx] ;
 
       // initialize (optimization reasons)
       
       currentClass =  DiscValues(current, 0) ;
      
        
      // first we compute distances of  all other examples to current
      computeDistances(current) ;

      // compute distance factors
      prepareDistanceFactors(current, distanceType) ;

      for (cl=1 ; cl<=NoClasses ; cl++)
      {
         // compute sum of diffs
         incContDiffA.init(0.0) ;
         incDiscDiffA.init(0.0) ;
         distanceSum = 0.0 ;
         for (i=0 ; i < distanceArray[cl].filled() ; i++)
         {
            neighbourIdx = distanceArray[cl][i].value ;
            normDistance = distanceArray[cl][i].key ;
            distanceSum += normDistance ;
                 
            // adjust the weights for all the estimated attributes and values
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
            {
               idx = iAttr - contAttrFrom ;
               Adiff = ContDistance(neighbourIdx, iAttr) ;
               incContDiffA[idx] += Adiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
            {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscDiffA[idx] +=  Adiff * normDistance  ;
            }
         }
         if (cl == currentClass) // hit or miss
         {
            // hit
            // normalization of increments
            for (idx=0 ; idx < NoContEstimated ; idx++)
              if (incContDiffA[idx] > epsilon)
                 PhitCont[idx] += incContDiffA[idx]/distanceSum ;
            for (idx=0 ; idx < NoDiscEstimated ; idx++)
              if (incDiscDiffA[idx] > epsilon)
                PhitDisc[idx] += incDiscDiffA[idx]/distanceSum ;
          }
          else
          {
             // miss
             // normalization of increments
             for (idx=0 ; idx < NoContEstimated ; idx++)
               if (incContDiffA[idx] > epsilon)
                 PmissCont[idx] += clNorm(cl, currentClass) * incContDiffA[idx]/distanceSum ;
             for (idx=0 ; idx < NoDiscEstimated ; idx++)
               if (incDiscDiffA[idx] > epsilon)
                 PmissDisc[idx] += clNorm(cl, currentClass) * incDiscDiffA[idx]/distanceSum ;
          }
      }
   }  
   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
   {
      idx = iAttr - contAttrFrom ;
      ContEstimation[iAttr] = (PmissCont[idx] - PhitCont[idx])/double(NoIterations) ;
      #ifdef DEBUG
      if (ContEstimation[iAttr] > 1.00001 || ContEstimation[iAttr] < -1.00001)
        error("estimation::ReliefF", "computed continuous weights are out of scope") ;
      #endif
   }
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
   {
      idx = iAttr - discAttrFrom ;
      DiscEstimation[iAttr] = (PmissDisc[idx] - PhitDisc[idx])/double(NoIterations) ;
      #ifdef DEBUG
      if (DiscEstimation[iAttr] > 1.00001 || DiscEstimation[iAttr] < -1.00001)
        error("estimation::ReliefF", "computed discrete weights are out of scope") ;
      #endif
   }
}


// ***************************************************************************
//
//               stratifiedExpCostSample
//                       -------
//
//   sampling stratified by expected misclassification cost
//                   
//
// ***************************************************************************
void estimation::stratifiedExpCostSample(marray<int> &sampleIdx, int sampleSize, int domainSize, marray<double> &probClass, marray<int> &noExInClass) 
{
   marray<double> expCost(NoClasses+1, 0.0) ;
   double expCostSum = 0.0 ;
   int i, j ;
   for (i=1 ; i<=NoClasses ; i++) {
     for (j=1 ; j<=NoClasses ; j++)
         if (j != i) {
           expCost[i] +=  probClass[j]* fTree->CostMatrix(i, j) ;
         }
     expCost[i] /= (1.0 - probClass[i]) ;
     expCostSum += probClass[i] * expCost[i] ;
   }
   // prepare order of iterations by the expected cost of missclassification
   marray<int> samplePrep(domainSize) ;
   int iterFromClass, times, sIdx, rndIdx, upper ;
   for (i=1 ; i <=NoClasses ; i++) {
       expCost[i] = probClass[i]*expCost[i]/ expCostSum ; // get probability
       iterFromClass = int(expCost[i] * sampleSize)  ;
       expCost[i] += expCost[i-1] ; // cumulative probability distribution
       for (j=0, sIdx = 0 ; j < domainSize ; j++)
           if (DiscValues(j,0) == i) {
              samplePrep[sIdx] = j ;
              sIdx++ ;
           }
       if (sIdx != noExInClass[i])
           error("estimation::stratifiedExpCostSample", "internal assumption invalid") ;
       times = int(iterFromClass / noExInClass[i]) ;
       // if number of iterations from class is larger than number of samples from class
       // we first multiply all of them for "times" times,
       // the rest are choosen randomly without replacements
       for (j=int(expCost[i-1]*NoIterations), sIdx = 0 ; 
           j < int(expCost[i-1]*sampleSize) + times * noExInClass[i]; j++) {
              sampleIdx[j] = samplePrep[sIdx % noExInClass[i]] ;
              sIdx++ ;
           }
       upper = noExInClass[i] ;
       for (j = int(expCost[i-1]*sampleSize) + times * noExInClass[i], sIdx = 0 ; 
           j < int(expCost[i]*NoIterations) ; j++) {
               rndIdx = randBetween(0, upper) ;
               sampleIdx[j] = samplePrep[rndIdx] ;
               samplePrep[rndIdx] = samplePrep[upper-1] ;
               upper -- ;
           }
   }
   while (j < sampleSize) { // beacuase of possible floating point rounding error
      sampleIdx[j]  =  randBetween(0, domainSize) ;
      j++ ;
   }
}

// ***************************************************************************
//
//                      gainRatioC
//                       -------
//
//   Gain ratio with cost informationt
//                   
//
// ***************************************************************************
void estimation::gainRatioC(int discAttrFrom, int discAttrTo)
{
     // prepare estimations arrays
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;
   marray<int> noExInClass(NoClasses+1, 0) ;

   // number of examples belonging to each of the classes
   int i,j;
   for (i=0 ; i < TrainSize ; i++)
      noExInClass[ DiscValues(i,0) ]++ ;
 
   // probabilities of the classes
   marray<double> pC(NoClasses+1, double(0)) ;
   for (i=1 ; i <= NoClasses ;i++)   
      pC[i] = double(noExInClass[i]) / TrainSize ;                 
   
   marray<double> eC(NoClasses+1, 0) ;
   double eCsum = 0.0 ;
   for (i=1 ; i<=NoClasses ; i++) {
     for (j=1 ; j<=NoClasses ; j++)
         if (j != i) 
           eC[i] +=  pC[j]* fTree->CostMatrix(i, j) ;
     eC[i] /= (1.0 - pC[i]) ;
     eCsum +=  pC[i] * eC[i] ;
   }
   double Ec = 0.0 ;  // entropy
   marray<double> pC1(NoClasses+1, 0) ;
   for (i=1 ; i <= NoClasses ;i++)   {
      pC1[i] = pC[i] * eC[i] / eCsum ;                 
      if (pC1[i] > 0 && pC[i] < 1.0)
         Ec -= pC1[i] * log2(pC1[i]) ;
      else error("estimation::gainRatioC","invalid probability") ;
   }
   
   double Eca, Ea, Hca, tempP ;
   int valIdx, classIdx, noOK ;
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
     if (noOK <= 0 )  {
        DiscEstimation[discIdx] = -1.0 ;
        continue ;
     }
      
     // computation of Informaion gain
     Eca = Ea = 0.0 ;
     for (valIdx = 1 ; valIdx <= discNoValues[discIdx] ; valIdx++)
     {  
        if (valNo[valIdx] > 0)
        {
           for (i=1 ; i <= NoClasses ;i++)   
             pC[i] = double(noClassAttrVal(i,valIdx))/valNo[valIdx] ;
           eC.init(0) ;
           eCsum = 0.0 ;
           for (i=1 ; i<=NoClasses ; i++) {
              for (j=1 ; j<=NoClasses ; j++)
                 if (j != i) 
                    eC[i] +=  pC[j]* fTree->CostMatrix(i, j) ;
               eC[i] /= (1.0 - pC[i]) ;
               eCsum +=  pC[i] * eC[i] ;
            }
           Hca = 0.0 ;
           pC1.init(0) ;
           for (i=1 ; i <= NoClasses ;i++)   {
              pC1[i] = pC[i] * eC[i] / eCsum ;                 
              if (pC1[i] > 0)
                Hca -= pC1[i] * log2(pC1[i]) ;
              if (pC1[i] <0.0 || pC1[i] >1.0)
                error("estimation::gainRatioC","invalid conditional probability") ;
           }
           Eca += double(valNo[valIdx])/TrainSize * Hca ;
        }
        if (valNo[valIdx] != noOK)
        {
           tempP = double(valNo[valIdx]) / double(noOK) ;
           Ea -= tempP * log2(tempP) ;
        }
     }
     if (Ea > 0.0)
       DiscEstimation[discIdx] = (Ec - Eca) / Ea ;
     else
       DiscEstimation[discIdx] = -1.0 ;
   }
}

// ***************************************************************************
//
//                      DKMc
//                     -------
//
//                DKM with cost information
//                   
//
// ***************************************************************************
void estimation::DKMc(int discAttrFrom, int discAttrTo)
{

   // prepare estimations arrays
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;
   // int NoDiscEstimated = discAttrTo - discAttrFrom ;
  
   // number of examples belonging to each of the classes
   marray<int> noExInClass(NoClasses+1, 0) ;
   int i, j;
   for (i=0 ; i < TrainSize ; i++)
      noExInClass[ DiscValues(i,0) ]++ ;
 
   // probabilities of the classes
   marray<double> pC(NoClasses+1, 0) ;
   for (i=1 ; i <= NoClasses ;i++)   
      pC[i] = double(noExInClass[i]) / TrainSize ;                 
   
   marray<double> eC(NoClasses+1, 0) ;
   double eCsum = 0.0 ;
   for (i=1 ; i<=NoClasses ; i++) {
     for (j=1 ; j<=NoClasses ; j++)
         if (j != i) 
           eC[i] +=  pC[j]* fTree->CostMatrix(i, j) ;
     eC[i] /= (1.0 - pC[i]) ;
     eCsum +=  pC[i] * eC[i] ;
   }
   double q = -1.0 ;
   marray<double> pC1(NoClasses+1, 0) ;
   for (i=1 ; i <= NoClasses ;i++)   {
      pC1[i] = pC[i] * eC[i] / eCsum ;     
      if (pC1[0] < 0.0 || pC1[i] > 1.0)
        error("estimation::DKMc","invalid probability") ;
      if (pC1[i] > q)
         q = pC1[i] ;
   }
   // probability of the 'majority' class
   double DKMprior ;
   if (q <= 0.0 || q >= 1.0)
   {
      DiscEstimation.init(discAttrFrom,discAttrTo,-1.0) ;
      return ;
   }
   else DKMprior = 2.0 * sqrt(q*(1.0-q)) ;

   double DKMpost, qCond ;
   int valIdx, noOK ;
   int classIdx, discIdx ;
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
      if (noOK <= 0 )     {
         DiscEstimation[discIdx] = -1.0 ;
         continue ;
      }

      // computation of DKM   
      DKMpost = 0.0 ;
      for (valIdx = 1 ; valIdx <= discNoValues[discIdx] ; valIdx++)  
        if (valNo[valIdx] > 0) {
           // probabilities of the classes
           for (i=1 ; i <= NoClasses ;i++)   
              pC[i] = double(noClassAttrVal(i, valIdx)) / valNo[valIdx] ;                 
   
           eC.init(0) ;
           eCsum = 0.0 ;
           for (i=1 ; i<=NoClasses ; i++) {
             for (j=1 ; j<=NoClasses ; j++)
                if (j != i) 
                    eC[i] +=  pC[j]* fTree->CostMatrix(i, j) ;
             eC[i] /= (1.0 - pC[i]) ;
             eCsum +=  pC[i] * eC[i] ;
           }
           qCond = -1.0 ;
           for (  i=1 ; i <= NoClasses ;i++)   {
             pC1[i] = pC[i] * eC[i] / eCsum ;                 
             if (pC1[i] < 0.0 || pC1[i] >1.0)
                error("estimation::DKMc","invalid conditional probability") ;
             if (pC1[i] > qCond)
                qCond = pC1[i] ;
           }
           if (qCond > 0.0 && qCond < 1.0)
             DKMpost += double(valNo[valIdx])/noOK * sqrt(qCond*(1.0-qCond)) ;
        }
     DiscEstimation[discIdx] = DKMprior-2*DKMpost ;
   }
}


// ***************************************************************************
//
//                       MDLsmp
//                       -------
//
//    MDL criterion with expected cost sampling
//
// ***************************************************************************
void estimation::MDLsmp(int discAttrFrom, int discAttrTo)
{

   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // number of examples belonging to each of the classes
   marray<int> noExInClass(NoClasses+1) ;
   marray<double> probClass(NoClasses+1) ;
   noExInClass.init(0) ;
   probClass.init(0.0) ;
   int i, idx ;
   for (i=0 ; i < TrainSize ; i++)  {
      noExInClass[ DiscValues(i,0) ]++ ;
      probClass[ DiscValues(i,0) ] += weight[i] ;
   }
   double wAll = 0.0 ;
   for (idx=1 ; idx <= NoClasses ; idx++)
      wAll +=  probClass[idx] ;
   // compute estimations of class value probabilities with their relative frequencies
   for (idx=1 ; idx <= NoClasses ; idx++)
      probClass[idx] = probClass[idx] / wAll ;
       
   // prepare order of iterations
   marray<int> sampleIdx(TrainSize);
   stratifiedExpCostSample(sampleIdx, NoIterations, TrainSize, probClass, noExInClass) ;

   // now do the real MDL: on sampled data
   noExInClass.init(0) ;
   // number of examples belonging to each of the classes
   int classIdx;
   for (i=0 ; i < TrainSize ; i++)
      noExInClass[ DiscValues(sampleIdx[i],0) ]++ ;
 
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
         noClassAttrVal(DiscValues(sampleIdx[i], 0), DiscValues(sampleIdx[i], discIdx) ) ++ ;

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

