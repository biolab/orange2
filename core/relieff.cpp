#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>


//#define PRINT_EACH_K
//#define PRINT_EACH_ITERATION


#include "estimator.h"                
#include "general.h"
#include "contain.h"
#include "utils.h"
#include "options.h"

extern Options *opt ;


// ***************************************************************************
//
//                         constructor
//
// ***************************************************************************
estimation::estimation(featureTree *fTreeParent, marray<int> &inDTrain,
                 marray<double> &inpDTrain, int inTrainSize)
{
   fTree = fTreeParent ;
   initialize(inDTrain, inpDTrain, inTrainSize) ;
}

void estimation::initialize(marray<int> &inDTrain, marray<double> &inpDTrain, 
							int inTrainSize)  {

   //-------------------------------------------------------------
   // copy essential values
   //-------------------------------------------------------------
   currentContSize = NoContinuous = fTree->NoContinuous ;
   currentDiscSize = NoDiscrete = fTree->NoDiscrete ;

   //-------------------------------------------------------------
   // select the training examples for ReliefF's estimation
   //-------------------------------------------------------------
   marray<int> DTrain ;
   int i, j, k ;
   // will we use all exmples or just a subsample
   if (inTrainSize <= opt->attrEvaluationInstances || opt->attrEvaluationInstances == 0)
   {
      TrainSize = inTrainSize ;
      DTrain.copy(inDTrain) ;
      weight.copy(inpDTrain) ;
   }
   else
   {
       //-------------------------------------------------------------
       // randomly select exactly attrEvaluationInstances without repetitions
       //-------------------------------------------------------------

       marray<int> selected(inTrainSize) ;
       for (i=0 ; i < inTrainSize ; i++)
           selected[i] = i ;
       selected.setFilled(inTrainSize) ;

       TrainSize = opt->attrEvaluationInstances ;
       DTrain.create(TrainSize) ;
       weight.create(TrainSize) ;

       for (i=0 ; i < TrainSize ; i++)  {
           j = randBetween(0, selected.filled()) ; 
           DTrain[i] = inDTrain[selected[j]] ;
           weight[i] = inpDTrain[selected[j]]  ;
           selected[j] = selected[selected.filled()-1] ;
           selected.setFilled(selected.filled()-1) ;
       }
   }

   //-------------------------------------------------------------
   //  copy discrete and continuous data
   //-------------------------------------------------------------
   DiscValues.create(TrainSize, NoDiscrete) ;
   for (i=0 ; i < NoDiscrete ; i++)
     for (j=0 ; j < TrainSize ; j++)
         DiscValues.Set(j,i,fTree->DiscData(DTrain[j],i) );

   ContValues.create(TrainSize, NoContinuous) ;
   for (i=0 ; i < NoContinuous ; i++)
      for (j=0 ; j < TrainSize ; j++)
           ContValues.Set(j,i,fTree->ContData(DTrain[j],i) );

   //-------------------------------------------------------------
   // create estimation arrays
   //-------------------------------------------------------------
   DiscEstimation.create(NoDiscrete, -2.0) ;
   ContEstimation.create(NoContinuous, -2.0) ;
   //-------------------------------------------------------------
   // create distance matrix
   //-------------------------------------------------------------
   ContDistance.create(TrainSize, NoContinuous) ;
   DiscDistance.create(TrainSize, NoDiscrete) ;


   //-------------------------------------------------------------
   // set number of iterations in main reliefF loop
   //-------------------------------------------------------------
   if (opt->ReliefIterations == 0 ||
       opt->ReliefIterations > TrainSize)
       NoIterations = TrainSize ;
   else if (opt->ReliefIterations == -1)
       NoIterations = (int)log(double(TrainSize)) ;
   else if (opt->ReliefIterations == -2)
       NoIterations = (int)sqrt(double(TrainSize)) ;
   else
      NoIterations = opt->ReliefIterations ;


   //-------------------------------------------------------------
   // set slopes and distances for ramp function of continuous attributes and class
   //-------------------------------------------------------------
#ifdef RAMP_FUNCTION
   DifferentDistance.create(NoContinuous) ;
   EqualDistance.create(NoContinuous) ;
   CAslope.create(NoContinuous) ;
   for (i=0 ; i < NoContinuous ; i++)
   {
     DifferentDistance[i] = fTree->AttrDesc[fTree->ContIdx[i]].DifferentDistance ;
     EqualDistance[i] = fTree->AttrDesc[fTree->ContIdx[i]].EqualDistance ;
     if (DifferentDistance[i] != EqualDistance[i])
         CAslope[i] = double(1.0)/(DifferentDistance[i] - EqualDistance[i]) ;
     else
        CAslope[i] = FLT_MAX ;
   }
#endif

   //-------------------------------------------------------------
   //  set number of values for discrete
   //-------------------------------------------------------------
   discNoValues.create(NoDiscrete) ;
   for (i=0 ; i < NoDiscrete ; i++)
      discNoValues[i] = fTree->AttrDesc[fTree->DiscIdx[i]].NoValues  ;

   NoClasses = discNoValues[0] ;

   //-------------------------------------------------------------
   // compute probabilities for discrete missing values (knowing the class value)
   //-------------------------------------------------------------

   double denominator, valueProb ;

   NAdiscValue.create(NoClasses+1, NoDiscrete) ;

   for (i=1 ; i < NoDiscrete ; i++)
      for (j=1 ; j <= NoClasses ; j++)
          NAdiscValue(j,i).create(discNoValues[i] +1, 0.0) ;

   for (i=1; i < NoDiscrete ; i++)
     for (j=0 ; j < TrainSize ; j++)
        NAdiscValue(DiscValues(j,0), i)[DiscValues(j,i)] += 1.0 ;

   for (i=1 ; i < NoDiscrete ; i++)
   {
      for (k=1 ; k <= NoClasses ; k++)
      {
         // denominator initialy equals Laplacian correction 
         denominator = discNoValues[i]  ; 
         for (j=1; j < NAdiscValue(k,i).len() ; j++)
            denominator += NAdiscValue(k,i)[j] ;
           
         NAdiscValue(k,i)[0] = 0.0 ; //initially for both missing
         for (j=1; j < NAdiscValue(k,i).len() ; j++)
         {
            valueProb = (NAdiscValue(k,i)[j]+double(1.0))/denominator ;
            NAdiscValue(k,i)[j] =  double(1.0) - valueProb ; // diff =  1 - prob
            // both are missing
            NAdiscValue(k,i)[0] += valueProb * valueProb  ;
         }
         NAdiscValue(k,i)[0] = double(1.0) - NAdiscValue(k,i)[0] ; 
      }
   }

   //-------------------------------------------------------------
   //  continuous attribute missing values
   //   it would be better to use density estimation with kernel functions
   //-------------------------------------------------------------

   minValue.copy(fTree->minValue) ;
   maxValue.copy(fTree->maxValue) ;
   valueInterval.copy(fTree->valueInterval) ;
   step.create(NoContinuous) ;
   NAcontValue.create(NoClasses+1,NoContinuous) ;

   if (TrainSize/constNAdiscretizationIntervals < constAverageExamplesPerInterval )
      noNAdiscretizationIntervals = Mmax(2,int(TrainSize/constAverageExamplesPerInterval)) ;
   else
      noNAdiscretizationIntervals = constNAdiscretizationIntervals ;

   for (i=0; i < NoContinuous ; i++)
   {
      step[i] =  valueInterval[i]/noNAdiscretizationIntervals*double(1.000001) ; // 1.000001 - to avoid overflows due to numerical aproximation
      for (j=1 ; j <= NoClasses ; j++) 
        NAcontValue(j,i).create(noNAdiscretizationIntervals +1, 0.0) ;
   }

   for (i=0; i < NoContinuous ; i++)
     for (j=0 ; j < TrainSize ; j++)
       if (ContValues(j,i) != NAcont)
         NAcontValue( DiscValues(j,0), i )[int((ContValues(j,i)-minValue[i])/step[i])+1] += 1 ;

   for (i=0 ; i < NoContinuous ; i++)
   {
      for (k=1 ; k <= NoClasses ; k++)
      {
         // denominator initialy equals Laplacian correction 
         denominator = noNAdiscretizationIntervals ; ;
         for (j=1; j < NAcontValue(k,i).len() ; j++)
             denominator += NAcontValue(k,i)[j] ;
      
         NAcontValue(k,i)[0] = 0.0 ;  // for both missing
         for (j=1; j < NAcontValue(k,i).len() ; j++)
         {
            valueProb = (NAcontValue(k,i)[j]+double(1.0))/denominator ;
            NAcontValue(k,i)[j] =  double(1.0) - valueProb ;
            // both are missing - computing same value probability
            NAcontValue(k,i)[0] += valueProb * valueProb  ;
         }
         NAcontValue(k,i)[0] = double(1.0) - NAcontValue(k,i)[0] ;
      }
   }
   
   //-------------------------------------------------------------
   //  set k nearest with distance density and standard deviation for distance density
   //-------------------------------------------------------------
   if (opt->kNearestEqual <= 0)
     kNearestEqual = TrainSize-1 ;
   else
     kNearestEqual = Mmin(opt->kNearestEqual, TrainSize-1) ;

   if (opt->kNearestExpRank <= 0)
     kDensity = TrainSize - 1 ;
   else
     kDensity = Mmin(opt->kNearestExpRank, TrainSize-1) ;


   //-------------------------------------------------------------
   // variance of distance density
   //-------------------------------------------------------------
   varianceDistanceDensity = sqr(opt->quotientExpRankDistance) ;

   //-------------------------------------------------------------
   // structure for sorting/selecting cases by distance
   //-------------------------------------------------------------
   distanceArray.create(NoClasses+1) ;
   diffSorted.create(NoClasses+1) ;
 
}


// ***************************************************************************
//
//                       ReliefF
//                       -------
//
//   contains two versions of ReliefF:
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//
// ***************************************************************************
// ***************************************************************************
void estimation::ReliefF(int contAttrFrom, int contAttrTo,
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
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)
   {
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
       
   #if defined(PRINT_EACH_ITERATION)
     char path[MaxPath] ;
     int iPrint, contCount=0, discCount=0 ; 
     FILE *fileRelief ;
     sprintf(path,"%s%s.%02dei",fTree->resultsDirectory, fTree->domainName,fTree->currentSplitIdx) ; // estimation of weights at each iteration
     if ((fileRelief = fopen(path,"w"))==NULL)
     {
        error("estimation::ReliefF cannot open file for writting weights of each iteration: ", path)  ;
     }
     else {
        fprintf(fileRelief, "\nRelief weights changing with number of iterations\n") ;
        fTree->printEstimationHead(fileRelief) ;
     }

   #endif 

   // prepare order of iterations
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
      #if defined(PRINT_EACH_ITERATION)
        fprintf(fileRelief, "%18d,", iterIdx+1) ;
        contCount = discCount = 0 ; 
        for (iPrint=1 ; iPrint <= fTree->NoAttr; iPrint++)
        if (fTree->AttrDesc[iPrint].continuous)
        {
          fprintf(fileRelief, "%10.5f, ", (PmissCont[contCount] - PhitCont[contCount])/double(iterIdx+1)) ;
          contCount++ ;
        }
        else {
          fprintf(fileRelief, "%10.5f, ", (PmissDisc[discCount] - PhitDisc[discCount])/double(iterIdx+1)) ;
          discCount++ ;
        }
        fprintf(fileRelief, "\n") ;
      #endif 
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
   #if defined(PRINT_EACH_ITERATION)
     fclose(fileRelief) ;
   #endif 
 
}




// ***************************************************************************
//
//                    computeDistances
// difference between two training instances in attribute space
//
// ***************************************************************************
void estimation::computeDistances(int Example)
{
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
          DiscDistance.Set(j, i, DAdiff(i,Example,j)) ;
      }
   }
}


// ***************************************************************************
//
//                    CaseDistance
// difference between two training instances in attribute space
//
// ***************************************************************************
double estimation::CaseDistance(int I1)
{
   double Distance = 0.0;

   int i ;
   for (i=1 ; i < NoDiscrete ; i++)
      Distance += DiscDistance(I1,i) ;

   for (i=0; i<NoContinuous; i++)
      Distance += ContDistance(I1,i) ;

   return  Distance ;
}


// ***************************************************************************
//
//                    CARamp
//          ramp function of continuous attribute (or class)
//
// ***************************************************************************
#if defined(RAMP_FUNCTION)
inline double estimation::CARamp(int AttrIdx, double distance)
{
  if (distance >= DifferentDistance[AttrIdx])
     return 1.0 ;
  if (distance <= EqualDistance[AttrIdx])
     return 0.0 ;

  return  (distance - EqualDistance[AttrIdx]) * CAslope[AttrIdx] ;
}
#endif

// ***************************************************************************
//
//                   CAdiff
//              diff function for continuous attribute
//
// ***************************************************************************
double estimation::CAdiff(int AttrIdx, int I1, int I2)
{
   double cV1 = ContValues(I1, AttrIdx) ;
   double cV2 = ContValues(I2, AttrIdx) ;
   if (cV1 == NAcont)
      return NAcontDiff(AttrIdx,DiscValues(I1,0), cV2) ;
    else
      if (cV2 == NAcont)
        return NAcontDiff(AttrIdx, DiscValues(I2,0), cV1) ;
       else
         #ifdef RAMP_FUNCTION
           return CARamp(AttrIdx, fabs(cV2 - cV1) ) ;
        #else
           return  fabs(cV2 - cV1) / valueInterval[AttrIdx] ;
        #endif
}



// ***************************************************************************
//
//                   DAdiff
//              diff function of discrete attribute
//
// ***************************************************************************
inline double estimation::DAdiff(int AttrIdx, int I1, int I2)
{

  // we assume that missing value has value 0
  int dV1 = DiscValues(I1, AttrIdx) ;
  int dV2 = DiscValues(I2, AttrIdx) ;
  if (dV1 == NAdisc)
     return NAdiscValue(DiscValues(I1,0),AttrIdx)[int(dV2)] ;
  else
    if (dV2 == NAdisc)
      return NAdiscValue(DiscValues(I2,0),AttrIdx)[int(dV1)] ;
     else
       if (dV1 == dV2)
         return  0.0 ;
       else
         return 1.0 ;
}

// ***************************************************************************
//
//                   NAcontDiff
//         diff function for missing values at continuous attribute
//
// ***************************************************************************
double estimation::NAcontDiff(int AttrIdx, int ClassValue, double Value)
{
   if (Value == NAcont)
      return NAcontValue(ClassValue, AttrIdx)[0] ;

   return NAcontValue(ClassValue, AttrIdx)[int((Value-minValue[AttrIdx])/step[AttrIdx]) +1] ;
}




// ***************************************************************************
//
//                          prepareDistanceFactors
// computation of distance probability weight factors for given example
//
// ***************************************************************************
void estimation::prepareDistanceFactors(int current, int distanceType)
{

// we use only original attributes to obtain distance in attribute space

   int kSelected = 0 ;
   switch (distanceType)
   {
      case estReliefFkEqual:
              kSelected = kNearestEqual ;
              break ;

      case estReliefFexpRank: 
      case estReliefFdistance:
      case estReliefFsqrDistance:
      case estReliefFexpC:
      case estReliefFavgC:
      case estReliefFpe:
      case estReliefFpa:
      case estReliefFsmp:
              kSelected = kDensity ;
              break ;

      case estReliefFbestK:
              kSelected = TrainSize ;  // we have to consider all neighbours
              break ;


      default: error("estimation::prepareDistanceFactors","invalid distance type") ;
   }

   int i, cl ;
   sortRec tempSort ;

   for (cl = 1 ; cl <= NoClasses; cl++)
   {
      // empty data structures
      distanceArray[cl].clear() ;
      diffSorted[cl].clear() ;
   }

   // distances in attributes space
   for (i=0 ; i < TrainSize; i++)
   {
      if (i==current)  // we skip current example
         continue ;
      tempSort.key =  CaseDistance(i) ;
      tempSort.value = i ;
      diffSorted[DiscValues(i,0)].addEnd(tempSort) ;
   }

   // sort examples 
   for (cl=1 ; cl <= NoClasses ; cl++)
   {
      // we sort groups of examples of the same class according to
      // ascending distance from current
      if (diffSorted[cl].filled() > 1)
         diffSorted[cl].sortKdsc(Mmin(kSelected, diffSorted[cl].filled())) ;
   }

   int upper, idx ;
   double factor ;
   // depending on tpe of distance, copy the nearest cases
   // and their distance factors into resulting array
   switch (distanceType)
   {
        
        case estReliefFkEqual: 
        case estReliefFbestK:
          {
            for (cl=1; cl <= NoClasses ; cl++)
            {
               idx =  diffSorted[cl].filled() -1;
               upper = Mmin(kSelected, diffSorted[cl].filled()) ;
               for (i=0 ; i < upper ; i++)
               {
                  distanceArray[cl][i].value = diffSorted[cl][idx].value ;
                  idx -- ;
                  distanceArray[cl][i].key = 1.0  ;
               }
               distanceArray[cl].setFilled(upper) ;
            }
          }
          break ;
        case estReliefFexpRank: 
        case estReliefFexpC:
        case estReliefFavgC:
        case estReliefFpe:
        case estReliefFpa:
        case estReliefFsmp:
          {
            for (cl=1; cl <= NoClasses ; cl++)
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
               for (i=1 ; i < upper ; i++)
               {
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
                  if (diffSorted[cl][i].key > 0.0)
                  {
                     if (diffSorted[cl][i].key < minNonZero)
                        minNonZero = diffSorted[cl][i].key ;
                     break;
                  }
            if (minNonZero == FLT_MAX)
               minNonZero = 1.0 ;

            for (cl=1; cl <= NoClasses ; cl++)
            {
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
        case estReliefFsqrDistance:
          {
            double minNonZero = FLT_MAX ; // minimal non zero distance
            for (cl=1; cl <= NoClasses ; cl++)
               for (i= diffSorted[cl].filled() -1 ; i >= 0 ; i--)
                  if (diffSorted[cl][i].key > 0.0)
                  {
                     if (diffSorted[cl][i].key < minNonZero)
                        minNonZero = diffSorted[cl][i].key ;
                     break;
                  }
            if (minNonZero == FLT_MAX)
               minNonZero = 1.0 ;

            for (cl=1; cl <= NoClasses ; cl++)
            {
               idx =  diffSorted[cl].filled() -1;
               upper = Mmin(kSelected, diffSorted[cl].filled()) ;
               for (i=0 ; i < upper ; i++)
               {
                  if (diffSorted[cl][idx].key > 0)
                     factor = 1.0 / sqr(diffSorted[cl][idx].key) ;
                  else 
                     factor = 2.0 / sqr(minNonZero) ;
                  distanceArray[cl][i].value = diffSorted[cl][idx].value ;
                  distanceArray[cl][i].key = factor  ;
                  idx -- ;
               }
               distanceArray[cl].setFilled(upper) ;
            }
          }
          break ;
        default: error("estimation::prepareDistanceFactors","invalid distanceType detected") ;
   }
}

  

// ***************************************************************************
//
//                       ReliefFbestK
// 
//                       ------------ 
//   contains the version of ReliefF:
//   - with best estimate of all possible k nearest 
//                   
//
// ***************************************************************************
void estimation::ReliefFbestK(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, int distanceType)
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
   mmatrix<double> PhitDisc(maxK, NoDiscEstimated, 0.0) ;
   mmatrix<double> PmissDisc(maxK, NoDiscEstimated, 0.0) ;
   mmatrix<double> PhitCont(maxK, NoContEstimated, 0.0) ;
   mmatrix<double> PmissCont(maxK, NoContEstimated, 0.0) ;
 
   // data structure to hold nearest hits/misses
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)
   {
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
   marray<double> contCorrection(NoContEstimated), discCorrection(NoDiscEstimated) ;

   // prepare order of iterations
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
         distanceSum = 0.0 ;
         discCorrection.init(0.0) ;
         contCorrection.init(0.0) ;

         if (cl == currentClass)
         {
            // hit
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
                contCorrection[idx] += Adiff * normDistance  ;
                PhitCont(i, idx) += contCorrection[idx]/distanceSum;
              }
              for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
              {
                idx = iAttr - discAttrFrom ;
                Adiff = DiscDistance(neighbourIdx, iAttr) ;
                discCorrection[idx] += Adiff * normDistance  ;
                PhitDisc(i, idx) += discCorrection[idx]/distanceSum;

              }
            }
            if (i > 0)
              while (i < maxK)
              {
                for (idx=0 ; idx < NoContEstimated ; idx ++)
                  PhitCont(i,idx) += contCorrection[idx]/distanceSum ;
                for (idx=0 ; idx < NoDiscEstimated ; idx ++)
                  PhitDisc(i,idx) += discCorrection[idx]/distanceSum ;
                i++ ;
              }

         }
         else
         {

            for (i=0 ; i < distanceArray[cl].filled() ; i++)
            {
              neighbourIdx = distanceArray[cl][i].value ;
              normDistance = distanceArray[cl][i].key ;
              distanceSum += normDistance ;
                 
              // adjust the weights for all the aestimated attributes and values
              for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
              {
                idx = iAttr - contAttrFrom ;
                Adiff = ContDistance(neighbourIdx, iAttr) ;
                contCorrection[idx] += clNorm(cl, currentClass) * Adiff * normDistance  ;
                PmissCont(i, idx) += contCorrection[idx]/distanceSum;
              }
              for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
              {
                idx = iAttr - discAttrFrom ;
                Adiff = DiscDistance(neighbourIdx, iAttr) ;
                discCorrection[idx] += clNorm(cl, currentClass) * Adiff * normDistance  ;
                PmissDisc(i, idx) += discCorrection[idx]/distanceSum;
              }
            }
            if (i >0)
               while (i < maxK)
               {
                 for (idx=0 ; idx < NoContEstimated ; idx ++)
                   PmissCont(i,idx) += contCorrection[idx]/distanceSum ;
                 for (idx=0 ; idx < NoDiscEstimated ; idx ++)
                   PmissDisc(i,idx) += discCorrection[idx]/distanceSum ;
                 i++ ;
               }

         }
      }
   }  
   
   double bestEst, est ;
   int k ;

   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
   {
      idx = iAttr - contAttrFrom ;
      bestEst = (PmissCont(0,idx) - PhitCont(0,idx))/double(NoIterations) ;
      for (k=1 ; k < maxK ; k++)
      {
         est = (PmissCont(k,idx) - PhitCont(k,idx))/double(NoIterations) ;        
         if (est > bestEst)
             bestEst = est ;
      }
      ContEstimation[iAttr] = bestEst ;
      #ifdef DEBUG
      if (ContEstimation[iAttr] > 1.00001 || ContEstimation[iAttr] < -1.00001)
        error("estimation::ReliefF", "computed continuous weights are out of scope") ;
      #endif
   }
   for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
   {
      idx = iAttr - discAttrFrom ;
      bestEst = (PmissDisc(0,idx) - PhitDisc(0,idx))/double(NoIterations) ;
      for (k=1 ; k < maxK ; k++)
      {
         est = (PmissDisc(k,idx) - PhitDisc(k,idx))/double(NoIterations) ;        
         if (est > bestEst)
             bestEst = est ;
      }
      DiscEstimation[iAttr] = bestEst ;
      #ifdef DEBUG
      if (DiscEstimation[iAttr] > 1.00001 || DiscEstimation[iAttr] < -1.00001)
        error("estimation::ReliefF", "computed discrete weights are out of scope") ;
      #endif
   }

   
   #if defined(PRINT_EACH_K)
     char path[MaxPath] ;
     FILE *fileRelief ;
     sprintf(path,"%s%s.%02dek",fTree->resultsDirectory, fTree->domainName,fTree->currentSplitIdx) ; // estimation of weights for each k
     if ((fileRelief = fopen(path,"w"))==NULL)
     {
        error("estimation::ReliefFbestK cannot open file for writting estimations for each k: ", path)  ;
     }
     fprintf(fileRelief, "\nReliefF weights changing with k nearest neighbours\n") ;
     fTree->printEstimationHead(fileRelief) ;

     int contCount,discCount; 
     for (k=0 ; k < maxK ; k++)
     {
       fprintf(fileRelief, "%18d,",k+1) ;  
       contCount = discCount = 0 ;
       for (i=1 ; i <= fTree->NoAttr; i++)
       if (fTree->AttrDesc[i].continuous)
       {
          fprintf(fileRelief, "%10.5f, ", (PmissCont(k, contCount) - PhitCont(k, contCount))/double(NoIterations)) ;
          contCount++ ;
       }
       else {
         fprintf(fileRelief, "%10.5f, ", (PmissDisc(k, discCount) - PhitDisc(k, discCount))/double(NoIterations)) ;
         discCount++ ;
       }
       fprintf(fileRelief, "\n") ;
     }
     fclose(fileRelief) ;
   #endif


}


// ***************************************************************************
//
//                       Relief
//                       -------
//
//   original (Kira nad Rendell) Relief
// ***************************************************************************
void estimation::Relief(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo)
{

   ContEstimation.init(contAttrFrom,contAttrTo,0.0) ;
   DiscEstimation.init(discAttrFrom,discAttrTo,0.0) ;

   // prepare estimations arrays
   int NoContEstimated = contAttrTo - contAttrFrom ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
  
   // initialize weights for all the attributes and all the k
   marray<double> PhitDisc(NoDiscEstimated, 0.0) ;
   marray<double> PmissDisc(NoDiscEstimated, 0.0) ;
   marray<double> PhitCont(NoContEstimated, 0.0) ;
   marray<double> PmissCont(NoContEstimated, 0.0) ;
 
   // we have to compute distances up to the folowing attributes
   discUpper = Mmax(NoDiscrete, discAttrTo) ;
   contUpper = Mmax(NoContinuous, contAttrTo) ;

   int current, idx, iAttr, hit, miss ;
   
   // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
  
   // main ReliefF loop
   for (int iterIdx=0 ; iterIdx < NoIterations ; iterIdx++)
   {
   
      current = sampleIdx[iterIdx] ;
  
	   // first we compute distances of  all other examples to current
      computeDistances(current) ;

      // compute distance factors
      findHitMiss(current, hit, miss) ;


      // adjust the weights for all the estimated attributes and values
      for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
      {
           idx = iAttr - contAttrFrom ;
           PhitCont[idx] += ContDistance(hit, iAttr) ;
           PmissCont[idx] += ContDistance(miss, iAttr) ; 
      }
      for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
      {
         idx = iAttr - discAttrFrom ;
         PhitDisc[idx] += DiscDistance(hit, iAttr) ;
         PmissDisc[idx] += DiscDistance(miss, iAttr) ; 
      
      }
   }  
   for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
   {
      idx = iAttr - contAttrFrom ;
      ContEstimation[iAttr] = (PmissCont[idx] - PhitCont[idx])/double(NoIterations) ;
      #ifdef DEBUG
      if (ContEstimation[iAttr] > 1.00001 || ContEstimation[iAttr] < -1.00001)
        error("estimation::Relief", "computed continuous weights are out of scope") ;
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
//                          findHitMiss
// find two nearest neighbors of current: hit and miss
//
// ***************************************************************************
void estimation::findHitMiss(int current, int &hit, int &miss)
{

   // we use only original attributes to obtain distance in attribute space

   double hitDistance = FLT_MAX, missDistance = FLT_MAX, distance ;
   
   for (int i=0 ; i < TrainSize; i++)
   {
      if (i==current)  // we skip current example
         continue ;
      
      distance = CaseDistance(i) ;
      if (DiscValues(current, 0) == DiscValues(i, 0)) // hit
      {
         if (distance < hitDistance)
         {
            hitDistance = distance ;
            hit = i ;
         }
      }
      else {  // miss
         if (distance < missDistance)
         {
            missDistance = distance ;
            miss = i ;
         }
      }
   }
}

  

// ***************************************************************************
//
//                       ReliefFmerit
//                       -------
//
//   contains two versions of ReliefF with merit:
//   1. with k nearest with equal influence
//   2. with k nearest with exponentially decreasing influence
//                   
//
// ***************************************************************************
void estimation::ReliefFmerit(int contAttrFrom, int contAttrTo,
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
   for (int iClss = 1 ; iClss <= NoClasses; iClss++)
   {
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

   double distanceSum, normDistance, Adiff, sumAdiff ;
   int current, neighbourIdx, cl, iAttr, currentClass ;
   
   marray<double> incContDiffA(NoContEstimated), incDiscDiffA(NoDiscEstimated) ;
       
    // prepare order of iterations
   marray<int> sampleIdx(NoIterations);
   randomizedSample(sampleIdx, NoIterations, TrainSize) ;
  
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
                 
            sumAdiff = 0.0 ;
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
               sumAdiff += ContDistance(neighbourIdx, iAttr) ;
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
               sumAdiff += DiscDistance(neighbourIdx, iAttr) ;

            // adjust the weights for all the estimated attributes and values
            for (iAttr=contAttrFrom ; iAttr < contAttrTo ; iAttr ++)
            {
               idx = iAttr - contAttrFrom ;
               Adiff = ContDistance(neighbourIdx, iAttr) ;
               incContDiffA[idx] += Adiff/sumAdiff * normDistance ;
            }
            for (iAttr=discAttrFrom ; iAttr < discAttrTo ; iAttr ++)
            {
               idx = iAttr - discAttrFrom ;
               Adiff = DiscDistance(neighbourIdx, iAttr) ;
               incDiscDiffA[idx] +=  Adiff/sumAdiff * normDistance  ;
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

