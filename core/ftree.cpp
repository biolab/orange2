/********************************************************************
*
*   Name:              modul regtree (regression  tree)
*
*   Description:  builds regression trees
*
*********************************************************************/


#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "error.h"
#include "ftree.h"
#include "utils.h"
#include "estimator.h"
#include "mathutil.h"
#include "constrct.h"
#include "options.h"

extern Options *opt ;

//const double epsilon = 1e-7 ;   // computational error


//************************************************************
//
//                    featureTree
//
//                    constructor
//
//************************************************************
featureTree::featureTree()
{
}

//************************************************************
//
//                      ~featureTree
//
//                        destructor
//
//************************************************************
featureTree::~featureTree()
{
}

//************************************************************
//
//                   constructTree
//
//       the main procedure for tree construction,
//              prepares data, calls tree builder
//
//************************************************************
int featureTree::constructTree()    //--//
{
   if (state<data && !readProblem())   //--//
      return 0;

   NoAttr = NoOriginalAttr ;
   // prepare training data
   marray<int> DTrain ;
   marray<double> pDTrain ;
   int TrainSize ;

   int i ;

   DTrain.create(NoTeachCases) ;
   pDTrain.create(NoTeachCases, 1.0) ;
   rootDTrain.create(NoTeachCases) ;
   for (i = 0 ; i < NoTeachCases ; i++)
   {
       rootDTrain[i] = DTrain[i] = DTeach[i];
   }
   rootTrainSize = TrainSize = NoTeachCases ;
   rootWeight = TrainSize ;

 
   // prepare the cache for the attributes
   int cachedNodes = int(pow(2, opt->constructionDepth)-0.5) ;
   if (cachedNodes > 1000 || cachedNodes < 0)
      cachedNodes = 1000 ; 
   CachedConstructs.create(opt->noCachedInNode * cachedNodes) ;

   // primary and secondary estimates
   //primaryEstimate.create(NoAttr+1, 0.0) ;
   //secondaryEstimate.create(NoAttr+1, 0.0) ;

   // call tree building algorithm
   destroy(root) ;
   root = 0 ;
   root = buildTree(DTrain,pDTrain,TrainSize, 1) ;

   if (root)   {
       switch (opt->selectedPruner)       {
       case 0: break;
       case 1: mdlBottomUpPrune() ;
               break ;
       case 2: mPrune() ;
               break ;
       }
      state = tree;
      return 1 ;
   }
   else
   {
      error("Tree construction unsuccessful.","");
      return 0 ;
   }
}


//************************************************************
//
//                 buildTree
//                 ---------
//
//    builds featured tree with constructive induction ;
//             recursive TDIDT algorithm
//
//************************************************************
binnode* featureTree::buildTree(marray<int> &DTrain, marray<double> &pDTrain, 
                                   int TrainSize, int currentDepth)
{
   binnode* Node = new binnode ;
   CurrentNode = Node ;
   CurrentTrainSize = TrainSize ;
   CurrentExamples = &DTrain ;
   Node->DTrain.copy(DTrain) ;
   Node->DTrain.setFilled(TrainSize) ;
   Node->weight = 0.0 ;
   Node->Classify.create(NoClasses+1, 0.0) ;
   int i, j ;

   // compute class distribution and weight of a node
   for (i=0 ; i < TrainSize ; i++)
   {
      Node->weight += pDTrain[i] ;
      Node->Classify[DiscData(DTrain[i],0)] += pDTrain[i] ;
   }
   
   //-------------------------------------------------------------
   // compute most probable discrete values used instead of missing values
   //-------------------------------------------------------------
   Node->NAdiscValue.create(NoDiscrete) ;
   marray<marray<double> > NAdiscCounter(NoDiscrete) ;

   for (i=0 ; i < NoDiscrete ; i++)
      NAdiscCounter[i].create(AttrDesc[DiscIdx[i]].NoValues +1, 0.0) ;

   for (i=0; i < NoDiscrete ; i++)
     for (j=0 ; j < TrainSize ; j++)
        NAdiscCounter[i][DiscData(DTrain[j],i)] += pDTrain[j] ;

   int max ;
   for (i=0 ; i < NoDiscrete ; i++)
   {
      max = 1 ;
      for (j=2; j <= AttrDesc[DiscIdx[i]].NoValues ;  j++)
         if (NAdiscCounter[i][j] > NAdiscCounter[i][max])
            max = j ;
      Node->NAdiscValue[i] = max ;
    }

    // ---------------------------------------------------------------
    //  set the majority class
    // ---------------------------------------------------------------
    Node->majorClass = Node->NAdiscValue[0] ;

   //-------------------------------------------------------------
   //  continuous attribute missing values - use the average atribute value instead
   //   it would be better to use density estimation with kernel functions
   //-------------------------------------------------------------

   Node->NAcontValue.create(NoContinuous) ;
   marray<double> NAcontWeight(NoContinuous,0.0) ;
   marray<double> NAcontSum(NoContinuous,0.0) ;

   for (i=0; i < NoContinuous ; i++)
   {
     for (j=0 ; j < TrainSize ; j++)
       if (ContData(j,i) != NAcont)
       {
          NAcontWeight[i] += pDTrain[j] ;
          NAcontSum[i] += pDTrain[j] * ContData(j,i) ;
       }
     if (NAcontWeight[i] > 0)
       Node->NAcontValue[i] =  NAcontSum[i]/NAcontWeight[i] ;
     else
       Node->NAcontValue[i] = (maxValue[i] + minValue[i]) / 2.0 ;
    }

   // for estimation of the attributes, constructs, binarization, and discretization
   estimation Estimator(this, DTrain, pDTrain, TrainSize) ;

   // build model for the data (used in case of a leaf and for estimation)
   buildModel(Estimator, Node) ;


   // stopping criterion
   if (time2stop(Node) )
   {
      createLeaf(Node) ;

      return Node ;
   }
   else
   {
       // select/build splitting attribute/construct
      if (! buildConstruct(Estimator, Node, currentDepth) )
      {
            createLeaf(Node) ;
  
            return Node ;
      }
  
      marray<int> LeftTrain, RightTrain ;
      marray<double> pLeftTrain, pRightTrain ;
      int LeftSize = 0, RightSize = 0;
      double wLeft=0.0, wRight = 0.0 ;

      // split the data according to attribute (call by reference)
      split(DTrain, pDTrain, TrainSize, Node, LeftTrain, pLeftTrain, LeftSize,
               RightTrain, pRightTrain, RightSize, wLeft, wRight) ;

      Node->weightLeft = wLeft ;
      // is the resulting split inappropriate
      if (LeftSize==0 || RightSize==0) // || wLeft < minNodeWeight || wRight < minNodeWeight)
      {
         createLeaf(Node) ;

         return Node ;
      }

     
      // recursively call building on both partitions

      Node->left  = buildTree(LeftTrain, pLeftTrain, LeftSize, currentDepth+1) ;

      Node->right = buildTree(RightTrain, pRightTrain, RightSize, currentDepth+1) ;

     
      return  Node;
   }
}


//************************************************************
//
//                        time2stop
//                        ---------
//
//            check the various stopping criteria
//
//************************************************************
boolean featureTree::time2stop(binnode *Node)
{
   // absolute training weight (number of examples) is too small
   if (Node->weight < opt->minNodeWeight)
      return TRUE ;

   // proportion of training examples is too small
   if (Node->weight/rootWeight < opt->relMinNodeWeight)
      return TRUE ;

 
   // proportion of majority class is bigger than parameter
   if (Node->Classify[Node->majorClass]/Node->weight >= opt->majorClassProportion)
      return TRUE ;


   return FALSE ;
}


//**********************************************************************
//
//                         createLeaf
//                         ----------
//
//
//                transform node into a leaf
//
//**********************************************************************
void featureTree::createLeaf(binnode *Node)
{
   // create leaf, label it properly
   Node->Identification = leaf ;

   Node->left = Node->right = 0 ;
   Node->Construct.destroy() ;

}



//**********************************************************************
//
//                         split
//                         -----
//
//
//    split the data acording to given feature into a left and
//                     a right branch
//
//**********************************************************************
void featureTree::split(marray<int> &DTrain,
     marray<double> &pDTrain, int TrainSize, binnode* Node,
     marray<int> &LeftTrain, marray<double> &pLeftTrain, int &LeftSize,
     marray<int> &RightTrain, marray<double> &pRightTrain, int &RightSize,
     double &wLeft, double &wRight)
{
   // data needed to compute probabilities of left and right branch
   // in the case of missing values
   double weightLeft = 0.0, weightOK = 0.0 ;

   double cVal ;
   int   dVal ;
   int i;
   // are there any unknown values in current node
   switch  (Node->Identification)
   {
       case continuousAttribute:
          for (i=0 ; i < TrainSize ; i++)
          {
             cVal = Node->Construct.continuousValue(DiscData, ContData, DTrain[i]) ;
             if (cVal != NAcont)
             {
                weightOK += pDTrain[i] ;
                if (cVal <= Node->Construct.splitValue)
                  weightLeft += pDTrain[i] ;
             }
          }
          break ;
       case discreteAttribute:
          for (i=0 ; i < TrainSize ; i++)
          {
             dVal = Node->Construct.discreteValue(DiscData, ContData, DTrain[i]) ;
             if (dVal != NAdisc)
             {
                weightOK += pDTrain[i] ;
                if (Node->Construct.leftValues[dVal])
                  weightLeft += pDTrain[i] ;
             }
          }
          break ;
       default: error("featureTree::split","Invalid identification of the node") ;
   }

   double probLeftNA ;
   if (weightOK > epsilon)
     probLeftNA= weightLeft / weightOK ;
   else  // invalid split: all the values are missing
     probLeftNA = 0.0 ;

   //  data for split
   marray<int> exLeft(TrainSize) ;
   marray<int> exRight(TrainSize) ;
   marray<double> probLeft(TrainSize) ;
   marray<double> probRight(TrainSize) ;
   LeftSize = RightSize = 0 ;
   wLeft = wRight = 0.0 ;
   int k ;
   // split the examples
   switch  (Node->Identification)
   {
      case continuousAttribute:
          for (k=0  ; k < TrainSize ; k++)
          {
             cVal = Node->Construct.continuousValue(DiscData, ContData, DTrain[k]) ; ;
             if (cVal == NAcont)
             {
                // unknown value
                exLeft[LeftSize] = DTrain[k];
                probLeft[LeftSize] = pDTrain[k] * probLeftNA ;
                exRight[RightSize] = DTrain[k];
                probRight[RightSize] = pDTrain[k] - probLeft[LeftSize] ;
                if (probLeft[LeftSize] > opt->minInstanceWeight)
                {
                   wLeft += probLeft[LeftSize] ;
                   LeftSize++ ;
                }
                if (probRight[RightSize] > opt->minInstanceWeight)
                {
                   wRight += probRight[RightSize] ;
                   RightSize++ ;
                }
             }
             else
               if (cVal <= Node->Construct.splitValue)
               {
                  exLeft[LeftSize] = DTrain[k];
                  probLeft[LeftSize] = pDTrain[k] ;
                  wLeft += pDTrain[k] ;
                  LeftSize ++ ;
               }
               else
               {
                  exRight[RightSize] = DTrain[k];
                  probRight[RightSize] = pDTrain[k] ;
                  wRight += pDTrain[k] ;
                  RightSize ++ ;
               }
          }
          break ;
      case discreteAttribute:
          for (k=0  ; k < TrainSize ; k++)
          {
             dVal = Node->Construct.discreteValue(DiscData, ContData, DTrain[k]) ; ;
             if (dVal == NAdisc)
             {
                // unknown value
                exLeft[LeftSize] = DTrain[k];
                probLeft[LeftSize] = pDTrain[k] * probLeftNA ;
                exRight[RightSize] = DTrain[k];
                probRight[RightSize] = pDTrain[k] - probLeft[LeftSize] ;
                if (probLeft[LeftSize] > opt->minInstanceWeight)
                {
                   wLeft += probLeft[LeftSize] ;
                   LeftSize++ ;
                }
                if (probRight[RightSize] > opt->minInstanceWeight)
                {
                   wRight += probRight[RightSize] ;
                   RightSize++ ;
                }
             }
             else
               if (Node->Construct.leftValues[dVal])
               {
                  exLeft[LeftSize] = DTrain[k];
                  probLeft[LeftSize] = pDTrain[k] ;
                  wLeft += pDTrain[k] ;
                  LeftSize ++ ;
               }
               else
               {
                  exRight[RightSize] = DTrain[k];
                  probRight[RightSize] = pDTrain[k] ;
                  wRight += pDTrain[k] ;
                  RightSize ++ ;
               }
          }
          break ;
      case leaf:
          error("featureTree::split", "node type cannot be leaf") ;
          break ;
   }
   // try not to waste space ;
   LeftTrain.create(LeftSize) ;
   pLeftTrain.create(LeftSize) ;
   for (k = 0; k < LeftSize ; k++)
   {
      LeftTrain[k] = exLeft[k] ;
      pLeftTrain[k] = probLeft[k] ;
   }

   RightTrain.create(RightSize) ;
   pRightTrain.create(RightSize) ;
   for (k = 0; k < RightSize ; k++)
   {
      RightTrain[k] = exRight[k] ;
      pRightTrain[k] = probRight[k] ;
   }
}




// ************************************************************
//
//                 buildConstruct
//                 --------------
//
//    builds construct in a node
//
// ************************************************************
boolean featureTree::buildConstruct(estimation &Estimator, binnode* Node, int currentDepth)
{
   //  will we construct 
   if  (currentDepth > opt->constructionDepth || opt->constructionMode == cSINGLEattribute)
       // singleAttribute
       return singleAttributeModel(Estimator, Node) ;

      // do the construction 

      attributeCount bestAttrType ;
   
      // estimate original attributes
      int bestAttrIdx = Estimator.estimate(opt->selectionEstimator, 0,NoContinuous,1,NoDiscrete, bestAttrType) ;
  
      // copy primary estimations
      //for (int attrIdx = 1 ; attrIdx <= NoAttr; attrIdx++)
      //  if (AttrDesc[attrIdx].continuous)
      //     primaryEstimate[attrIdx] = Estimator.ContEstimation[AttrDesc[attrIdx].tablePlace] ;
      //   else
      //     primaryEstimate[attrIdx] =  Estimator.DiscEstimation[AttrDesc[attrIdx].tablePlace] ;
       
      if (bestAttrIdx == -1)
        return FALSE ;
      double bestEstimate  ;
      if (bestAttrType == aCONTINUOUS)
        bestEstimate = Estimator.ContEstimation[bestAttrIdx] ;
      else 
        bestEstimate = Estimator.DiscEstimation[bestAttrIdx] ;

  
      if ( bestEstimate < opt->minReliefEstimate && 
           (opt->selectionEstimator == estReliefFkEqual || opt->selectionEstimator == estReliefFexpRank ||
            opt->selectionEstimator == estReliefFbestK || opt->selectionEstimator == estRelief ||
            opt->selectionEstimator == estReliefFmerit || opt->selectionEstimator == estReliefFdistance ||
            opt->selectionEstimator == estReliefFsqrDistance) )
         return FALSE ;  
      
      // caches
      marray<construct> stepConjCache(opt->noCachedInNode) ;
      marray<double> stepConjCacheEst(opt->noCachedInNode) ;
      marray<construct> stepSumCache(opt->noCachedInNode) ;
      marray<double> stepSumCacheEst(opt->noCachedInNode) ;
      marray<construct> stepMplyCache(opt->noCachedInNode) ;
      marray<double> stepMplyCacheEst(opt->noCachedInNode) ;
      
      // best
      construct bestConjunct, bestSum, bestProduct ;
      double bestConjunctEst = -FLT_MAX, bestSumEst=-FLT_MAX, bestProductEst=-FLT_MAX ;

      if (opt->constructionMode & cPRODUCT)
        bestProductEst = multiplicator(Estimator, bestProduct, stepMplyCache, stepMplyCacheEst) ; 
      if (opt->constructionMode & cSUM)
        bestSumEst = summand(Estimator, bestSum, stepSumCache, stepSumCacheEst) ; 
      if (opt->constructionMode & cCONJUNCTION)
        bestConjunctEst = conjunct(Estimator, bestConjunct, stepConjCache, stepConjCacheEst) ;
      
      // copy this step caches into global cache

      // prepare space if neccessary
      if (CachedConstructs.filled() + opt->noCachedInNode > CachedConstructs.len())
        CachedConstructs.enlarge(Mmax(CachedConstructs.len() *2, CachedConstructs.len()+2*opt->noCachedInNode)) ;
      
      // select the best from the cached
      int i, conjIdx = 0, sumIdx = 0, mplyIdx = 0 ;
      double Est1, Est2, Est3 ;
      if (conjIdx < stepConjCache.filled())
        Est1 = stepConjCacheEst[conjIdx] ;
      else
        Est1 = -FLT_MAX ;
      if (sumIdx < stepSumCache.filled())
         Est2 = stepSumCacheEst[sumIdx] ;
      else
        Est2 = -FLT_MAX ;
      if (mplyIdx < stepMplyCache.filled())
        Est3 = stepMplyCacheEst[mplyIdx] ;
      else
        Est3 = -FLT_MAX ;

      // fill the cache
      for (i=0 ; i < opt->noCachedInNode ; i++)
      {
         
         if (Est1 >= Est2 && Est1 >= Est3 && Est1 != -FLT_MAX) 
         {
           CachedConstructs.addEnd(stepConjCache[conjIdx]) ;
           conjIdx ++ ;
           if (conjIdx < stepConjCache.filled())
             Est1 = stepConjCacheEst[conjIdx] ;
           else
             Est1 = -FLT_MAX ;
         }
         else 
            if (Est2 >= Est1 && Est2 >= Est3 && Est2 != -FLT_MAX) 
            {
              CachedConstructs.addEnd(stepSumCache[sumIdx]) ;
              sumIdx ++ ;
             if (sumIdx < stepSumCache.filled())
               Est2 = stepSumCacheEst[sumIdx] ;
             else
               Est2 = -FLT_MAX ;
            }
            else
               if (Est3 >= Est1 && Est3 >= Est2 && Est3 != -FLT_MAX) 
            {
              CachedConstructs.addEnd(stepMplyCache[mplyIdx]) ;
              mplyIdx ++ ;
              if (mplyIdx < stepMplyCache.filled())
                Est3 = stepMplyCacheEst[mplyIdx] ;
              else
                Est3 = -FLT_MAX ;
            }
            else
               break ; // none is valid
      }

      // put the best construct into the node
      if ( (opt->constructionEstimator == opt->selectionEstimator && bestEstimate >= bestConjunctEst 
            && bestEstimate >= bestSumEst && bestEstimate >= bestProductEst) || 
            (bestConjunctEst == -FLT_MAX && bestSumEst == -FLT_MAX && bestProductEst == -FLT_MAX) )
      {
        // revert to single attribute
        makeSingleAttrNode(Node, Estimator, bestAttrIdx, bestAttrType) ;
      }
      else
      {
         if (bestConjunctEst >= bestSumEst && bestConjunctEst >= bestProductEst)
           makeConstructNode(Node, Estimator, bestConjunct) ;
         else
            if (bestSumEst >= bestConjunctEst && bestSumEst >= bestProductEst)
              makeConstructNode(Node, Estimator, bestSum) ;
            else
              if (bestProductEst >= bestConjunctEst && bestProductEst >= bestSumEst)
                makeConstructNode(Node, Estimator, bestProduct) ;
              else
              {
                 error("featureTree::buildConstruct", "cannot select the best construct") ;
                 return FALSE ;
              }
      }
      return TRUE ;
}


