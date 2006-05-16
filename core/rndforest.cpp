#include <stdio.h>
#include <float.h>
//#include <conio.h>

#include "ftree.h"
#include "estimator.h"
#include "utils.h"
#include "rndforest.h"
#include "rfUtil.h"
#include "options.h"

extern Options *opt ;

//************************************************************
//
//                      buildForest
//                      ------------
//
//               builds random forest
//
//************************************************************
int featureTree::buildForest() {
   if (state<data && !readProblem())
      return 0 ;

   forest.create(opt->rfNoTrees) ;

   int trainSize = NoTeachCases, it, i ;
   if (opt->rfNoSelAttr==0) 
	    rfNoSelAttr = Mmax(1, intRound(sqrt(double(NoAttr)))) ;
   else if (opt->rfNoSelAttr==-1) 
	    rfNoSelAttr = Mmax(1, 1+int(log2(double(NoAttr)))) ;
   else if (opt->rfNoSelAttr==-2 || opt->rfNoSelAttr >= NoAttr)
      rfNoSelAttr = NoAttr ;
   else rfNoSelAttr = opt->rfNoSelAttr ;

   rootWeight = trainSize ;
   marray<double> weight(trainSize, 1.0), wProb(NoAttr+1), eProb(NoAttr+1) ;

   // prepare weighs for no weighting (set equal weights)
   eProb[0] = 0.0 ;
   for (i=1 ; i <= NoAttr; i++) 
      eProb[i] = double(i)/ NoAttr ;
   eProb[NoAttr] = 1.0 ;

   if (opt->rfPropWeightedTrees > 0) {
		// estimate attributes
		attributeCount attrType ;
		double minEval = DBL_MAX  ;
		estimation Estimator(this, DTeach, weight, trainSize) ;
		Estimator.estimate(estReliefFexpRank, 0, NoContinuous, 1, NoDiscrete, attrType) ;
		for (i=1 ; i <= NoAttr; i++) {
			if (AttrDesc[i].continuous)
				wProb[i] = Estimator.ContEstimation[AttrDesc[i].tablePlace] ;
			else
				wProb[i] =  Estimator.DiscEstimation[AttrDesc[i].tablePlace] ;
			if (wProb[i] < minEval)
				minEval = wProb[i] ;
		}
		// prepare cumulative probability distribution
		double eSum = 0.0 ;
		if (minEval < 0) {
			double negReduction = 10.0 ;
			minEval = -minEval/negReduction + 1e-6 ; ;
			for (i=1 ; i <= NoAttr; i++) {
				if (wProb[i] < 0)
					wProb[i] = wProb[i]/negReduction + minEval;
				else
					wProb[i] += minEval ;
			    eSum += wProb[i] ;
			}
		}
		else 
			for (i=1 ; i <= NoAttr; i++) 
     			eSum += wProb[i] ;
		wProb[0] = 0.0 ;
		for (i=1 ; i <= NoAttr; i++) 
			wProb[i] = wProb[i-1] + wProb[i]/eSum ;
		wProb[NoAttr] = 1.0 ;
   }
   else wProb.copy(eProb) ;
  
   mmatrix<int> oobEval(trainSize, NoClasses+1, 0) ;
   // build forest
   // printf("\niter accuracy\n") ;
   int selEst = opt->selectionEstimator ;
   int noAvailableEst = 5 ;
   marray<int> availEst(noAvailableEst) ;
   availEst[0] = estReliefFexpRank ;
   availEst[1] = estGainRatio ;
   availEst[2] = estMdl ;
   availEst[3] = estGini ;
   availEst[4] = estReliefFmyopic ;
   //availEst[5] = estReliefFexpRank ;
   for (it = 0 ; it < opt->rfNoTrees ; it++) {
	   // prepare training data
	   if (opt->rfSampleProp==0) 
		   bootstrapSample(trainSize, DTeach, forest[it].ib, forest[it].oob) ;
	   else
           randomSample(trainSize, opt->rfSampleProp, DTeach, forest[it].ib, forest[it].oob) ;

	   if (opt->rfMultipleEst)
           opt->selectionEstimator = availEst[it % noAvailableEst] ;
	   if ( i/double(opt->rfNoTrees) < opt->rfPropWeightedTrees) {
		   if (opt->rfNoTerminals ==0)
	         forest[it].t.root = buildForestTree(trainSize, forest[it].ib, wProb) ;
		   else 
			 forest[it].t.root = rfBuildLimitedTree(opt->rfNoTerminals, trainSize, forest[it].ib, wProb) ;
	   }
	   else {
   	      if (opt->rfNoTerminals ==0)
  	         forest[it].t.root = buildForestTree(trainSize, forest[it].ib, eProb) ;
		   else 
			 forest[it].t.root = rfBuildLimitedTree(opt->rfNoTerminals, forest[it].ib.len(), forest[it].ib, eProb) ;
	   }

       // oobEstimate = oobEvaluate(forest[it].t.root, DTeach, forest[it].oob, oobEval) ;
       //printf("%03d %.3f\n",it, oobEstimate) ;
   }
   opt->selectionEstimator = selEst ;
   // regularization
   rfA.create(opt->rfNoTrees, 1.0/opt->rfNoTrees) ;
   if (NoClasses == 2 && opt->rfRegType==1) // global regularization 
        rfRegularize() ;
   
   // compute average accuracy and margin
   marray<double> margin(trainSize) ;
   marray<int> maxOther(trainSize) ;
   double varMargin;
   oobEvaluate(oobEval) ; 
   avgOobAccuracy = oobAccuracy(oobEval) ;
   avgOobMargin = oobMargin(oobEval, maxOther, varMargin) ;
   avgOobCorrelation = varMargin / sqr(oobSTD(maxOther));
   state = random_forest ;
   return 1 ;
}

//************************************************************
//
//                      oobInplaceEvaluate
//                      -----------------
//
//               evaluates trees computed so far    
//
//************************************************************
double featureTree::oobInplaceEvaluate(binnode *root, marray<int> &dSet, marray<boolean> &oobSet, mmatrix<int> &oob) {
// for data set dSet, oobSet[i] contains indicator wheather dSet[i] is an out of bag instance
   int i, j, max, oobCorrect=0, valid=0  ;
   marray<double> probDist(NoClasses+1) ;
   for (i=0; i < dSet.filled() ; i++) {
	  if (oobSet[i]) {
	    // update with current tree
		probDist.init(0.0) ;
        max = rfTreeCheck(root,dSet[i], probDist) ; // prediction        
        // prediction with majority class (disregarding costs)     
        //max=1;
        //for (j=2 ; j<=NoClasses ; j++)
        //  if (probDist[j] > probDist[max])
        //    max=j;
        oob(i,max)++ ;
      }
      // compute oob estimate
	  max = 1 ;
      for (j=2 ; j<=NoClasses ; j++)
         if (oob(i,j) > oob(i,max))
            max=j;
	  if (oob(i, max) > 0) {
         valid ++ ;
	     if (DiscData(dSet[i], 0) == max)
		    oobCorrect++ ;
	  }
   }
   return double(oobCorrect)/valid ;
}


//************************************************************
//
//                      buildForestTree
//                      ---------------
//
//             builds one tree of random forest
//
//************************************************************
binnode* featureTree::buildForestTree(int TrainSize, marray<int> &DTrain, marray<double> &attrProb) {
   binnode* Node = new binnode ;
   Node->weight = TrainSize ;
   Node->Classify.create(NoClasses+1, 0.0) ;
   int i, j ;
   // compute class distribution and weight of a node
   for (i=0 ; i < TrainSize ; i++)
      Node->Classify[DiscData(DTrain[i],0)] += 1.0 ;
   Node->majorClass = 1 ;
   for (j=2 ; j <= NoClasses ; j++)
	   if (Node->Classify[j] > Node->Classify[Node->majorClass])
		   Node->majorClass = j ;

   // stopping criterion
   if (time2stop(Node) )   {
   createLeaf:
     // create leaf, label it properly
     Node->Identification = leaf ;
     Node->DTrain.copy(DTrain) ;
     Node->DTrain.setFilled(TrainSize) ;
     Node->Model.createMajority(Node->majorClass) ;

     Node->left = Node->right = 0 ;
     Node->Construct.destroy() ;
	 Node->NAcontValue.destroy() ;
	 Node->NAdiscValue.destroy() ;

     return Node ;
   }

   //CurrentNode = Node ;
   //CurrentTrainSize = TrainSize ;
   //CurrentExamples = &DTrain ;

   //-------------------------------------------------------------
   // compute most probable values used instead of missing values
   //-------------------------------------------------------------
   Node->NAdiscValue.create(NoDiscrete) ;
   marray<marray<int> > NAdiscCounter(NoDiscrete) ;

   for (i=0 ; i < NoDiscrete ; i++)
      NAdiscCounter[i].create(AttrDesc[DiscIdx[i]].NoValues +1, 0) ;

   for (i=0; i < NoDiscrete ; i++)
     for (j=0 ; j < TrainSize ; j++)
        NAdiscCounter[i][DiscData(DTrain[j],i)] ++ ;

   int max ;
   for (i=0 ; i < NoDiscrete ; i++)
   {
      max = 1 ;
      for (j=2; j <= AttrDesc[DiscIdx[i]].NoValues ;  j++)
         if (NAdiscCounter[i][j] > NAdiscCounter[i][max])
            max = j ;
      Node->NAdiscValue[i] = max ;
   }

   //  continuous attribute missing values - use the average atribute value instead
   Node->NAcontValue.create(NoContinuous) ;
   marray<int> NAcontWeight(NoContinuous,0) ;
   marray<double> NAcontSum(NoContinuous,0.0) ;

   for (i=0; i < NoContinuous ; i++)   {
     for (j=0 ; j < TrainSize ; j++)
       if (ContData(j,i) != NAcont)       {
          NAcontWeight[i] ++ ;
          NAcontSum[i] += ContData(j,i) ;
       }
     if (NAcontWeight[i] > 0)
       Node->NAcontValue[i] =  NAcontSum[i]/NAcontWeight[i] ;
     else
       Node->NAcontValue[i] = (maxValue[i] + minValue[i]) / 2.0 ;
    }

   // for estimation of the attributes, constructs, binarization, and discretization
   marray<double> pDTrain(TrainSize, 1.0) ;
   estimation Estimator(this, DTrain, pDTrain, TrainSize) ;

   // select/build splitting attribute/construct
   if (rfBuildConstruct(Estimator, Node, attrProb) == -FLT_MAX )      
       goto createLeaf ;
  
      marray<int> LeftTrain, RightTrain ;
      int LeftSize = 0, RightSize = 0;

      // split the data according to attribute (call by reference)
      rfSplit(DTrain, TrainSize, Node, LeftTrain, LeftSize, RightTrain, RightSize) ;

      Node->weightLeft = LeftSize ;
      // is the resulting split inappropriate
	  if (LeftSize==0 || RightSize==0) 
          goto createLeaf;
     
      // recursively call building on both partitions
      Node->left  = buildForestTree( LeftSize, LeftTrain, attrProb ) ;
      Node->right = buildForestTree(RightSize, RightTrain, attrProb) ;
      return  Node;
}

//************************************************************
//
//                      buildLimitedTree
//                      ---------------
//
//       builds one tree of random forest limited in size  
//
//************************************************************
binnode* featureTree::rfBuildLimitedTree(int noTerminal, int TrainSize, marray<int> &DTrain, marray<double> &attrProb) {
   // create root bode and pout it into priority list

   binnode *rtNode = rfPrepareLeaf(TrainSize, DTrain) ;
   if (time2stop(rtNode)) {
	   rfRevertToLeaf(rtNode) ;
	   return rtNode ;
   }

   marray<BinNodeRec> pq(noTerminal) ; // priority queue of the nodes
   BinNodeRec nodeEl ;
   nodeEl.value = rtNode ;
   // for estimation of the attributes
   marray<double> pDTrain(TrainSize, 1.0) ;
   estimation Estimator(this, DTrain, pDTrain, TrainSize) ;
   if ((nodeEl.key = rfBuildConstruct(Estimator, rtNode, attrProb)) == -FLT_MAX)  {    
	 rfRevertToLeaf(rtNode) ;
     return rtNode ;
   }

   // add to priority queue
   pq.addPQmax(nodeEl) ;

   marray<int> LeftTrain, RightTrain ;
   int LeftSize = 0, RightSize = 0;
   binnode *Node ;

   while (pq.filled()>0  && pq.filled() < noTerminal-1) {
	  // expand highest priority node
	  pq.deleteMaxPQmax(nodeEl) ;
	  Node = nodeEl.value ;

	  // split the data according to attribute (call by reference)
      rfSplit(DTrain, TrainSize, Node, LeftTrain, LeftSize, RightTrain, RightSize) ;
      Node->weightLeft = LeftSize ;
	  if (LeftSize==0 || RightSize==0) {   // is the resulting split inappropriate
   	     rfRevertToLeaf(Node) ;
  		 --noTerminal ;
	  }
	  else {
         Node->left = rfPrepareLeaf(LeftSize, LeftTrain) ;
		 if (time2stop(Node->left))   {
	        rfRevertToLeaf(Node->left) ;
			--noTerminal ;
		 }
		 else {
			 Estimator.initialize(LeftTrain,pDTrain,LeftSize) ; 
			 if ((nodeEl.key = rfBuildConstruct(Estimator, Node->left, attrProb)) == -FLT_MAX) {     
	              rfRevertToLeaf(Node->left) ;
				  --noTerminal ;
			 }
			 else {
				 nodeEl.value = Node->left ;
				 pq.addPQmax(nodeEl) ;
			 }
		 }
         Node->right = rfPrepareLeaf(RightSize, RightTrain) ;
		 if (time2stop(Node->right))   {
	        rfRevertToLeaf(Node->right) ;
			--noTerminal ;
		 }
		 else {
			 Estimator.initialize(RightTrain,pDTrain,RightSize) ; 
			 if ((nodeEl.key = rfBuildConstruct(Estimator, Node->right, attrProb)) == -FLT_MAX) {     
	              rfRevertToLeaf(Node->right) ;
				  --noTerminal ;
			 }
			 else {
				 nodeEl.value = Node->right ;
				 pq.addPQmax(nodeEl) ;
			 }
		 }
	  }
   }
   // turn remaining into leaves
   for (int i = 0 ; i < pq.filled() ; i++)
      rfRevertToLeaf(pq[i].value) ;

   return rtNode ;

}



//************************************************************
//
//                      rfBuildConstruct
//                      ----------------
//
//                  builds one node of the random forests' tree
//
//************************************************************
double featureTree::rfBuildConstruct(estimation &Estimator, binnode* Node, marray<double> &attrProb) {
   int i, j ;
   // select attributes to take into consideration
   marray<boolean>  selAttr(NoAttr+1) ;
   selAttr.setFilled(NoAttr+1) ;
   if (rfNoSelAttr == NoAttr)
	   // no selection - pure bagging
	   selAttr.init(TRUE) ;
   else {
	    // select attributes
	    selAttr.init(FALSE) ;
        double rndNum ;
		i=0 ; 
		while ( i < rfNoSelAttr) {
			rndNum = randBetween(0.0, 1.0) ;
			for (j=1 ; j <= NoAttr ; j++)
				if ( rndNum <= attrProb[j] )
					break ;
			if (selAttr[j]==FALSE) {
				selAttr[j] = TRUE ;
				i++ ;
			}
		}
   }
   // estimate the attributes
   attributeCount bestType ;
   int bestIdx = Estimator.estimateSelected(opt->selectionEstimator, selAttr,  bestType) ;
   if (bestIdx == -1)
     return -FLT_MAX ;

   makeSingleAttrNode(Node, Estimator, bestIdx, bestType) ;
   if (bestType == aDISCRETE)
	   return Estimator.DiscEstimation[bestIdx] ;
   else 
	   return Estimator.ContEstimation[bestIdx];   
}


//************************************************************
//
//						rfCheck									
//                      ---------------
//
//   returns probability distribution of the instance caseIdx computed by forest
//
//************************************************************
void featureTree::rfCheck(int caseIdx, marray<double> &probDist) {
    marray<double> distr(NoClasses+1) ;
    probDist.init(0.0) ;
    int i, j, max ;
	for (i=0 ; i < opt->rfNoTrees ; i++) {
		max = rfTreeCheck(forest[i].t.root, caseIdx, distr) ;

		if (opt->rfPredictClass) {
           //  max = 1 ;        
		   //for (j=2 ; j<=NoClasses ; j++) 
           //  if (distr[j] > distr[max])
           //     max=j;
		   probDist[max] += 1.0 ;
		   //probDist[max] += rfA[i] ;
		}
		else { // predict with distribution
           for (j=1 ; j <= NoClasses ; j++)
			   probDist[j] += distr[j] ;
		}
	}
	double sum = 0.0 ;
	for (j=1 ; j <= NoClasses ; j++)
       sum += probDist[j] ;
    for (j=1 ; j <= NoClasses ; j++)
       probDist[j] /= sum ;
}


//************************************************************
//
//						rfCheckReg									
//                      ---------------
//
//   returns regularized probability distribution of the instance caseIdx computed by forest
//    
//
//************************************************************
void featureTree::rfCheckReg(int caseIdx, marray<double> &probDist) {
    marray<double> distr(NoClasses+1) ;
    probDist.init(0.0) ;
    int i, max;
	double score = rfA0, maxScore = 0.0 ; 
	for (i=0 ; i < opt->rfNoTrees ; i++) {
		max = rfTreeCheck(forest[i].t.root, caseIdx, distr) ;
        // for two class problems only 
  	    maxScore += fabs(rfA[i]) ;
		if (max == 1)
		   score += rfA[i] ;
		else  score -= rfA[i] ;
	}
	if (score >= 0) {
      probDist[1] = score/maxScore ;
	  probDist[2] = 1.0 - probDist[1] ;
	}
	else {
	   probDist[2] = -score/maxScore ;
	   probDist[1] = 1.0 - probDist[2] ;
	}
}



// importance of attributes
//************************************************************
//
//                      varImportance
//                      ---------------
//
//          evaluates importance of the attributes by forest
//
//************************************************************
void featureTree::varImportance(marray<double> &varEval) {
    marray<int> discOrig(NoCases), discTemp(NoCases) ;
	marray<double> contOrig(NoCases), contTemp(NoCases) ;
	discOrig.setFilled(NoCases) ;
	discTemp.setFilled(NoCases) ;
	contOrig.setFilled(NoCases) ;
	contTemp.setFilled(NoCases) ;   
	mmatrix<int> oob(NoTeachCases, NoClasses+1) ;
    marray<int> maxOther(NoTeachCases) ; // dummy placeholder
    double varMargin ; // dummy placeholder
	for (int iA = 1 ; iA <= NoAttr ; iA++) {
		if (AttrDesc[iA].continuous) {
  	        // save original values of instances and reschuffle it
			ContData.outColumn(AttrDesc[iA].tablePlace, contOrig) ;
			contTemp.copy(contOrig) ;
			contTemp.shuffle() ;
			ContData.inColumn(contTemp, AttrDesc[iA].tablePlace) ;
		}
		else {
			DiscData.outColumn(AttrDesc[iA].tablePlace, discOrig) ;
			discTemp.copy(discOrig) ;
			discTemp.shuffle() ;
			DiscData.inColumn(discTemp, AttrDesc[iA].tablePlace) ;	
		}
	    // compute margin
		oobEvaluate(oob) ;
        varEval[iA] = avgOobMargin - oobMargin(oob, maxOther, varMargin) ;
        
		if (AttrDesc[iA].continuous) 
			ContData.inColumn(contOrig, AttrDesc[iA].tablePlace) ;
		else
			DiscData.inColumn(discOrig, AttrDesc[iA].tablePlace) ;

	}
}

// importance of attribute values
//************************************************************
//
//                      avImportance
//                      ------------
//   
//     evaluates importance of the discrete attributes' values  by forest
//
//************************************************************
void featureTree::avImportance(marray<marray<double> > &avEval) {
    marray<int> discOrig(NoCases), discTemp(NoCases) ;
	marray<double> contOrig(NoCases), contTemp(NoCases) ;
	discOrig.setFilled(NoCases) ;
	discTemp.setFilled(NoCases) ;
	contOrig.setFilled(NoCases) ;
	contTemp.setFilled(NoCases) ;   
	mmatrix<int> oob(NoTeachCases, NoClasses+1) ;
	marray<double> avMarginOrig, avMarginShuffled ;
	for (int iA = 1 ; iA < NoDiscrete ; iA++) {
		DiscData.outColumn(iA, discOrig) ;
		avMarginOrig.create(AttrDesc[DiscIdx[iA]].NoValues+1) ;
	    oobEvaluate(oob) ;
		oobMarginAV(oob, AttrDesc[DiscIdx[iA]].NoValues, discOrig, avMarginOrig) ;
		discTemp.copy(discOrig) ;
		discTemp.shuffle() ;
		// shuffleChange(AttrDesc[DiscIdx[iA]].NoValues, discTemp) ;
		DiscData.inColumn(discTemp, iA) ;	

   	    // evaluate changes
	    oobEvaluate(oob) ;
        
        // compute margins
		avMarginShuffled.create(AttrDesc[DiscIdx[iA]].NoValues+1) ;
		oobMarginAV(oob, AttrDesc[DiscIdx[iA]].NoValues, discOrig, avMarginShuffled) ;
		for (int iV=0 ; iV <= AttrDesc[DiscIdx[iA]].NoValues ; iV++) 
            avEval[iA][iV] = avMarginOrig[iV] - avMarginShuffled[iV] ;
		DiscData.inColumn(discOrig, iA) ;
	}
}



//************************************************************
//
//                      oobEvaluate
//                      ------------
//
//          evaluation of the oob instances
//
//************************************************************
void featureTree::oobEvaluate(mmatrix<int> &oob) {
   marray<double> distr(NoClasses+1) ;
   int iT, i, max ;
   oob.init(0) ;
   for (iT = 0 ; iT < opt->rfNoTrees ; iT++) {
  	 for (i=0 ; i < NoTeachCases ; i++)
		if (forest[iT].oob[i]) {
			max = rfTreeCheck(forest[iT].t.root, DTeach[i], distr) ;
            //max=1;
            //for (j=2 ; j<=NoClasses ; j++)
            //   if (distr[j] > distr[max])
            //      max=j;
            oob(i, max)++ ;
		}
   }
}

//************************************************************
//
//                      oobMargin
//                      ----------
//
//    estimates margin of the training instances by the oob sample
//
//************************************************************
double featureTree::oobMargin(mmatrix<int> &oob, marray<int> &maxOther, double &varMargin) {
   int sum, correctClass, i, j ;
   double margin, sumMargin = 0.0, margin2 = 0.0 ;
   for (i=0 ; i < NoTeachCases ; i++) {
       sum = 0 ;
	   correctClass = DiscData(DTeach[i], 0) ;
       if (correctClass > 1) 
           maxOther[i] = 1 ;
       else maxOther[i] = 2 ;
	   for (j=1 ; j <=NoClasses ; j++) {
           sum += oob(i,j) ;		
           if (j != correctClass && oob(i,j) > oob(i,maxOther[i]))
				maxOther[i] = j ;
	   }
       if (sum > 0) 
          margin = (oob(i, correctClass) - oob(i,maxOther[i])) / double(sum) ;
	   else margin = 0.0 ;  
	   sumMargin += margin ;
       margin2 += sqr(margin) ;
   }
   double avgMargin = sumMargin / double(NoTeachCases) ;
   varMargin = (margin2 / NoTeachCases) - sqr(avgMargin) ;
   return avgMargin ;
}


//************************************************************
//
//                      oobMarginAV
//                      ------------
//
//    estimates margin of the training instances by the oob sample
//    the score is estimated separately for each original ttribute's value   
//
//
//************************************************************
void featureTree::oobMarginAV(mmatrix<int> &oob, int noVal, marray<int> &origVal, 
								marray<double> &avMargin) {
   int sum, correctClass, i, j, maxOther ;
   double margin ;
   avMargin.init(0.0) ;
   marray<int> noCases(avMargin.len(), 0) ;
   for (i=0 ; i < NoTeachCases ; i++) {
       sum = 0 ;
	   correctClass = DiscData(DTeach[i], 0) ;
       if (correctClass > 1) 
           maxOther = 1 ;
       else maxOther = 2 ;
	   for (j=1 ; j <=NoClasses ; j++) {
           sum += oob(i,j) ;		
           if (j != correctClass && oob(i,j) > oob(i,maxOther))
				maxOther = j ;
	   }
       if (sum > 0) 
          margin = (oob(i, correctClass) - oob(i,maxOther)) / double(sum) ;
	   else margin = 0.0 ;  
	   if (origVal[i] != NAdisc) {
	     avMargin[origVal[i]] += margin ;
	     ++ noCases[origVal[i]] ;
	     avMargin[0] += margin ;
		 ++ noCases[0] ;
	   }
   }
   for (int k=0 ; k <= noVal ; k++)
	   avMargin[k] /= double(noCases[k]) ;
}


//************************************************************
//
//                      oobSTD
//                      -------
//
//   computes standard deviation of the forest with oob instances
//
//************************************************************
double featureTree::oobSTD(marray<int> &maxOther) {
   marray<double> distr(NoClasses+1) ;
   int iT, i, max, p1,p2,all ;
   double sd = 0.0 ;
   for (iT = 0 ; iT < opt->rfNoTrees ; iT++) {
     p1 = p2 = all = 0 ;
  	 for (i=0 ; i < NoTeachCases ; i++)
		if (forest[iT].oob[i]) {
            all++ ;
			max = rfTreeCheck(forest[iT].t.root, DTeach[i], distr) ;     
            //max=1;
            //for (j=2 ; j<=NoClasses ; j++)
            //   if (distr[j] > distr[max])
            //      max=j;
            if (DiscData(DTeach[i], 0) == max)
                p1++ ;
            else if (max == maxOther[i])
                p2++ ;
        }
     sd += sqrt((p1+p2)/double(all)+sqr((p1-p2)/double(all))) ;
   }
   sd /= opt->rfNoTrees ;
   return sd ;
}

//************************************************************
//
//                      oobAccuracy
//                      ------------
//
//     computes accuracy of the forst with oob evaluation
//
//************************************************************
double featureTree::oobAccuracy(mmatrix<int> &oob) {
   int i, j,max, correct=0 ;
   for (i=0 ; i < NoTeachCases ; i++) { 
	  max = 1 ;
      for (j=2 ; j<=NoClasses ; j++)
         if (oob(i,j) > oob(i,max))
            max=j;
      if (DiscData(DTeach[i], 0) == max)
		  correct++ ;
	}
   return double(correct)/NoTeachCases ; 
}



//************************************************************
//
//                      rfResultHead
//                      ----------------
//
//              prints header of random forest results 
//
//************************************************************
void featureTree::rfResultHead(FILE *to) const
{
   fprintf(to,"\n%3s %5s %5s %5s %5s   %5s %5s %5s   %5s %5s %5s %5s",
       "idx", "accTr","cstTr","infTr","AUCtr", "oobAc","oobMg", "oobRo", "accTe","cstTe","infTe","AUCte") ;
   if (NoClasses == 2)
      fprintf(to,"  %5s %5s","senTe","spcTe") ;
   fprintf(to,"\n") ;
   printLine(to,"-",86) ; 
}


//************************************************************
//
//                      rfResultLine
//                      ---------------
//
//        prints results of one random forest into a single line
//
//************************************************************
void featureTree::rfResultLine(FILE *to, int idx,
        double TrainAccuracy, double TrainCost, double TrainInf, double TrainAuc, 
        double oobAccuracy, double oobMargin, double oobCorrelation, 
        double TestAccuracy, double TestCost, double TestInf, double TestAuc, 
		double sensitivity, double specificity) const 
{
    char idxStr[32] ;
    if (idx>=0) sprintf(idxStr,"%3d",idx);
    else if (idx == -1) strcpy(idxStr,"avg") ;
    else if (idx == -2) strcpy(idxStr,"std") ;
    else strcpy(idxStr,"???") ;
       
   fprintf(to,"%3s %5.3f %5.3f %5.3f %5.3f   %5.3f %5.3f %5.3f   %5.3f %5.3f %5.3f %5.3f",
               idxStr, TrainAccuracy, TrainCost, TrainInf, TrainAuc,
               oobAccuracy, oobMargin, oobCorrelation,
               TestAccuracy,  TestCost, TestInf, TestAuc) ;
   if (NoClasses == 2) {
      fprintf(to,"  %5.3f %5.3f",sensitivity,specificity) ;
   }
   fprintf(to, "\n") ;

}


//************************************************************
//
//                      printAttrEval
//                      -------------
//
//        prints random forests' attribute value evaluation results 
//
//************************************************************
void featureTree::printAttrEval(FILE *to, marray<int> &idx, marray<marray<double> > &attrEval) {
    char idxStr[32] ;
	int i, j, iA ;
	// print header
	fprintf(to, "\n%18s", "Attribute name") ;
	for (i=0 ; i < attrEval.filled() ; i++) {	
		if (idx[i]>=0) sprintf(idxStr,"%3d",idx[i]);
		else if (idx[i] == -1) strcpy(idxStr,"avg") ;
		else if (idx[i] == -2) strcpy(idxStr,"std") ;
		else strcpy(idxStr,"???") ;

		fprintf(to, "  %6s",idxStr) ;
	}
    fprintf(to,"\n") ;
	for (j=0 ; j < 18 + 7*attrEval.filled() ; j++)
		fprintf(to,"-") ;
	for (iA=1 ; iA <= NoAttr; iA++) {
	  fprintf(to, "\n%18s",AttrDesc[iA].AttributeName) ;
  	  for (i=0 ; i < attrEval.filled() ; i++) 
		  fprintf(to,"  %6.3f",attrEval[i][iA]) ;
	}
    fprintf(to,"\n") ;
}


//**********************************************************************
//
//                         split
//                         -----
////
//    split the data acording to given feature into the left and right branch
//
//**********************************************************************
void featureTree::rfSplit(marray<int> &DTrain, int TrainSize, binnode* Node,
						  marray<int> &LeftTrain, int &LeftSize, 
						  marray<int> &RightTrain, int &RightSize){
   double cVal ;
   int   dVal ;
   //  data for split
   marray<int> exLeft(TrainSize) ;
   marray<int> exRight(TrainSize) ;
   LeftSize = RightSize = 0 ;
   int k ;
   // split the examples
   switch  (Node->Identification)
   {
      case continuousAttribute:
          for (k=0  ; k < TrainSize ; k++)
          {
               cVal = Node->Construct.continuousValue(DiscData, ContData, DTrain[k]) ; 
               if (cVal == NAcont)   
			  	  cVal = Node->NAcontValue[Node->Construct.root->attrIdx] ;

               if (cVal <= Node->Construct.splitValue) {
                  exLeft[LeftSize] = DTrain[k];
                  LeftSize ++ ;
               }
               else  {
                  exRight[RightSize] = DTrain[k];
                  RightSize ++ ;
               }
          }
          break ;
      case discreteAttribute:
          for (k=0  ; k < TrainSize ; k++)  {
               dVal = Node->Construct.discreteValue(DiscData, ContData, DTrain[k]) ; ;
               if (dVal == NAdisc)
	   			   dVal = Node->NAdiscValue[Node->Construct.root->attrIdx] ;
              if (Node->Construct.leftValues[dVal]) {
                  exLeft[LeftSize] = DTrain[k];
                  LeftSize ++ ;
               }
               else  {
                  exRight[RightSize] = DTrain[k];
                  RightSize ++ ;
               }
          }
          break ;
      case leaf:
          error("featureTree::rfSplit", "node type cannot be leaf") ;
          break ;
   }
   // try not to waste space ;
   LeftTrain.create(LeftSize) ;
   for (k = 0; k < LeftSize ; k++)
      LeftTrain[k] = exLeft[k] ;

   RightTrain.create(RightSize) ;
   for (k = 0; k < RightSize ; k++)
      RightTrain[k] = exRight[k] ;
}

//************************************************************
//
//                      rfTreeCheck
//                      -----------
//
//        computes classification for single case in one tree
//
//************************************************************
int featureTree::rfTreeCheck(binnode *branch, int caseIdx, marray<double> &probDist)
{
   switch (branch->Identification)  {   
        case leaf:
              branch->Model.predict(branch, caseIdx, probDist) ;
			  return branch->majorClass ;
        case continuousAttribute:
			  {
				double contValue = branch->Construct.continuousValue(DiscData, ContData, caseIdx) ;
                if (contValue == NAcont)
					contValue = branch->NAcontValue[branch->Construct.root->attrIdx] ;
                if (contValue <= branch->Construct.splitValue)
                    return rfTreeCheck(branch->left, caseIdx, probDist) ;
                else 
                   return rfTreeCheck(branch->right, caseIdx,probDist) ;
			  }
			  break ;
		case discreteAttribute: 
			   {
                int discValue = branch->Construct.discreteValue(DiscData, ContData, caseIdx) ;
                if (discValue == NAdisc)
					discValue = branch->NAdiscValue[branch->Construct.root->attrIdx] ;
                if (branch->Construct.leftValues[discValue])
                    return rfTreeCheck(branch->left, caseIdx, probDist) ;
                else 
                    return rfTreeCheck(branch->right, caseIdx,probDist) ;
			   }
			   break ;
        default:
                error("featureTree::check", "invalid branch identification") ;
				return -1 ;
   }
}

//************************************************************
//
//                      rfNearCheck
//                      ----------
//
//        computes classification for single case in one tree
//        but taking locality into account   
//
//************************************************************
void featureTree::rfNearCheck(int caseIdx, marray<double> &probDist) {
    marray<IntSortRec> nearr(NoCases) ;
    int i, j, max, iT ;
	for (i=0 ; i < NoCases ; i++) {
		nearr[i].key = 0 ;
		nearr[i].value = i ;
	}
	marray<double> distr(NoClasses+1) ;
	for (iT=0 ; iT < opt->rfNoTrees ; iT++) 
		rfFindNearInTree(forest[iT].t.root, caseIdx, nearr) ;
	nearr.setFilled(NoCases) ;
	nearr[caseIdx].key = 0 ;
	int k = Mmin(opt->rfkNearestEqual, NoTeachCases-1) ;
	nearr.sortKdsc(k) ;
    
	marray<sortRec> treeMg(opt->rfNoTrees) ;
	for (iT=0 ; iT < opt->rfNoTrees ; iT++) {
		treeMg[iT].key = 0 ;
		treeMg[iT].value = iT ; 
	}
	int treeCount, maxOther ;
	double sumTreeMg = 0.0 ;
	for (iT=0 ; iT < opt->rfNoTrees ; iT++)  {
		treeCount = 0 ;
	   for (i=nearr.filled()-1 ; i > nearr.filled()-1-k ; i--) 
  		  if (! forest[iT].ib.member(nearr[i].value)) {
             treeCount++ ; 
			 max = rfTreeCheck(forest[iT].t.root, nearr[i].value, distr) ;
			 if (DiscData(nearr[i].value,0)==1)
			  	maxOther = 2 ;
			 else maxOther = 1 ;
			 for (j=maxOther+1 ; j <= NoClasses; j++)
				if (j != DiscData(nearr[i].value,0) && distr[j] > distr[maxOther])
					maxOther = j ;
			 treeMg[iT].key += distr[DiscData(nearr[i].value,0)] - distr[maxOther] ; 
		  }
       treeMg[iT].key /= double(treeCount) ;
       if (treeMg[iT].key > 0)
	     sumTreeMg += treeMg[iT].key ;
	}
	treeMg.setFilled(opt->rfNoTrees) ;
	// treeMg.qsortDsc() ;
    double treeWeight ;
    probDist.init(0.0) ; 
	for (iT=0 ; iT < treeMg.filled() ; iT++) {
		if (treeMg[iT].key <= 0)
			continue ;
		max = rfTreeCheck(forest[treeMg[iT].value].t.root, caseIdx, distr) ;
		treeWeight = treeMg[iT].key/sumTreeMg  ;
		if (opt->rfPredictClass) 
		   probDist[max] += treeWeight ;
		else  // predict with distribution
           for (j=1 ; j <= NoClasses ; j++)
			   probDist[j] += distr[j]*treeWeight ;
		
	}
	double sum = 0.0 ;
	for (j=1 ; j <= NoClasses ; j++)
       sum += probDist[j] ;
    for (j=1 ; j <= NoClasses ; j++)
       probDist[j] /= sum ;
}


//************************************************************
//
//                      rfTreeCheck
//                      ------------
//
//        computes classification for single case in one tree
//        taking locality into account 
//
//************************************************************
void featureTree::rfFindNearInTree(binnode *branch, int caseIdx, marray<IntSortRec> &nearr)
{
   switch (branch->Identification)  {   
        case leaf:
			{
				for (int i=0 ; i < branch->DTrain.len() ; i++) 
					nearr[branch->DTrain[i]].key++ ;
			   return ;
			}
        case continuousAttribute:
			  {
				double contValue = branch->Construct.continuousValue(DiscData, ContData, caseIdx) ;
                if (contValue == NAcont)
					contValue = branch->NAcontValue[branch->Construct.root->attrIdx] ;
                if (contValue <= branch->Construct.splitValue)
                    rfFindNearInTree(branch->left, caseIdx, nearr) ;
                else 
                   rfFindNearInTree(branch->right, caseIdx,nearr) ;
			  }
			  return ;
		case discreteAttribute: 
			   {
                int discValue = branch->Construct.discreteValue(DiscData, ContData, caseIdx) ;
                if (discValue == NAdisc)
					discValue = branch->NAdiscValue[branch->Construct.root->attrIdx] ;
                if (branch->Construct.leftValues[discValue])
                    rfFindNearInTree(branch->left, caseIdx, nearr) ;
                else 
                   rfFindNearInTree(branch->right, caseIdx, nearr) ;
			   }
			   return ;
        default:
                error("featureTree::rfFindNearInTree", "invalid branch identification") ;
				return  ;
   }
}

void featureTree::rfRevertToLeaf(binnode *Node) {
   Node->Construct.destroy() ;
   Node->NAcontValue.destroy() ;
   Node->NAdiscValue.destroy() ;
   Node->Identification = leaf ;
}

binnode* featureTree::rfPrepareLeaf(int TrainSize, marray<int> &DTrain) {
   binnode* Node = new binnode ;
   Node->weight = TrainSize ;
   Node->Classify.create(NoClasses+1, 0.0) ;
   int i, j ;
   // compute class distribution and weight of a node
   for (i=0 ; i < TrainSize ; i++)
      Node->Classify[DiscData(DTrain[i],0)] += 1.0 ;
   Node->majorClass = 1 ;
   for (j=2 ; j <= NoClasses ; j++)
	   if (Node->Classify[j] > Node->Classify[Node->majorClass])
		   Node->majorClass = j ;

   // create leaf, label it properly
   Node->Identification = leaf ;
   Node->DTrain.copy(DTrain) ;
   Node->DTrain.setFilled(TrainSize) ;
   Node->Model.createMajority(Node->majorClass) ;

   Node->left = Node->right = 0 ;

   // prepare things for interior node
  
   // compute most probable values used instead of missing values
   Node->NAdiscValue.create(NoDiscrete) ;
   marray<marray<int> > NAdiscCounter(NoDiscrete) ;

   for (i=0 ; i < NoDiscrete ; i++)
      NAdiscCounter[i].create(AttrDesc[DiscIdx[i]].NoValues +1, 0) ;

   for (i=0; i < NoDiscrete ; i++)
     for (j=0 ; j < TrainSize ; j++)
        NAdiscCounter[i][DiscData(DTrain[j],i)] ++ ;

   int max ;
   for (i=0 ; i < NoDiscrete ; i++)   {
      max = 1 ;
      for (j=2; j <= AttrDesc[DiscIdx[i]].NoValues ;  j++)
         if (NAdiscCounter[i][j] > NAdiscCounter[i][max])
            max = j ;
      Node->NAdiscValue[i] = max ;
   }

   //  continuous attribute missing values - use the average atribute value instead
   Node->NAcontValue.create(NoContinuous) ;
   marray<int> NAcontWeight(NoContinuous,0) ;
   marray<double> NAcontSum(NoContinuous,0.0) ;

   for (i=0; i < NoContinuous ; i++)   {
     for (j=0 ; j < TrainSize ; j++)
       if (ContData(j,i) != NAcont)       {
          NAcontWeight[i] ++ ;
          NAcontSum[i] += ContData(j,i) ;
       }
     if (NAcontWeight[i] > 0)
       Node->NAcontValue[i] =  NAcontSum[i]/NAcontWeight[i] ;
     else
       Node->NAcontValue[i] = (maxValue[i] + minValue[i]) / 2.0 ;
   }
   return Node ;
}


 
