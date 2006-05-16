/*********************************************************************
*   Name:              modul trutil  (tree utillities)
*
*   Description:  utillities for  tree
*
*********************************************************************/

#include <string.h>     // dealing with names
#include <stdlib.h>     // min, max
#include <stdio.h>
#include <float.h>

#include "utils.h"
#include "error.h"
#include "ftree.h"
#include "constrct.h"
#include "estimator.h"
#include "frontend.h"
#include "options.h"

extern featureTree* gFT ; // used by ContDataRetriever
extern Options *opt ;


//************************************************************
//
//                      printFTree
//                      ----------
//
//   recursively prints the entire feature tree on a given stream;
//   if features are to long, make a abbrevaition and full
//                  description below
//
//************************************************************
void featureTree::printFTree(FILE *out,  int &FeatureNo,
    marray<binnode*> &FeatureNode, marray<binnode*> &ModelNode,
    int &LeavesNo, binnode *branch, int place)
{
   if (branch)
   {
      if (branch->left) // not the leaf yet
      {
         int fNo = FeatureNo++ ;   // reserve current index

         printFTree(out, FeatureNo, FeatureNode, ModelNode, LeavesNo,
                    branch->left, place+5);

         fprintf(out,"%*sf%d\n",place," ",fNo) ;
         FeatureNode[fNo] = branch ;

         printFTree(out, FeatureNo, FeatureNode, ModelNode, LeavesNo,
                    branch->right, place+5);
      }
      else
      {
         fprintf(out,"%*sl%d\n",place," ",LeavesNo) ;
         ModelNode[LeavesNo] = branch ;
         LeavesNo++ ;
      }
   }
}


//************************************************************
//
//                   printFTreeFile
//                   --------------
//
//             prints the feature tree on a file
//
//************************************************************
void featureTree::printFTreeFile(char *FileName, int idx,
        int Leaves, int freedom,
        double TrainAccuracy, double TrainCost, double TrainInf,double TrainAuc,
        double TestAccuracy, double TestCost, double TestInf, double TestAuc,
        mmatrix<int> &TrainPMx, mmatrix<int> &TestPMx, double TestSens, double TestSpec) 
{
   FILE *to, *toDot ;
   if ((to=fopen(FileName,"w"))==NULL)
   {
       error("Cannot open tree output file",FileName) ;
       return ;
   }
   outVersion(to) ;
   opt->outConfig(to) ;
   printLine(to,"-",83)  ;
   printResultsHead(to) ;
   printResultLine(to, idx,
                            Leaves, freedom,
                            TrainAccuracy, TrainCost, TrainInf, TrainAuc,
                            TestAccuracy, TestCost, TestInf, TestAuc, TestSens, TestSpec) ;
   printLine(to,"-",70)  ;

   int FeatureNo  = 0;
   int noLeaf = noLeaves() ;
   marray<binnode*> FeatureNode(noLeaf) ;
   marray<binnode*> ModelNode(noLeaf) ;
   int LeavesNo  = 0;

   char buf[MaxFeatureStrLen] ;

   printFTree(to, FeatureNo, FeatureNode, ModelNode, LeavesNo, root, 0);

   printLine(to,"-",70)  ;

   if (opt->printTreeInDot)
   {
      int fNo = 0, lNo = 0 ;  
	  strcpy(buf, FileName) ;
	  strcat(buf, ".dot") ;
      if ((toDot=fopen(buf,"w"))==NULL)
	  {
         error("Cannot open dot tree output file",buf) ;
	  }
	  else {
          fprintf(toDot, "digraph \"%s\" {\n\tsize = \"7,10\"  /* set appropriately to see the whole graph */\n", FileName) ;

		  printFTreeDot(toDot, root, fNo, lNo) ;
      }
   }

   int i ;
   for (i=0; i<FeatureNo ; i++)
   {
      Feature2Str(FeatureNode[i], buf);
      fprintf(to, "f%d: %s\n", i, buf) ;
 	  if (opt->printTreeInDot)
    	fprintf(toDot, "\tf%d [label = \"%s\"]\n", i, buf) ;
   }

   int headLen  = 0 ;
   headLen += 20;
   fprintf(to, "\n        Leaf weight") ;

   for (i=0 ; i<NoClasses; i++)
   {
      fprintf(to,"%*s",Mmax(int(strlen(AttrDesc[0].ValueName[i])+2),9),AttrDesc[0].ValueName[i]) ;
      headLen += Mmax(int(strlen(AttrDesc[0].ValueName[i])+2),9) ;
   }
   fprintf(to,"  prediction\n") ;
   headLen += 12 ;
   for (i=0 ; i<headLen; i++)
     fprintf(to,"-") ;
   fprintf(to, "\n") ;
   fprintf(to, "      |            ") ;
   for (i=0 ; i<NoClasses; i++)
      fprintf(to,"%*.4f", Mmax(int(strlen(AttrDesc[0].ValueName[i])+2),9), AttrDesc[0].valueProbability[i+1]) ;
   fprintf(to, "  a priori\n") ;
   for (i=0 ; i<headLen; i++)
     fprintf(to, "-") ;
   fprintf(to,"\n") ;

   int j ;
   char *ModelDescription ;
   
   for (i=0 ; i<LeavesNo ; i++)
   {
      fprintf(to, "l%-4d |%12.2f",i,ModelNode[i]->weight) ;
      for (j=0 ; j<NoClasses ; j++)
         fprintf(to,"%*.4f", Mmax(int(strlen(AttrDesc[0].ValueName[j])+2),9),
                            ModelNode[i]->Classify[j+1] / ModelNode[i]->weight );
   
      ModelDescription = ModelNode[i]->Model.descriptionString() ;
      fprintf(to,"  %s\n", ModelDescription ) ;

	  if (opt->printTreeInDot)
		  fprintf(toDot, "\tl%d [shape = box, label = \"%s\"]\n", i, ModelDescription) ;

      delete [] ModelDescription ;
   }
   for (i=0 ; i<headLen; i++)
     fprintf(to,"-") ;
   fprintf(to,"\n\n") ;

   if (opt->printTreeInDot)
   {
      fprintf(toDot, "}\n") ;
      fclose(toDot) ;
   }

   // print prediction matrixes
   fprintf(to,"Prediction matrix for training set after pruning (%d instances)\n",TrainPMx(0,0)) ;
   printLine(to,"-",65)  ;
   for (i=0 ; i<NoClasses; i++)
      fprintf(to," (%c)  ",'a'+i) ;
   fprintf(to,"    <- classified as\n") ;
   for (i=0 ; i<NoClasses*6; i++)
      fprintf(to, "-") ;
   fprintf(to,"\n") ;
   for (j=1 ; j <= NoClasses; j++)
   {
      for (i=1 ; i<=NoClasses; i++)
         fprintf(to,"%4d  ", TrainPMx(i,j)) ;
      fprintf(to,"    (%c): %s\n",'a'+j-1,AttrDesc[0].ValueName[j-1]) ;
   }
   fprintf(to, "\n") ;
   
   fprintf(to,"Prediction matrix for testing set after pruning (%d instances)\n",TestPMx(0,0)) ;
   printLine(to,"-",65)  ;
   for (i=0 ; i<NoClasses; i++)
      fprintf(to," (%c)  ",'a'+i) ;
   fprintf(to,"    <- classified as\n") ;
   for (i=0 ; i<NoClasses*6; i++)
      fprintf(to, "-") ;
   fprintf(to,"\n") ;
   for (j=1 ; j <= NoClasses; j++)
   {
      for (i=1 ; i<=NoClasses; i++)
         fprintf(to,"%4d  ",TestPMx(i,j)) ;
      fprintf(to,"    (%c): %s\n",'a'+j-1,AttrDesc[0].ValueName[j-1]) ;
   }
   fprintf(to, "\n") ;
   if (NoClasses == 2) {
	  fprintf(to,"\nPositives: %s, negatives: %s", AttrDesc[0].ValueName[0], AttrDesc[0].ValueName[1]) ;
	  fprintf(to, "\nSensitivity: %.3f\nSpecificity: %.3f\n", TestSens, TestSpec) ;
   }
   fclose(to) ;
}

//************************************************************
//
//                      printFTreeDot
//                      -------------
//
//   recursively prints the entire feature tree on a given stream
//   in a dot format
//
//************************************************************
void featureTree::printFTreeDot(FILE *outDot,  binnode *branch, int &FeatureNo, int &LeavesNo)
{
   if (branch)
   {
      if (branch->left) // not the leaf yet
      {
         int fNo = FeatureNo++ ;   // reserve current index
        
         if (branch->left->left) // is left one the leaf
		   fprintf(outDot, "\tf%d -> f%d [label = \"yes\"]\n", fNo, FeatureNo) ;
		 else 
		   fprintf(outDot, "\tf%d -> l%d [label = \"yes\"]\n", fNo, LeavesNo) ;

         printFTreeDot(outDot, branch->left, FeatureNo, LeavesNo);

         if (branch->right->left) // is right one the leaf
		   fprintf(outDot, "\tf%d -> f%d [label = \"no\"]\n", fNo, FeatureNo) ;
		 else 
		   fprintf(outDot, "\tf%d -> l%d [label = \"no\"]\n", fNo, LeavesNo) ;

         printFTreeDot(outDot, branch->right, FeatureNo, LeavesNo);
	  }
      else  {
         // fprintf(outDot, "\tl%d [shape = box]\n", LeavesNo) ;
         LeavesNo++ ;
      }
   }
}



//************************************************************
//
//                      outDomainSummary
//                      ---------------
//
//     prints various parameters of the data
//
//************************************************************

void featureTree::outDomainSummary(FILE *to) const
{
    fprintf(to,"\n\n DATA INFO") ;
    fprintf(to,"\n-----------------------------------------------------------") ;
    fprintf(to,"\nDomain name: %s", opt->domainName) ;
    fprintf(to,"\nNumber of examples: %d", NoCases) ;
    fprintf(to,"\nNumber of class values: %d", NoClasses) ;
    fprintf(to,"\nNumber of attributes: %d", NoAttr) ;
    fprintf(to,"\nNumber of discrete attributes: %d", NoDiscrete-1) ;
    fprintf(to,"\nNumber of continuous attributes: %d", NoContinuous) ;
    fprintf(to,"\n-----------------------------------------------------------\n") ;
}



//************************************************************
//
//                      test
//                      ----
//
//        performs testing on testing examples
//
//************************************************************
void featureTree::test(marray<int> &DSet, int SetSize, double &Accuracy, 
           double &avgCost, double &Inf, double &Auc,  
		   mmatrix<int> &PredictionMatrix, double &sensitivity, double &specificity, FILE *probabilityFile){
   if (state<tree)   {
      error("featureTree::test", "You cannot perform testing, without first constructing/reading tree");
      return  ;
   }
   if (SetSize == 0) {
      error("featureTree::test","There is no data set available.");
      return ;
   }
   
   int correct = 0 ;
   int i,j,cMin, c ;
   marray<double> probDist(NoClasses+1) ;
   double infi=0.0, Cost = 0.0 ;
   double pClPrior, pClObtain, minRisk, cRisk ;
   PredictionMatrix.init(0) ;
   PredictionMatrix(0,0) = SetSize ;
   mmatrix<marray<double> > mm(NoClasses+1,NoClasses+1) ;
   for (i=1; i <= NoClasses ; i++)
      for (j=1; j <= NoClasses ; j++)
         mm(i,j).create(SetSize) ;
   
   for (i=0; i < SetSize ; i++)
   {
      probDist.init(0.0) ;

	  if  (learnRF) {
          if (opt->rfkNearestEqual>0)
		     rfNearCheck(DSet[i], probDist) ;       
		  else if (NoClasses==2 && opt->rfRegType==1)
			 rfCheckReg(DSet[i], probDist) ;
		  else 	rfCheck(DSet[i], probDist) ;
	  }
	  else check(root,DSet[i], probDist) ;      
     
      // prediction with majority class (disregarding costs)     
      //max=1;
      //for (j=2 ; j<=NoClasses ; j++)
      //{
      //   if (probDist[j] > probDist[max])
      //      max=j;
      //}
      //if (DiscData(DSet[i],0) == max)
      //   correct++ ;

      // prediction with costs

      // compute conditional risk
      minRisk = DBL_MAX ;
      cMin = 0 ;
      for (c=1; c<= NoClasses; c++)
      {
          cRisk = 0.0 ;
          for (j=1; j <= NoClasses ; j++)
             cRisk += probDist[j] * CostMatrix(c,j) ;
          if (cRisk  < minRisk) {
              minRisk = cRisk ;
              cMin = c ;
          }
      }
      if (DiscData(DSet[i],0) == cMin)
         correct++ ;
      Cost += CostMatrix(DiscData(DSet[i],0), cMin) ;
      PredictionMatrix(DiscData(DSet[i],0), cMin) ++ ;

     // compute information score
     pClPrior = AttrDesc[0].valueProbability[DiscData(DSet[i],0)] ;
     pClPrior = Mmax(epsilon, Mmin(pClPrior, 1.0 - epsilon)) ; // computational correction, if neccessary
     pClObtain = probDist[DiscData(DSet[i],0)] ; 
     pClObtain = Mmax(epsilon, Mmin(pClObtain, 1-epsilon)) ; // computational correction, if neccessary
     if (pClObtain >= pClPrior)     {
        infi += ( -log2(pClPrior) + log2(pClObtain) ) ;
     }
     else    {
        infi -= ( -log2(1.0 - pClPrior) + log2(1.0 - pClObtain) ) ;
     }


     // AUC, M measure
     for (c=1; c<= NoClasses; c++)  
       mm(c, DiscData(DSet[i],0)).addEnd(probDist[c]) ;

     // probability distribution output
     if (probabilityFile != NULL)
     {
		fprintf(probabilityFile,"%d",DSet[i]+1) ;
        for (j=1 ; j<=NoClasses ; j++)
          fprintf(probabilityFile,", %f",probDist[j]) ;
        fprintf(probabilityFile,"\n") ;
     }
   }
   
   Accuracy = double(correct)/double(SetSize);
   Inf =  infi/double(SetSize) ;
   avgCost = Cost/double(SetSize) ;
   if (NoClasses == 2) {
	  sensitivity = double(PredictionMatrix(1,1))/double(PredictionMatrix(1,1)+PredictionMatrix(1,2)) ;
      specificity = double(PredictionMatrix(2,2))/double(PredictionMatrix(2,1)+PredictionMatrix(2,2)) ;
   }

    
   // AUC, M measure
   Auc = 0.0 ;
   marray<sortRec> sa(SetSize*2) ;
   sortRec tRec ;
   int k, n0, n1, s0, noPairs=0 ;
   for (i=1; i<= NoClasses; i++) {
      n0 = mm(i,i).filled() ; 
      if (n0==0) 
           continue ;
       for (j=1 ; j<=NoClasses ; j++) {
           if (i==j)
              continue ;
           n1 = mm(j,i).filled() ; 
           if (n1==0)
               continue ;
           sa.clear() ;
           for (k=0 ; k < n0 ; k++) {
             tRec.key = mm(i,i)[k] ;
             tRec.value = i ;
             sa.addEnd(tRec) ;
           }
           for (k=0 ; k < n1 ; k++) {
              tRec.key = mm(j,i)[k] ;
              tRec.value = j ;
              sa.addEnd(tRec) ;
           }
           sa.qsortAsc() ;
           s0 = 0 ;
           for (k=0 ; k < sa.filled() ; k++)
              if (sa[k].value == i)
                  s0 += k+1 ;
           Auc += double(s0 - n0*(n0+1)/2)/double(n0)/double(n1) ;
           noPairs++ ;
       }
   }
   Auc /= double(noPairs) ; // double(NoClasses*(NoClasses-1)) ;   
}


//************************************************************
//
//                      check
//                      -----
//
//        computes classification for single case
//
//************************************************************
void featureTree::check(binnode *branch, int caseIdx, marray<double> &probDist)
{
   double contValue = NAcont;
   int discValue = NAdisc;
   int i ;
   switch (branch->Identification)
   {   
           case leaf:
              branch->Model.predict(branch, caseIdx, probDist) ;
              return ;
           case continuousAttribute:
                contValue = branch->Construct.continuousValue(DiscData, ContData, caseIdx) ;
                break ;
           case discreteAttribute:
                discValue = branch->Construct.discreteValue(DiscData, ContData, caseIdx) ;
                break ;
           default:
                error("featureTree::check", "invalid branch identification") ;
   }
   if ((branch->Identification == continuousAttribute && contValue == NAcont) ||
       (branch->Identification == discreteAttribute  && discValue == NAdisc) )
   {   // missing value
   
       marray<double> leftTable(probDist.len()) ;
       marray<double> rightTable(probDist.len()) ;
       
       check(branch->left, caseIdx, leftTable) ;
       check(branch->right, caseIdx, rightTable);
       
       for (i = 1; i < probDist.len() ; i++)
          probDist[i] = (leftTable[i] + rightTable[i])/2.0  ;
   }
   else
     if ((branch->Identification == continuousAttribute && contValue <= branch->Construct.splitValue)
           ||(branch->Identification == discreteAttribute &&  branch->Construct.leftValues[discValue]) )
         // going left
        check(branch->left, caseIdx, probDist) ;
      else // going right
        check(branch->right, caseIdx,probDist) ;
}


//************************************************************
//
//                      printResultsHead
//                      ----------------
//
//              prints head of results table
//
//************************************************************
void featureTree::printResultsHead(FILE *to) const
{
   fprintf(to,"\n%3s %5s %5s %5s %8s %5s %5s   %5s %8s %5s %5s",
       "idx", "#leaf","dFree","accTr","costTr","infTr","AUCtr","accTe","costTe","infTe","AUCte") ;
   if (NoClasses==2)
      fprintf(to," %5s %5s","SenTe","SpeTe") ;
   fprintf(to,  "\n") ;
   printLine(to,"-",83) ;
}


//************************************************************
//
//                      printResultLine
//                      ---------------
//
//        prints results for one tree into a single line
//
//************************************************************
void featureTree::printResultLine(FILE *to, int idx,
        int Leaves, int freedom,
        double TrainAccuracy, double TrainCost, double TrainInf, double TrainAuc,
        double TestAccuracy, double TestCost, double TestInf, double TestAuc, double TestSens, double TestSpec) const 
{
    char idxStr[32] ;
    if (idx>=0) sprintf(idxStr,"%3d",idx);
    else if (idx == -1) strcpy(idxStr,"avg") ;
    else if (idx == -2) strcpy(idxStr,"std") ;
    else strcpy(idxStr,"???") ;
       
   fprintf(to,"%3s %5d %5d %5.3f %8.3f %5.3f %5.3f   %5.3f %8.3f %5.3f %5.3f",
                           idxStr, Leaves,  freedom,
                                  TrainAccuracy,  TrainCost, TrainInf,TrainAuc,
                                  TestAccuracy,  TestCost, TestInf, TestAuc) ;
   if (NoClasses==2)
	  fprintf(to," %5.3f %5.3f", TestSens, TestSpec) ;
   fprintf(to,"\n") ;

}



//************************************************************
//
//                    printResultSummary
//                    ---------------
//
//           prints the report about domain testing
//              with current parameters on a file
//
//************************************************************
void featureTree::printResultSummary(FILE *to,
        marray<int> &Leaves, marray<int> &freedom,
        marray<double> &TrainAccuracy, marray<double> &TrainCost, marray<double> &TrainInf, marray<double> &TrainAuc,
        marray<double> &TestAccuracy, marray<double> &TestCost, marray<double> &TestInf, marray<double> &TestAuc, 
		marray<double> &TestSens, marray<double> &TestSpec) const
{
   double avgL, stdL, avgF, stdF, avgAtrain, stdAtrain, avgCtrain, stdCtrain, 
       avgItrain, stdItrain, avgUtrain, stdUtrain,
       avgAtest, stdAtest, avgCtest, stdCtest, avgItest, stdItest,avgUtest, stdUtest, avgSensTest, stdSensTest, avgSpecTest, stdSpecTest  ;

   AvgStd(Leaves, opt->numberOfSplits, avgL, stdL) ;
   AvgStd(freedom, opt->numberOfSplits, avgF, stdF) ;
   AvgStd(TrainAccuracy, opt->numberOfSplits, avgAtrain, stdAtrain) ;
   AvgStd(TrainCost, opt->numberOfSplits, avgCtrain, stdCtrain) ;
   AvgStd(TrainInf, opt->numberOfSplits, avgItrain, stdItrain) ;
   AvgStd(TrainAuc, opt->numberOfSplits, avgUtrain, stdUtrain) ;
   AvgStd(TestAccuracy, opt->numberOfSplits, avgAtest, stdAtest) ;
   AvgStd(TestCost, opt->numberOfSplits, avgCtest, stdCtest) ;
   AvgStd(TestInf, opt->numberOfSplits, avgItest, stdItest) ;
   AvgStd(TestAuc, opt->numberOfSplits, avgUtest, stdUtest) ;
   AvgStd(TestSens, opt->numberOfSplits, avgSensTest, stdSensTest) ;
   AvgStd(TestSpec, opt->numberOfSplits, avgSpecTest, stdSpecTest) ;

   
   printLine(to,"-", 83) ;
   printResultLine(to, -1, int(avgL+0.5), int(avgF+0.5),
              avgAtrain, avgCtrain, avgItrain, avgUtrain, avgAtest, avgCtest, avgItest, avgUtest, avgSensTest, avgSpecTest) ;
   printResultLine(to, -2, int(stdL+0.5), int(stdF+0.5),
              stdAtrain, stdCtrain, stdItrain, stdUtrain, stdAtest, stdCtest, stdItest, stdUtest, stdSensTest, stdSpecTest) ;
   fprintf(to, "\n\nAverages and standard deviations :") ;

   fprintf(to, "\n\nNumber of leaves: %.2f(%.2f)\n", avgL, stdL) ;
   fprintf(to, "Degrees of freedom: %.2f(%.2f)\n", avgF, stdF) ;
   fprintf(to, "Accuracy for train sample: %.2f(%.2f)\n", avgAtrain, stdAtrain) ;
   fprintf(to, "Cost per instance for train sample: %.2f(%.2f)\n", avgCtrain, stdCtrain) ;
   fprintf(to, "Information score for train sample: %.2f(%.2f)\n", avgItrain, stdItrain) ;
   fprintf(to, "Area under curve for train sample: %.2f(%.2f)\n", avgUtrain, stdUtrain) ;

   fprintf(to, "\nAccuracy for test sample  : %.2f(%.2f)\n",avgAtest, stdAtest) ;
   fprintf(to, "Cost per instance for test sample: %.2f(%.2f)\n",avgCtest, stdCtest) ;
   fprintf(to, "Information score for test sample: %.2f(%.2f)\n", avgItest, stdItest) ;
   fprintf(to, "Area under curve for test sample: %.2f(%.2f)\n", avgUtest, stdUtest) ;
   fprintf(to, "Area under curve for test sample: %.2f(%.2f)\n", avgUtest, stdUtest) ;
   if (NoClasses==2){
      fprintf(to, "Sensitivity for test sample: %.2f(%.2f)\n", avgSensTest, stdSensTest) ;
      fprintf(to, "Specificity for test sample: %.2f(%.2f)\n", avgSpecTest, stdSpecTest) ;
   }
}




// ************************************************************
//
//                      Feature2Str
//                      -----------
//
//        converts feature (a node) to a description string
//
// ************************************************************
void featureTree::Feature2Str(binnode *Node, char* const Str)
{
   Node->Construct.descriptionString(Str) ;
}


// ************************************************************
//
//                      ContDataRetriever
//                      -----------------
//
//        retrieves data from data table for model fitting procedure
//
// ************************************************************
void ContDataRetriever(double Index, double Data[], marray<int> &Mask, int DataSize)
{
   int i ;
#ifdef DEBUG
   if (Mask.len() != gFT->NoContinuous +1)
      error("ContDataRetriever","Invalid mask") ;
#endif
   int example = intRound(Index) ;
   int counter = 1 ;
   for (i=1 ; i < gFT->NoContinuous ; i++)
   {
      if (Mask[i] == 1)
      {
         Data[counter] = gFT->ContData(example,i) ;
         if (Data[counter] == NAcont)
           Data[counter] = gFT->CurrentNode->NAcontValue[i] ;
         counter ++ ;
      }
   }
   if (Mask[gFT->NoContinuous] == 1)
     Data[counter] = 1.0 ;  // also constant is needed
}



// ************************************************************
//
//                      MdlCodeLen
//                      ----------
//
//     computes codel len for optimization of the model
//
// ************************************************************
double MdlCodeLen(double parameter[], marray<int> &Mask)
{
   int i,j ;
   int modelSize = 0 ; // gFT->NoContinuous  class excluded, constant included - constant being the last
   for (i=1 ; i < Mask.len() ; i++)
      modelSize ++ ;

   // selection of attributes
   marray<double> Multinom(2,0.0) ;  
   Multinom[0] = modelSize ;
   Multinom[1] = gFT->NoContinuous - Multinom[0] ;
   Multinom.setFilled(2)  ;
   double len = multinomLog2(Multinom) ;

   long int intValue ;
   // codes for model
   int counter = 1;  // counter for real model size
   for (j=1 ; j <= gFT->NoContinuous  ; j++)
   {
      if (Mask[j] == 1)
      {
         intValue = longRound(labs(long(parameter[counter]/opt->mdlModelPrecision))) ;
         if (intValue == 0)
            len += 1.0 ;
         else 
           len += 1.0 + mdlIntEncode(intValue) ;
         counter ++ ;
      }
    }
    double prediction ;
    // codes for errors
    for (i=0 ; i < gFT->CurrentTrainSize ; i++)
    {
       prediction = 0.0 ;
       counter = 1 ;
       for (j=1 ; j <gFT->NoContinuous  ; j++)
       {
          if (Mask[j] == 1)
          {
             if (gFT->ContData((*(gFT->CurrentExamples))[i],j) == NAcont)
                prediction += parameter[counter] * gFT->CurrentNode->NAcontValue[j] ;
             else
                prediction += parameter[counter] * gFT->ContData( (*(gFT->CurrentExamples))[i], j) ;
             counter++ ;
          }
       }
       prediction += parameter[counter] ;
       intValue = longRound(labs(long(
         (gFT->ContData((*(gFT->CurrentExamples))[i],0)
                      - prediction)/opt->mdlErrorPrecision))) ;
       if (intValue == 0)
          len += 1.0 ;
       else
         len += 1.0 + mdlIntEncode(intValue)  ;
    }
    return len ;

}

