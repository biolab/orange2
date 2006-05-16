#if !defined(FTREE_H)
#define FTREE_H

#include <stdio.h>

#include "dectree.h"
#include "contain.h"
#include "expr.h"
#include "rndforest.h"

const int MaxFeatureStrLen = 2048 ;
const double epsilon = 1e-7 ;   // computational error


enum testState { no, yes, missing } ;
class estimation ; // forward
class construct ; // forward

// class dealing with trees and forests
class featureTree: public dectree {
  friend class construct ;
  friend void ContDataRetriever(double Index, double Data[], marray<int> &Mask, int DataSize) ;
  friend double MdlCodeLen(double parameter[],  marray<int> &Mask) ;
  friend class expr ;
  friend class estimation ;

protected:
   double rootWeight ;
   binnode* CurrentNode ;
   marray<int> *CurrentExamples ;
   int CurrentTrainSize ;
   marray<int> rootDTrain ;
   int rootTrainSize ;
   marray<construct> CachedConstructs ;
   marray<forestTree> forest ;
   int rfNoSelAttr ;
   marray<double> rfA ; // forest coeficients
   double rfA0 ;

   binnode* buildTree(marray<int> &DTrain, marray<double> &pDTrain, int TrainSize, int currentDepth) ;
   void split(marray<int> &DTrain, marray<double> &pDTrain, int TrainSize,
              binnode *Node, marray<int> &LeftTrain, marray<double> &pLeftTrain, int &LeftSize, marray<int> &RightTrain,
              marray<double> &pRightTrain, int &RightSize, double &wLeft, double &wRight) ;
   boolean time2stop(binnode *Node) ;
   void createLeaf(binnode *Node) ;
   void buildModel(estimation &Estimator, binnode* Node) ;
   void check(binnode *branch, int caseIdx, marray<double> &ClassTable) ;
   void printFTree(FILE *out,  int &FeatureNo,  marray<binnode*> &FeatureNode, marray<binnode*> &ModelString, int &LeavesNo, binnode *branch, int place) ;
   void printFTreeDot(FILE *outDot,  binnode *branch, int &FeatureNo, int &LeavesNo) ;
   void Feature2Str(binnode *Node, char* const Str) ;
   double mPrune(binnode *Node) ;
   double mdlBottomUpPrune(binnode *Node) ;
   double mdlCode(binnode *Node) ;

   boolean buildConstruct(estimation &Estimator, binnode* Node, int currentDepth) ;
   boolean singleAttributeModel(estimation &Estimator, binnode* Node) ;
   double conjunct(estimation &Estimator, construct &bestConjunct, marray<construct> &stepCache, marray<double> &stepCacheEst ) ;
   double summand(estimation &Estimator, construct &bestSummand, marray<construct> &stepCache, marray<double> &stepCacheEst ) ;
   double multiplicator(estimation &Estimator, construct &bestMultiplicator, marray<construct> &stepCache, marray<double> &stepCacheEst ) ;
   int prepareAttrValues(estimation &Estimator, marray<construct> &Candidates) ;
   int prepareContAttrs(estimation &Estimator, constructComposition composition, marray<construct> &Candidates, construct& bestCandidate) ;
   void makeConstructNode(binnode* Node, estimation &Estimator, construct &Construct) ;
   void makeSingleAttrNode(binnode* Node, estimation &Estimator, int bestIdx, attributeCount bestType) ;
   void selectBeam(marray<construct> &Beam, marray<construct> &stepCache, marray<double> &stepCacheEst, marray<construct> &Candidates, estimation &Estimator, attributeCount aCount) ;
   double oobInplaceEvaluate(binnode *root, marray<int> &dSet, marray<boolean> &oobSet, mmatrix<int> &oob) ;
   binnode* buildForestTree(int TrainSize, marray<int> &DTrain, marray<double> &attrProb) ;
   double rfBuildConstruct(estimation &Estimator, binnode* Node, marray<double> &attrProb) ;
   void rfCheck(int caseIdx, marray<double> &probDist) ;
   int rfTreeCheck(binnode *branch, int caseIdx, marray<double> &probDist) ;
   void rfSplit(marray<int> &DTrain, int TrainSize, binnode* Node, marray<int> &LeftTrain, int &LeftSize, marray<int> &RightTrain, int &RightSize) ;
   void rfNearCheck(int caseIdx, marray<double> &probDist) ;
   void rfFindNearInTree(binnode *branch, int caseIdx, marray<IntSortRec> &near) ;
   binnode* rfBuildLimitedTree(int noTerminal, int TrainSize, marray<int> &DTrain, marray<double> &attrProb) ;
   void rfRevertToLeaf(binnode *Node) ;
   binnode* rfPrepareLeaf(int TrainSize, marray<int> &DTrain) ;
   void rfCheckReg(int caseIdx, marray<double> &probDist) ;
   double rfEvalA0(void);

   double oobAccuracy(mmatrix<int> &oob) ;
   void oobEvaluate(mmatrix<int> &oob) ;
   double oobMargin(mmatrix<int> &oob, marray<int> &maxOther, double &varMargin) ;
   double oobSTD(marray<int> &maxOther) ;
   void oobMarginAV(mmatrix<int> &oob, int noVal, marray<int> &origVal, 
								marray<double> &avMargin) ;
   void shuffleChange(int noValues, marray<int> &valArray) ;
   void rfRegularize() ;
   void rfRegFrprmn(double lambda, marray<double> &p, int &iter, double &fret) ;
   double rfRegEval(marray<double> &a, marray<double> &g) ;
   void rfLinmin(marray<double> &p, marray<double> &xi, int n, double &fret) ;
   double rfFunc(marray<double> &a);
   void rfmnbrak(double &ax, double &bx, double &cx, double &fa, double &fb, double &fc);
   double rfBrent(double ax, double bx, double cx, double tol, double &xmin);
   double f1dim(double x);




public:
   boolean learnRF ;
   double avgOobAccuracy, avgOobMargin, avgOobCorrelation ;

   featureTree();
   ~featureTree();
   int constructTree(); //--//
   void test(marray<int> &DSet, int SetSize, double &Accuracy, double &avgCost, double &Inf, 
             double &Auc, mmatrix<int> &PredictionMatrix, double &sensitivity, double &specificity, FILE *probabilityFile) ;
   void outDomainSummary(FILE *to) const ;
   void printResultsHead(FILE *to) const ;
   void printResultLine(FILE *to, int idx, int Leaves, int freedom,
        double TrainAccuracy, double TrainCost, double TrainInf,double TrainAuc,
        double TestAccuracy, double TestCost, double TestInf, double TestAuc, double TestSens, double TestSpec) const ;
  void printResultSummary(FILE *to, marray<int> &Leaves, marray<int> &freedom,
        marray<double> &TrainAccuracy, marray<double> &TrainCost, marray<double> &TrainInf, marray<double> &TrainAuc,
        marray<double> &TestAccuracy, marray<double> &TestCost, marray<double> &TestInf, marray<double> &TestAuc,
		marray<double> &TestSens, marray<double> &TestSpec) const ;
  void printFTreeFile(char *FileName, int idx,  int Leaves, int freedom,
        double TrainAccuracy, double TrainCost, double TrainInf,double TrainAuc,
        double TestAccuracy, double TestCost, double TestInf, double TestAuc,
        mmatrix<int> &TrainPMx, mmatrix<int> &TestPMx, double TestSens, double TestSpec) ;
   double mPrune(void) { return mPrune(root) ; }
   double mdlBottomUpPrune(void) { return mdlBottomUpPrune(root) ; }
   int buildForest(void) ;
   void rfResultHead(FILE *to) const ;
   void rfResultLine(FILE *to, int idx,
        double TrainAccuracy, double TrainCost, double TrainInf, double TrainAuc,
        double oobAccuracy, double oobMargin, double oobCorrelation, 
        double TestAccuracy, double TestCost, double TestInf, double TestAuc, double TestSens, double TestSpec) const ;
   void varImportance(marray<double> &varEval) ;
   void printAttrEval(FILE *to, marray<int> &idx, marray<marray<double> > &attrEval) ;
   void avImportance(marray<marray<double> > &avEval) ;

} ;

#endif
