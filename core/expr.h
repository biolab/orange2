#if !defined(EXPR_H)
#define EXPR_H

#include "general.h"
#include "contain.h"

enum exprType  {majority=1, kNN=2, kNNkernel=3, simpleBayes=4} ;

struct exprNode
{
   int iMain, iAux ;
   exprNode *left, *right ;  // usually 2: left = 0, right = 1
} ;

typedef exprNode* PexprNode ;

struct binnode ; // forward definition

class estimation ;

class expr
{
//   friend class regressionTree ;
   exprType modelType ;
   exprNode *root ; // in case of constructs
   int majorClass ; // in case of major class
   marray<marray< marray<double> > > SBclAttrVal ; // in case of simple Bayes
   marray<marray<double> > SBattrVal ; 
   marray<double> SBcl ; 
   marray< marray<double> > Boundary ; // in case on simple Bayes and continuous attributes
   marray<double> equalDistance, differentDistance, CAslope ; // in case of kNN
   

   void destroy(void);
   void destroy(exprNode *node) ;
   void dup(exprNode *Source, PexprNode &Target) ;
   // void predict(binnode *treeNode, int Case, exprNode* Node) ;
   // char* descriptionString(exprNode* Node)  ;
   inline double CARamp(int AttrIdx, double distance) ;
   double DAdiff(binnode *treeNode, int AttrIdx, int I1, int I2) ;
   double CAdiff(binnode *treeNode, int AttrIdx, int I1, int I2) ;
   double examplesDistance(binnode *treeNode, int I1, int I2) ;

public:
   expr() { root = 0 ; }
   ~expr() ;
   expr(expr &Copy) ;
   void copy(const expr &Source) ;
   void operator=(const expr &Source) ;
   int operator== (expr &) { return 0 ; }
   void createMajority(int Value) ;
   void createKNN(void) ;
   void createKNNkernel(void) ;
   void createSimpleBayes(estimation &Estimator, binnode *treeNode) ;
   void predict(binnode *treeNode, int Case, marray<double> &probDist) ;
   char* descriptionString(void)  ;
   int degreesOfFreedom(void) { return 1; }

} ;

#endif

