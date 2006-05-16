#if !defined(DECTREE_H)
#define DECTREE_H

#include <string.h>
#include <stdio.h>

#include "general.h"
#include "expr.h"
#include "contain.h"
#include "bintree.h"

const char NAdisc = '\x00' ;
#define NAcont -1e30
//const double NAcont = -1e30 ;


//  estimators
enum estimatorConst { estReliefFkEqual=1,       
                      estReliefFexpRank=2, 
                      estReliefFbestK=3,
                      estRelief=4,
                      estInfGain=5,
                      estGainRatio=6,
                      estMdl=7,
                      estGini = 8,
                      estReliefFmyopic=9,
                      estAccuracy=10,
                      estBinAccuracy=11,
                      estReliefFmerit=12,
                      estReliefFdistance=13,
                      estReliefFsqrDistance=14,
                      estDKM=15,
                      estReliefFexpC=16,
                      estReliefFavgC=17,
                      estReliefFpe=18,
                      estReliefFpa=19,
                      estReliefFsmp=20,
                      estGainRatioC=21,
                      estDKMc=22,
                      estReliefFcostKukar=23,
                      estMDLsmp = 24
} ;

struct estDsc {
    char *brief, *dsc ;
} ;

enum discretizationConst { discrGreedy=1, discrEqFreq = 2} ;

// data needed to describe attribute
class attribute
{
public:
   char *AttributeName ;
   boolean continuous ;
   int NoValues ;
   marray<Pchar > ValueName ;
   int tablePlace ;
   boolean userDefinedDistance ;
   double DifferentDistance, EqualDistance ;
   marray<double> Boundaries ;
   marray<double> valueProbability ;
   attribute() ;
   ~attribute() ;
   void destroy(void) ;
   int operator== (attribute &) { return 1; }
   int operator< (attribute &) { return 1; }
   int operator> (attribute &) { return 1; }

} ;


// possible states regarding data needed for construct decision tree
enum datastate {empty, description, data, tree, random_forest} ;


class featureTree; // forward

//  basic class for decision trees, provides data input and output
class dectree :  public bintree
{
  friend class construct ;
  friend class estimation ;
  friend void singleEstimation(featureTree* const Tree) ;
  friend void allSplitsEstimation(featureTree* const Tree) ;
  friend void domainCharacteristics(featureTree* const Tree) ;
  friend void rf(featureTree* const Tree) ;
  
public:
   int NoAttr, NoOriginalAttr, NoContinuous, NoDiscrete ;
   mmatrix<int> DiscData ;    // discretisized data
   mmatrix<double> ContData ;   // continuous data
   marray<int> ContIdx ; // index pointing to the place in description for continuous variable
   marray<int> DiscIdx;  // index pointing to the place in description for discrete variable
   marray<double> minValue, maxValue, valueInterval ;

   mmatrix<double> CostMatrix ;  // [predicted] [true]

   void SetValueProbabilities(void) ;
   void SetDistances(void) ;
   void FreeC5(void) ;
   int names2dsc(void) ;
   int data2dat(void) ;
   void costsToCostMatrix(void) ;

   datastate state;
   marray<attribute> AttrDesc;

   int NoCases, NoTeachCases, NoTestCases, NoClasses ;
   marray<int> DTeach, DTest;


   dectree();
   ~dectree();
   int readProblem(void) ;  //--//
   int readDescription(void);
   int readCosts(void);
   int readData(void);
   int prepareDataSplits(void);
   int setDataSplit(int splitIdx);
   void clearData(void);
   void clearDescription(void);
   int writeDescription(const char* DescriptionFileName) const ;
   int writeData(const char* DataFileName) const ;
   void printEstimationHead(FILE *to) const ;
   void printEstimations(FILE *to, int splitIdx, marray<marray<double> > &Result) const;
   void printEstimationsInColumns(FILE *to, int splitIdx, marray<marray<double> > &Result) const ;
   int writeTree(const char* TreeFileName) const ;
   void writeSubTree(FILE *treeOut, binnode* Node, int tab) const ;
   int readTree(const char* TreeFileName) ;
   void readSubTree(FILE *treeIn, binnode* &Node) ;
   void printAVestimations(FILE *to, int splitIdx, marray<marray<double> > &Result) const ;
   void printAVestimationHead(FILE *to, char* methodStr) const ;
   void printAVestInColumns(FILE *to, marray<marray<double> > &Result, char *methodStr) const ;  
   void printAVestIn9Columns(FILE *to, char *methodStr,
	  marray<marray<double> > &ResultCpAp,
	  marray<marray<double> > &ResultCpAn, marray<marray<double> > &ResultCpAe,
	  marray<marray<double> > &ResultCnAp,
	  marray<marray<double> > &ResultCnAn, marray<marray<double> > &ResultCnAe,
	  marray<marray<double> > &ResultCeAp,
	  marray<marray<double> > &ResultCeAn, marray<marray<double> > &ResultCeAe
	  ) const  ;

} ;

typedef dectree* Pdectree ;
double MdlCodeLen(double parameter[], marray<int> &Mask) ;

#endif
