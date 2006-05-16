#if !defined(OPTIONS_H)
#define OPTIONS_H

#include "general.h"
#include "contain.h"

enum splitSelectionType {FROM_FILES=0, CROSS_VALIDATION=1, STRATIFIED_CV=2, 
                         LOOCV=3, ALL_TRAINING=4, RANDOM_SPLIT=5 } ;

class Options {
public:
   Options(void) { setDefault() ; } 
   
   // command line options
   char optionFile[MaxFileNameLen] ;
   char action[MaxNameLen] ;
   
   // data options
   char domainName[MaxFileNameLen] ;
   char dataDirectory[MaxPath] ;
   char resultsDirectory[MaxPath] ;
   char NAstring[MaxNameLen] ;
   int splitIdx ;
   int numberOfSplits ;
   splitSelectionType splitSelection ;
   double trainProportion ;
   long int rndSeedSplit ;


   // building options
   double minInstanceWeight ; // minimal probability of example to take it into further consideration
   double minReliefEstimate ; // minimal ReliefF's estimation to consider attribute worthy
   int selectionEstimator, constructionEstimator ; // secondaryEstimator  ;
   
   
   // attribute evaluation
   int attrEvaluationInstances ;  // maximal examples for estimation
   boolean binaryAttributes ;  
   boolean binarySplitNumericAttributes ; // are continuous attributes' splits considered binary (or greedily discretized) in applicable measures                               
   marray<boolean> estOn;

   // ReliefF
   int ReliefIterations ; // number of ReliefF's main loops for estimation
   int kNearestEqual, kNearestExpRank  ;
   double quotientExpRankDistance ;
   double numAttrProportionEqual, numAttrProportionDifferent ;

   // stopping options
   double minNodeWeight ; // minimum number of examples in a node to split further on
   double relMinNodeWeight ; // minimal proportion of examples in a leaf to spit further
   double majorClassProportion ;

   //  models in trees
   int modelType ;  // type of models in leaves
   int kInNN ;
   double nnKernelWidth ; 
   
   // constructive induction
   int constructionMode ; // what constructs to consider
   int constructionDepth ;
   int beamSize, maxConstructSize ;
   int noCachedInNode ;

   // discretization
   int discretizationLookahead ; // number of times current discretization can be worse than the best
   int discretizationSample ;
   int bayesDiscretization ;
   int bayesEqFreqIntervals ;

   // pruning
   int selectedPruner ;
   double mEstPruning ; // parameter for m-estimate in pruning
   double mEstPrediction ;
   double mdlModelPrecision ;
   double mdlErrorPrecision ;

   // random forest options
   int rfNoTrees  ;
   int rfNoSelAttr ;
   boolean rfMultipleEst ;
   int rfkNearestEqual ;
   double rfPropWeightedTrees ;
   boolean rfPredictClass ;
   boolean rfAttrEvaluate ;
   double rfSampleProp ; 
   int rfNoTerminals ;
   int rfRegType ;
   double rfRegLambda ;
   long int rfRndSeed ;
 
   // missceleanous
   boolean printTreeInDot ;
   boolean outProbDistr ;
   char defaultEditor[MaxPath] ;

   // methods
   void setDefault(void) ;
   void processOptions(void) ;
   int readConfig(char* ConfigName) ;
   void outConfig(FILE *to) const ;
   int writeConfig(char* ConfigName) const ;
   void parseOption(char *optString, char *keyword, char *key) ;
   void assignOption(char *optString) ;
 

} ;

#endif
