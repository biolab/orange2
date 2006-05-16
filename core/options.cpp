/********************************************************************
*
*   Name:              modul foptions (feature tree options)
*
*   Description:  reads the configuration file and interactively
*                 sets parameters for LFC (lookahead feature
*                 construction) tree
*
*********************************************************************/


#include <stdio.h>      // reading options and configuration file
#include <string.h>     // building menu items
#include <stdio.h>      // converting strings to doubleing point
#include <stdlib.h>
#include <time.h>

#include "general.h"

#if defined(UNIX) 
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#else
#include <process.h>
#include <errno.h>
#endif


#include "ftree.h"
#include "utils.h"
#include "menu.h"
#include "error.h"
#include "options.h"

extern char VersionString[] ;
extern int NoEstimators ;
extern estDsc estName[] ;

char keySeparators[] = "=" ;
char commentSeparators[] = "#%" ;


void Options::setDefault(void) { 
    optionFile[0] = '\0' ;
	strcpy(action, "none") ;
    domainName[0] = '\0' ;
    splitIdx = 0 ;
    strcpy(dataDirectory, "data") ;
    strcpy(resultsDirectory, "results") ;
    strcat(dataDirectory, strDirSeparator) ;  // we attach separator: / or backslash
    strcat(resultsDirectory, strDirSeparator) ;
    strcpy(NAstring,"?") ;
    numberOfSplits = 10 ;
	splitSelection = CROSS_VALIDATION ;
    trainProportion = 0.9 ;
    rndSeedSplit = -1 ;

    attrEvaluationInstances = 0 ; // means all
    binaryAttributes = FALSE ;
    binarySplitNumericAttributes = TRUE ;

    ReliefIterations = 0 ; // means all
	numAttrProportionEqual = 0.04 ;
    numAttrProportionDifferent = 0.1 ;
    minReliefEstimate = 0.0 ;
    kNearestEqual = 10 ;
    kNearestExpRank = 70 ;
    quotientExpRankDistance = 20.0 ;

	minInstanceWeight = 0.05 ;
	selectionEstimator = estReliefFexpRank ; // ReliefF with exponentialy decreasing rank 
    constructionEstimator =estReliefFexpRank; // ReliefF with distance density 

	estOn.create(NoEstimators+1,FALSE) ;
    estOn[2] = TRUE ; // ReliefFexpRank

	minNodeWeight = 5.0 ;
    relMinNodeWeight = 0.0;
    majorClassProportion = 0.97 ;


	mEstPruning = 2.0 ;
    mEstPrediction = 0.0 ;
    selectedPruner = 2 ; // m-estimate pruning
    mdlModelPrecision = 0.10 ;
    mdlErrorPrecision = 0.01 ;

	modelType = 1 ;  // majority class
    kInNN = 10 ;
    nnKernelWidth = 2.0 ;
	bayesDiscretization = 2 ; // equal frequency discretization
    bayesEqFreqIntervals = 4 ;

	constructionMode = cSINGLEattribute+cCONJUNCTION+cSUM+cPRODUCT ;  // single + conjunctions + addition + multiplication
    constructionDepth = 0 ; // 0 - no construction by default, 1- only at the root 
    beamSize=20 ;
    maxConstructSize = 3;
    noCachedInNode = 5 ;

	discretizationLookahead = 3 ;
    discretizationSample = 50 ;

    rfNoTrees = 100 ;
    rfNoSelAttr = 0; // meaning square root of the number of attributes
    rfMultipleEst = FALSE ;
    rfkNearestEqual = 30 ; 
    rfPropWeightedTrees = 0.0 ; // no weighting
    rfPredictClass = FALSE;
	rfAttrEvaluate = FALSE ;
	rfSampleProp = 0.0 ; // 0.0 = bootstrrap replication
    rfNoTerminals = 0 ; //0 = build whole tree
    rfRegType = 0 ; // no regularization
    rfRegLambda = 0.0 ; // lambda for regularization
    rfRndSeed = -1 ; // random seed for random forest

    printTreeInDot = FALSE ;
    outProbDistr = FALSE ;
    #if defined(UNIX)
       strcpy(defaultEditor, "vi") ; 
    #endif
    #if defined(MICROSOFT) || defined(BORLAND)
       strcpy(defaultEditor, "notepad.exe" ) ;
    #endif
}

// ************************************************************
//
//                 processOptions
//                 --------------
//
//           interactively lets user change
//             parameters
//
// ************************************************************
void Options::processOptions(void)
{
   char FileName[MaxFileNameLen] ;
   char *tempStr = getenv("TMP") ;
   if (tempStr != NULL)
     strcpy(FileName, tempStr) ;
   else                              
     strcpy(FileName, ".") ;
   strcat(FileName, strDirSeparator) ;
   strcat(FileName, "tmpOptions.par") ; 
   writeConfig(FileName) ;
   char CommandStr[2 * MaxFileNameLen] ;
   tempStr = getenv("EDITOR") ;
   if (tempStr != NULL)
      strcpy(CommandStr, tempStr) ;
   else 
      strcpy(CommandStr, defaultEditor) ;
   
#if defined(MICROSOFT) 
   intptr_t childUID = _spawnlp(_P_WAIT, CommandStr, CommandStr, FileName, NULL) ;
  if (childUID==-1)  {
     // error
      char buf[2048] ;
      sprintf(buf, "Cannot run editor %s because: ",CommandStr) ;
      switch (errno) {
          case E2BIG: strcat(buf,"Argument list exceeds 1024 bytes") ;
                      break ;
          case EINVAL:strcat(buf,"mode argument is invalid") ;
                      break ;
          case ENOENT:strcat(buf,"File or path is not found") ;
                      break ;
          case ENOEXEC:strcat(buf,"Specified file is not executable or has invalid executable-file format") ;
                      break ;
          case ENOMEM:strcat(buf,"Not enough memory is available to execute new process") ;
                      break ;
          case EACCES:strcat(buf,"Permission denied") ;
                      break ;
          default:strcat(buf,"unknown error code") ;
      }
      error(buf,"") ;  
  }
#endif
#if defined(CSET)
   _spawnlp(P_WAIT, CommandStr, CommandStr, FileName, NULL) ;
#endif
#if defined (BORLAND)
   spawnlp(P_WAIT, CommandStr, CommandStr, FileName, NULL) ;
#endif
#ifdef UNIX
  pid_t childUID = fork() ;
  switch (childUID)  {
     case -1:// error
             error("Cannot run the editor", CommandStr) ;  
			 break ;
	 case 0: // child
		     execlp(CommandStr, CommandStr, FileName, NULL) ;
             break ;
	 default: // parent
              waitpid(childUID, NULL, 0) ;
			  break ;
  }
  
#endif


   readConfig(FileName) ;

}


//************************************************************
//
//                      readConfig
//                      ----------
//
//      reads parameters for feature tree from given file
//
//************************************************************
int Options::readConfig(char* ConfigName)
{
	FILE *from ;
    if ((from=fopen(ConfigName,"r"))==NULL) {
        error("Cannot open configuration file ",ConfigName) ;
        return 0 ;
    }

    char buf[MaxNameLen]  ;
	while (!feof(from)) {
      fgets(buf,MaxNameLen,from) ;
      while (buf[strlen(buf)-1] == '\n' || buf[strlen(buf)-1] == '\r')
         buf[strlen(buf)-1] = '\0' ;
      strTrim(buf) ;
      if (buf[0] != '\0' && strchr(commentSeparators, buf[0])== NULL)
	     assignOption(buf) ;
	}      
    fclose(from) ;
    return 1 ;
}

//************************************************************
//
//                      writeConfig
//                      -----------
//
//      writes parameters for feature tree to given file
//
//************************************************************
int Options::writeConfig(char* ConfigName) const
{
    FILE *to ;
    if ((to=fopen(ConfigName,"w"))==NULL)
    {
       error("Cannot create configuration file ",ConfigName) ;
       return 0 ;
    }
    outConfig(to) ;
    if (ferror(to))  {
      error("Cannot write parameters to configuration file", ConfigName) ;
      fclose(to) ;
      return 0;
    }

    fclose(to) ;
    return 1 ;

}



void Options::outConfig(FILE *to) const
{
    fprintf(to, "# Options file for %s\n", VersionString) ;
	fprintf(to, "# Note the conventions:\n");
    fprintf(to, "# each option is on a separate line, the order of options is not important\n") ;
	fprintf(to, "# everything after # character is ignored\n") ;
	fprintf(to, "# if # is the first character, entire line is ignored\n") ;
    fprintf(to, "# the format of options is\n") ;
	fprintf(to, "# keyword=keyValue\n") ;
	fprintf(to, "#\n") ;

    fprintf(to, "# ---------- File and data options ----------\n") ;
    
    // Domain name
	fprintf(to,"domainName=%s  # domain name\n",domainName ) ;
    
    // Data directory
	fprintf(to,"dataDirectory=%.*s  # data directory\n", strlen(dataDirectory)-1, dataDirectory) ;

    // Results directory
	fprintf(to,"resultsDirectory=%.*s  # results directory\n", strlen(resultsDirectory)-1, resultsDirectory) ;

    // Definiton of train/test data splits 
    fprintf(to,"# Types of supported splits to training/testing data:  \n") ;
    fprintf(to,"# 0~read from files, 1~cross validation, 2~stratified cross-validation,\n") ;
    fprintf(to,"# 3~leave one out CV, 4~all data is for training, 5~random split to train/test\n") ;
	fprintf(to, "splitSelection=%d  # definiton of train/test data splits\n", splitSelection) ; 

    // Number of of iterations (data split to work on)
	fprintf(to, "numberOfSplits=%d  # number of data splits\n", numberOfSplits) ;

    // Train proportion
	fprintf(to,"trainProportion=%f  # the proportion of training instances in case of random split to train/test\n", trainProportion) ;

    // random seed for split
	fprintf(to,"rndSeedSplit=%ld  # random seed for data split determination (0~take from clock)\n", rndSeedSplit) ;

    // Split index
	fprintf(to,"splitIdx=%d  # in case of work on single split, the index of that split\n", splitIdx) ;
    
    
    // estimators

    fprintf(to, "# ---------- Estimation of attributes options ----------\n") ;

    // Treat all attributes as binary 
	fprintf(to,"binaryAttributes=%s  # treat attributes as binary\n", (binaryAttributes ? "Y" : "N")) ;

    // Treat numerical attribute splits as binary in applicable measures 
	fprintf(to,"binarySplitNumericAttributes=%s  # treat numerical attributes' splits as binary\n", (binarySplitNumericAttributes ? "Y" : "N")) ;

    // Number of examples  for estimation
	fprintf(to,"attrEvaluationInstances=%d  # number of instances for attribute evaluation (0 means all)\n", attrEvaluationInstances) ;


    // switches for estimation
    for (int estIdx = 1 ; estIdx <= NoEstimators ; estIdx++)  
	   fprintf(to, "est%s=%s  # %s\n" , estName[estIdx].brief, (estOn[estIdx] ? "Y" : "N"), estName[estIdx].dsc) ;

    
    fprintf(to, "# ---------- ReliefF options ----------\n") ;

     // number of iterations for ReliefF's estimation
	fprintf(to,"ReliefIterations=%d  # number of iterations for all variants of Relief  (0~TrainSize, -1~ln(TrainSize), -2~sqrt(TrainSize))\n",ReliefIterations) ;

    // Default proportion of continuous attribute to consider value equal
	fprintf(to,"numAttrProportionEqual=%f  # proportion of numerical attribute's range to consider values equal\n",numAttrProportionEqual) ;

    // Default proportion of continuous attribute to consider value different
	fprintf(to,"numAttrProportionDifferent=%f  # proportion of numerical attribute's range to consider values different\n",numAttrProportionDifferent) ;

    // Number of neighbours to consider - k
	fprintf(to, "kNearestEqual=%d  # number of neighbours to consider in equal k nearest evaluation\n",kNearestEqual) ;

    // Number of neighbours in  distance density estimation
	fprintf(to, "kNearestExpRank=%d  # number of neighbours to consider in exponential rank distance evaluation\n",kNearestExpRank) ;

    // Quotient in Gaussian function of distance density
	fprintf(to, "quotientExpRankDistance=%f  # quotient in exponential rank distance evaluation\n",quotientExpRankDistance) ;
    
    
    fprintf(to, "# ---------- Stopping options ----------\n") ;

    // minimal leaf's weight 
	fprintf(to,"minNodeWeight=%.2f  # minimal weight of a tree node\n", minNodeWeight) ;

    // Proportion of all examples in a node to stop
	fprintf(to,"relMinNodeWeight=%f  # minimal proportion of training instances in a tree node to stop\n",relMinNodeWeight) ;

    // Majority class proportion in a node
	fprintf(to,"majorClassProportion=%f  # proportion of majority class in a tree node\n",majorClassProportion) ;

    
    
    fprintf(to, "# ---------- Building  options ----------\n") ;

	// selected estimator
    fprintf(to, "# Available estimators: 1~ReliefF k-equal, 2~Relief with distance density, 3~ReliefF best of K,\n") ;
    fprintf(to, "#\t4~Kira's Relief, 5~Information gain, 6~Gain ratio, 7~Mdl, 8~Gini, 9~Myopic Relief, 10~Accuracy,\n") ;
	fprintf(to, "#\t11~Binarized accuracy, 12~ReliefF with merit, 13~ReliefF with distance, 14~ReliefF with squared distance,\n") ;
	fprintf(to, "#\t15~DKM, 16~ReliefF with cost and equal K nearest, 17~Relief with cost and densities\n") ;
	fprintf(to, "selectionEstimator=%d  # estimator for selection of attributes and binarization (1-%d)\n" , selectionEstimator, NoEstimators) ;

    // Minimal ReliefF's estimate of attribute to consider it further
	fprintf(to,"minReliefEstimate=%f  # in case of any Relief's variant the minimal evaluation of attribute to considerd it useful\n",minReliefEstimate) ;

	// Minimal probabillity of example to consider it
	fprintf(to,"minInstanceWeight=%.2f  # minimal weight of an instance\n",minInstanceWeight) ;

    // Type of models used in the leafs (0~point, 1~linear by MSE, 2~linear by MDL, 3~linear as in M5)
    fprintf(to, "# Available models: 1~majority class, 2~k-nearest neighbours, 3~k-nearest neighbors with kernel, 4~simple Bayes\n") ;
	fprintf(to,"modelType=%d  # type of models used in tree leaves\n", modelType) ;

    // k in k nearest neighbour models
	fprintf(to,"kInNN=%d  # number of neighbours in k-nearest neighbours models (0~all)\n", kInNN) ;

    // kernel  in kNN models
	fprintf(to,"nnKernelWidth=%f  # kernel width in k-nearest neighbours models\n", nnKernelWidth) ;

    // type of discretization for simple Bayes
	fprintf(to, "bayesDiscretization=%d  # type of discretization for naive Bayes models (1~greedy with selection estimator, 2~equal frequency)\n", bayesDiscretization) ;

	// number of intervals for equal frequency discretization for simple Bayes models
	fprintf(to, "bayesEqFreqIntervals=%d  # number of intervals in equal frequency discretization for naive Bayes models\n", bayesEqFreqIntervals) ;


    fprintf(to, "# ---------- Constructive induction options ----------\n") ;

    // which constructive operators to use
	fprintf(to,"constructionMode=%d  # constructive operators sum (1~single, 2~conjunction, 4~addition, 8~multiplication, e.g., all~1+2+4+8 i.e. 15) \n", constructionMode) ;

    // depth to which to perform  constructive induction
	fprintf(to,"constructionDepth=%d  # maximal depth (height) of the tree to do construction (0~do not do construction, 1~only at root, ...)\n", constructionDepth) ;

    // depth to which to perform  constructive induction
	fprintf(to,"noCachedInNode=%d  # number of cached attributes in each node where construction was performed\n", noCachedInNode) ;

    // construction estimator
	fprintf(to, "constructionEstimator=%d  # estimator for constructive induction (1-%d)\n" , constructionEstimator, NoEstimators) ;

    // beam size for beam search
	fprintf(to,"beamSize=%d  # size of the beam\n",beamSize) ;

    // maximal size of constructs
	fprintf(to,"maxConstructSize=%d  # maximal size of constructs\n", maxConstructSize) ;


    // Number of times current discretization can be worse than the best
	fprintf(to,"discretizationLookahead=%d  # Number of times current discretization can be worse than the best (0~try all possibilities)\n",discretizationLookahead) ;

    // Maximal number of points to try discretization (binarization)    
	fprintf(to,"discretizationSample=%d  # maximal number of points to try discretization (0 means all sensible)\n",discretizationSample) ;


    fprintf(to, "# ---------- Pruning  options ----------\n") ;

    // selected pruner
	fprintf(to, "selectedPruner=%d  # pruning method used (0~none, 1~MDL, 2~with m-estimate)\n", selectedPruner) ;
   
    // Precision of model coefficients in MDL pruning procedure
	fprintf(to, "mdlModelPrecision=%f  # precision of model coefficients in MDL pruning\n",mdlModelPrecision) ;

    // Precision of error coefficients in MDL 
	fprintf(to, "mdlErrorPrecision=%f  # precision of errors in MDL pruning\n",mdlErrorPrecision) ;
    
    // m - estimate for pruning
	fprintf(to,"mEstPruning=%f  # m-estimate for pruning\n",mEstPruning) ;


    fprintf(to, "# ---------- Random forest options ----------\n") ;

    // number of trees in forest
	fprintf(to,"rfNoTrees=%d  # number of trees in the random forest\n",rfNoTrees) ;

    // Number of randomly selected attributes in the node
	fprintf(to,"rfNoSelAttr=%d  # number of randomly selected attributes in the node (0~sqrt(numOfAttr), -1~log_2(numOfAttr)+1, -2~all)\n",rfNoSelAttr) ;

    // Use multiple estimators
	fprintf(to,"rfMultipleEst=%s  # use multiple estimators in the forest\n",(rfMultipleEst ? "Y" : "N")) ;

    // Number of nearest instances for weighted rf classification
	fprintf(to,"rfkNearestEqual=%d  # number of nearest intances for weighted random forest classification (0~no weighting)\n",rfkNearestEqual) ;

    // Proportion of trees where attribute probabilities are weighted with ReliefF
	fprintf(to,"rfPropWeightedTrees=%f  # proportion of trees where attribute probabilities are weighted\n",rfPropWeightedTrees) ;

    // Predict with majority class, otherwise use class distribution
	fprintf(to,"rfPredictClass=%s  # predict with majority class (otherwise with class distribution)\n",(rfPredictClass ? "Y" : "N")) ;

    // Evaluate attributes with out-of-bag evaluation
	fprintf(to,"rfAttrEvaluate=%s  # evaluate attributes with random forest out-of-bag evaluation\n",(rfAttrEvaluate ? "Y" : "N")) ;

	// Proportion of the training examples to be used in learning (0.0~bootstrap replication)
	fprintf(to,"rfSampleProp=%f  #proportion of the training set to be used in learning (0.0~bootstrap replication)\n",rfSampleProp) ;
    
	// Number of leaves in the individual trees (0-build a whole tree)
	fprintf(to,"rfNoTerminals=%d  # number of leaves in each tree (0~build the whole tree)\n",rfNoTerminals) ;

	// Type of regularization (0~no regularization, 1~global regularization, 2~local regularization)
	fprintf(to,"rfRegType=%d  # type of regularization (0~no regularization, 1~global regularization, 2~local regularization)\n",rfRegType) ;

	// Regularization parameter Lambda
	fprintf(to,"rfRegLambda=%f  # regularization parameter lambda\n",rfRegLambda) ;
	
    // random seed for forest
	fprintf(to,"rfRndSeed=%ld  # random seed for random forest (0~take from clock)\n", rfRndSeed) ;


    fprintf(to, "# ---------- Other  options ----------\n") ;
    
	// m - estimate for prediction
	fprintf(to,"mEstPrediction=%f  # m-estimate for prediction\n",mEstPrediction) ;
    
    // print tree also in dot format
	fprintf(to,"printTreeInDot=%s  # print tree also in dot format\n", (printTreeInDot ? "Y" : "N")) ;

    // output probability distribution
	fprintf(to,"outProbDistr=%s  # output class probability distribution for predicted instances\n", (outProbDistr ? "Y" : "N")) ;

	// Editor for options
	fprintf(to,"defaultEditor=%s  # editor for options file\n",defaultEditor) ;

    // Missing values indicator
	fprintf(to,"NAstring=%s  # string indicating missing value",NAstring) ;

 }



 void Options::parseOption(char *optString, char *keyword, char *key) {
    int strIdx = 0 ;
    strTrim(optString) ;
    char *token = myToken(optString, strIdx, keySeparators);
    strcpy(keyword, token) ;
    strTrim(keyword) ;
    token = myToken(optString, strIdx, commentSeparators);
	strcpy(key, token) ;
	strTrim(key) ;
 }
	
 
 void Options::assignOption(char *optString) {
    char  keyword[MaxNameLen], key[MaxNameLen], errBuf[MaxNameLen];
	int temp ;
	double dtemp ;

	parseOption(optString, keyword, key) ;
	
	// data options

	if (strcmp(keyword, "action")==0 || strcmp(keyword, "a")==0) {
		strcpy(action, key) ;
	}
	else if (strcmp(keyword, "domainName")==0) {
		// domain name
		strcpy(domainName, key) ;
	}
	else if (strcmp(keyword, "dataDirectory")==0) {
        // data directory
        strcpy(dataDirectory,key) ;
        temp = strlen(dataDirectory) ;
       if (dataDirectory[temp-1] != DirSeparator) {
          dataDirectory[temp] = DirSeparator ;
          dataDirectory[temp+1] = '\0' ;
       }
	}
	else if (strcmp(keyword, "resultsDirectory")==0) {
       // Results directory
       strcpy(resultsDirectory, key) ;
       temp = strlen(resultsDirectory) ;
       if (resultsDirectory[temp-1] != DirSeparator) {
          resultsDirectory[temp] = DirSeparator ;
          resultsDirectory[temp+1] = '\0' ;
       }
	}
	else if (strcmp(keyword, "splitSelection")==0) {
       // Definiton of train/test data splits 
       sscanf(key, "%d", &temp) ;
       if (temp >= 0 && temp <=5)
        splitSelection = (splitSelectionType)temp ;
       else
		   error("splitSelection (definiton of train/test data splits) should be one of supported (0-5)", "") ;
	}
	else if (strcmp(keyword, "numberOfSplits")==0) {
       // Number of data splits to work on
       sscanf(key,"%d", &temp) ;
       if (temp > 0)
 	  	 numberOfSplits = temp ;
       else
	 	 error("numberOfSplits (number of data splits) should be positive", "") ;
	}
	else if (strcmp(keyword, "trainProportion")==0) {
       // train proportion
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp > 0.0 && dtemp < 1.0)
          trainProportion = dtemp ;
       else
          error("trainProportion (the proportion of training instances in random split) should be between 0 and 1","") ;
	}
	else if (strcmp(keyword, "rndSeedSplit")==0) {
       // Random seed for data splits
       sscanf(key,"%ld", &rndSeedSplit) ;
       if (rndSeedSplit == 0)
           rndSeedSplit = -(long)time(NULL) ;
	}
	else if (strcmp(keyword, "splitIdx")==0) {
       // Split index
       sscanf(key, "%d", &temp) ;
       if (temp>=0)
          splitIdx = temp ;
       else
          error("splitIdx (data split index) should be positive", "") ;
	}
    
	// Estimator options
	else if (strcmp(keyword, "binaryAttributes")==0) {
	   // Treat all attributes as binary  
       if (key[0] == 'y' || key[0] == 'Y')
          binaryAttributes = TRUE ;
       else if (key[0] == 'n' || key[0] == 'N')
           binaryAttributes = FALSE ;
       else 
		   error("binaryAttributes (treat attributes as binary) should be on or off (Y or N)", "") ;
	}
    else if (strcmp(keyword, "binarySplitNumericAttributes")==0) {
	    // Treat continuous attribute splits as binary in applicable measures 
        if (key[0] == 'y' || key[0] == 'Y')
           binarySplitNumericAttributes = TRUE ;
        else if (key[0] == 'n' || key[0] == 'N')
           binarySplitNumericAttributes = FALSE ;
        else 
			error("binarySplitNumericAttributes (treat numerical attributes' splits as binary) should be on or off (Y or N)", "") ;
	}
    else if (strcmp(keyword, "attrEvaluationInstances")==0) {
       // number of examples  for attribute estimations
       sscanf(key,"%d", &temp) ;
       if (temp >= 0)
         attrEvaluationInstances = temp ;
       else
          error("attrEvaluationInstances (number of instances for attribute evaluation) should be non-negative", "") ;
	}
	else {
     // switches for estimation
        boolean estSwitch = FALSE ;
		char estKeyword[MaxNameLen] ;
        for (int estIdx = 1 ; estIdx <= NoEstimators ; estIdx++)  {
		   sprintf(estKeyword, "est%s",estName[estIdx].brief) ;
		   if (strcmp(keyword, estKeyword)==0) {
			   estSwitch =TRUE ;
               if (key[0] == 'y' || key[0] == 'Y')
                  estOn[estIdx] = TRUE ;
               else  if (key[0] == 'n' || key[0] == 'N')
                  estOn[estIdx] = FALSE ;
               else {
                 sprintf(errBuf, "est%s (attribute estimator \"%s\") should be on (y, Y) or off (n, N)", estName[estIdx].brief, estName[estIdx].dsc) ;
                 error(errBuf, "") ;
               }
			   break ;
		   }
		}
	if (!estSwitch) {

	//  ReliefF options

	if (strcmp(keyword, "ReliefIterations")==0) {
       // Number of iterations in ReliefF's main loop
       sscanf(key,"%d", &temp) ;
       if (temp >= -2)
          ReliefIterations = temp ;
       else
          error("ReliefIterations (number of iterations for all variants of Relief) should be larger or equal to -2", "") ;
	}
    else if (strcmp(keyword, "numAttrProportionEqual")==0) {
       // numerical attribute proportion equal
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp >= 0.0 && dtemp <= 1.0)
          numAttrProportionEqual = dtemp ;
       else
          error("numAttrProportionEqual (proportion of numerical attribute's range to consider values equal) should be between 0 and 1","") ;
	}
    else if (strcmp(keyword, "numAttrProportionDifferent")==0) {
       // numAttrProportionDifferent
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp >= 0 && dtemp <= 1.0)
          numAttrProportionDifferent = dtemp ;
       else 
          error("numAttrProportionDifferent (proportion of numerical attribute's range to consider values different) should be between 0 and 1","") ;
	}
    else if (strcmp(keyword, "kNearestEqual")==0) {
       // Number of neighbours to consider - kEqual
       sscanf(key,"%d", &temp) ;
       if (temp >= 0)
          kNearestEqual = temp ;
       else
          error("kNearestEqual (number of neighbours to consider in equal k nearest evaluation) should be nonnegative","") ;
	} 
    else if (strcmp(keyword, "kNearestExpRank")==0) {
       // Number of neighbours to consider - kExpRank
       sscanf(key,"%d", &temp) ;
       if (temp >= 0)
          kNearestExpRank = temp ;
       else
          error("kNearestExpRank (number of neighbours to consider in exponential rank distance evaluation) should be nonnegative","") ;
	} 
    else if (strcmp(keyword, "quotientExpRankDistance")==0) {
       // quotient in Gaussian function at exponential rank distance
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp > 0.0 )
          quotientExpRankDistance = dtemp ;
       else
           error("quotientExpRankDistance (quotient in exponential rank distance evaluation) should be positive", "") ;
	}
     
	// stoping options

    else if (strcmp(keyword, "minNodeWeight")==0) {
       // Minimal weight of a node to split
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp >= 0.0)
         minNodeWeight = dtemp ;
       else
          error("minNodeWeight (minimal weight of a tree node) should be non-negative","") ;
	}
    else if (strcmp(keyword, "relMinNodeWeight")==0) {
       // Proportion of all examples in a node to stop
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp >= 0.0 && dtemp <=1.0)
          relMinNodeWeight = dtemp ;
       else
          error("relMinNodeWeight (minimal proportion of training instances in a tree node) should be between 0 and 1","") ;
	}
    else if (strcmp(keyword, "majorClassProportion")==0) {    
       // Majority class proportion in a node
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp >= 0.0 && dtemp <=1.0)
          majorClassProportion = dtemp ;
       else
         error("majorClassProportion (proportion of majority class in a tree node) should be between 0 and 1", "") ;
	}

    // Building options

    else if (strcmp(keyword, "selectionEstimator")==0) {    
	   // selection estimator
       sscanf(key,"%d", &temp) ;
       if (temp > 0 && temp <= NoEstimators)
         selectionEstimator = temp ;
	   else {
         sprintf(errBuf, "selectionEstimator (estimator for selection of attributes and binarization) should be one of existing (1-%d)", NoEstimators) ;
		 error(errBuf, "") ;
	   }
	}
    else if (strcmp(keyword, "minReliefEstimate")==0) {     
      // Minimal ReliefF's estimate of attribute to consider it further
      sscanf(key,"%lf", &dtemp) ;
      if (dtemp >= -1.0 && dtemp <= 1.0)
         minReliefEstimate = dtemp ;
      else
        error("minReliefEstimate (minimal Relief's estimate of an attribute) should be in [-1, 1]", "") ;
	}
    else if (strcmp(keyword, "minInstanceWeight")==0) {     
  	   // Minimal weight of an instance
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp > 0.0 && dtemp <= 1.0)
         minInstanceWeight = dtemp ;
       else
         error("minInstanceWeight (minimal weight of an instance) should be between 0 and 1", "") ;
	}
    else if (strcmp(keyword, "modelType")==0) {     
       // Type of models used in the leafs (1-majority class, 2-kNN, 3-kNN with kernel, 4-simple Bayes):
       sscanf(key,"%d", &temp) ;
       if (temp >= 1 && temp <= 4)
         modelType = temp ;
       else
         error("modelType (type of models used in the leafs) should be 1-4", "") ;
	}
    else if (strcmp(keyword, "kInNN")==0) {     
       // k in kNN models
       sscanf(key,"%d", &temp) ;
       if (temp >= 0 )
         kInNN = temp ;
       else
         error("kInNN (number of neighbours in k-nearest neighbours models) should be positive", "") ;
	}
    else if (strcmp(keyword, "nnKernelWidth")==0) {     
       // kernel width in kNN models
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp > 0.0)
          nnKernelWidth = dtemp ;
       else
         error("nnKernelWidth (kernel width in k-nearest neighbours models) should be positive","") ;
	}
    else if (strcmp(keyword, "bayesDiscretization")==0) {     
       // type or discretizationn for simple bayes
       sscanf(key,"%d", &temp) ;
       if (temp >= 1 && temp <= 2)
          bayesDiscretization = temp ;
       else
         error("bayesDiscretization (discretization for naive Bayes) should be 1 or 2", "") ;
	}
    else if (strcmp(keyword, "bayesEqFreqIntervals")==0) {     
  	   // number of intervals for equal frequency discretization for simple Bayes models
       sscanf(key,"%d", &temp) ;
       if (temp > 1)
         bayesEqFreqIntervals = temp ;
       else
          error("bayesEqFreqIntervals (number of intervals in equal-frequency discretization for naive Bayes models) should be greater than 1", "") ;
	}
    // Constructive induction options

    else if (strcmp(keyword, "constructionMode")==0) {     
  	   // which constructive operators to use
       sscanf(key,"%d", &temp) ;
       if (temp >= 0 && temp <= cSINGLEattribute+cCONJUNCTION+cSUM+cPRODUCT)
           constructionMode = temp | cSINGLEattribute ;  // cSINGLEattribute MUST be included
       else
          error("constructionMode (selection of construction operatorts) contains unknown operators", "") ;
	}
    else if (strcmp(keyword, "constructionDepth")==0) {     
       // depth  constructive induction
       sscanf(key,"%d", &temp) ;
       if (temp >= 0)
          constructionDepth = temp ;
       else
          error("constructionDepth (depth of the tree where construction is applied) should be non-negative", "") ;
	}
    else if (strcmp(keyword, "noCachedInNode")==0) {     
       // how many attributes to cache in each node
       sscanf(key,"%d", &temp) ;
       if (temp >= 0)
          noCachedInNode = temp ;
       else
           error("noCachedInNode (number of cached constructs in eaach construction node) should be non-negative", "") ;
	}
    else if (strcmp(keyword, "constructionEstimator")==0) {     
       // construction estimator
       sscanf(key,"%d", &temp) ;
       if (temp > 0 && temp <= NoEstimators)
          constructionEstimator = temp ;
	   else {
         sprintf(errBuf, "constructionEstimator (estimator of constructs) should be one of existing (1-%d)", NoEstimators) ;
		 error(errBuf, "") ;
	   }
	}
    else if (strcmp(keyword, "beamSize")==0) {     
       // beam size for beam search
       sscanf(key,"%d", &temp) ;
       if (temp > 0 )
          beamSize = temp ;
       else
          error("beamSize (size of the beam in constructive induction) should be greater than 0", "") ;
	}
    else if (strcmp(keyword, "maxConstructSize")==0) {     
      // maximal size of constructs
      sscanf(key,"%d", &temp) ;
      if (temp > 0 )
         maxConstructSize = temp ;
      else
         error("maxConstructSize (maximal size of constructs) should be greater than 0", "") ;
	}
    else if (strcmp(keyword, "discretizationLookahead")==0) {     
       // Number of times current discretization can be worse than the best
       sscanf(key,"%d", &temp) ;
       if (temp >= 0)
         discretizationLookahead = temp ;
       else
         error("discretizationLookahead (number of times current discretization can be worse than the best) should be non-negative", "") ;
	}
    else if (strcmp(keyword, "discretizationSample")==0) {     
       // Maximal number of points to try discretization with Relief
      sscanf(key,"%d", &temp) ;
      if (temp >= 0)
        discretizationSample = temp ;
      else
         error("discretizationSample (maximal number of points to try discretization with Relief) should be non-negative","") ;
	}

    // Pruning options

    else if (strcmp(keyword, "selectedPruner")==0) {     
       // selected pruner (0-none, 1-MDL, 2-m-estimate)
       sscanf(key,"%d", &temp) ;
       if (temp >= 0 && temp <= 2)
          selectedPruner = temp ;
       else
         error("selectedPruner (the tree pruning method) should be one of existing (0-2)", "") ;
	}
    else if (strcmp(keyword, "mdlModelPrecision")==0) {     
      // Precision of the model coefficients in MDL 
      sscanf(key,"%lf", &dtemp) ;
      if (dtemp > 0.0 )
         mdlModelPrecision = dtemp ;
      else
         error("mdlModelPrecision (precision of the model coefficients in MDL pruning) should be positive","") ;
	}
    else if (strcmp(keyword, "mdlErrorPrecision")==0) {     
      // Precision of the error in MDL 
      sscanf(key,"%lf", &dtemp) ;
      if (dtemp > 0.0 )
         mdlErrorPrecision = dtemp ;
      else
         error("mdlErrorPrecision (precision of the error in MDL pruning) should be positive","") ;
	}
    else if (strcmp(keyword, "mEstPruning")==0) {     
       // m - estimate for pruning
       sscanf(key,"%lf", &dtemp) ;
       if (dtemp>=0)
         mEstPruning = dtemp ;
       else
          error("mEstPruning (m-estimate for pruning) should be non-negative","") ;
	}

    // Random forests options
    
    else if (strcmp(keyword, "rfNoTrees")==0) {     
      // number of trees in the forest
      sscanf(key,"%d", &temp) ;
      if (temp>0)
        rfNoTrees = temp ;
      else
         error("rfNoTrees (number of trees in the random forest) should be positive","") ;
	}
    else if (strcmp(keyword, "rfNoSelAttr")==0) {     
      // Number of randomly selected attributes in the node
      sscanf(key,"%d", &temp) ;
      if (temp>=-2)
         rfNoSelAttr = temp ;
      else
        error("rfNoSelAttr (number of randomly selected attributes in tree nodes) should be >=-2","") ;
	}
    else if (strcmp(keyword, "rfMultipleEst")==0) {     
      // Use multiple estimators
	  if (key[0] == 'y' || key[0] == 'Y')
	     rfMultipleEst = TRUE ;
	  else if (key[0] == 'n' || key[0] == 'N')
	     rfMultipleEst = FALSE ;
	  else 
		  error("rfMultipleEst (use of multiple estimators in the forest) should be on or off (Y or N)","") ;
	}
    else if (strcmp(keyword, "rfkNearestEqual")==0) {        
      // Number of nearest instances for weighted rf classification
      sscanf(key,"%d", &temp) ;
      if (temp>=0)
        rfkNearestEqual = temp ;
      else
        error("rfkNearestEqual (number of nearest instances for random forest weighting) should be nonnegative","") ;
	}
    else if (strcmp(keyword, "rfPropWeightedTrees")==0) {        
      // proportion of trees where attribute probabilities are weighted with ReliefF
	  sscanf(key,"%lf", &dtemp) ;
      if (dtemp >=0 && dtemp <= 1.0)
	    rfPropWeightedTrees = dtemp ;
	  else   
		error("rfPropWeightedTrees (proportion of trees where attribute probabilities are weighted with ReliefF) should be between 0 and 1","") ;
	}
    else if (strcmp(keyword, "rfPredictClass")==0) {        
      // Predict with majority class, otherwise use class distribution
	  if (key[0] == 'y' || key[0] == 'Y')
	    rfPredictClass = TRUE ;
	  else if (key[0] == 'n' || key[0] == 'N')
	    rfPredictClass = FALSE ;
	  else 
		  error("rfPredictClass (predict with majority class) should be on or off (Y or N)", "") ;
	}
    else if (strcmp(keyword, "rfAttrEvaluate")==0) {        
      // Evaluate attributes with out-of-bag evaluation
	  if (key[0] == 'y' || key[0] == 'Y')
	    rfAttrEvaluate = TRUE ;
	  else if (key[0] == 'n' || key[0] == 'N')
	     rfAttrEvaluate = FALSE ;
	  else 
		  error("rfAttrEvaluate (evaluate attributes with random forest out-of-bag evaluation) should be on or off (Y or N)", "") ;
	}
    else if (strcmp(keyword, "rfSampleProp")==0) {            
	   // proportion of the training examples to be used in learning (0.0-bootstrap replication)
	   sscanf(key,"%lf", &dtemp) ;
       if (dtemp >= 0.0 && dtemp <= 1.0)
	     rfSampleProp = dtemp ;
	   else 
		  error("rfSampleProp (proportion of the the training examples to be used in learning should be between 0.0 and 1.0", "") ;
	}
    else if (strcmp(keyword, "rfNoTerminals")==0) {            
      // Number of leaves in the individual trees (0-build a whole tree)
      sscanf(key,"%d", &temp) ;
      if (temp >= 0)
         rfNoTerminals = temp ;
      else 
		 error("rfNoTerminals (number of leaves in each tree) should be nonnegative","") ;
	}
	// Type of regularization (0-no regularization, 1-global regularization, 2-local regularization)
    else if (strcmp(keyword, "rfRegType")==0) {            
       sscanf(key,"%d", &temp) ;
       if (temp >= 0 && temp <= 2)
          rfRegType = temp ;
       else 
		  error("rfRegType (type of regularization) should be 0, 1, or 2","") ;
	}
    else if (strcmp(keyword, "rfRegLambda")==0) {            
  	  // Regularization parameter Lambda
	  sscanf(key,"%lf", &dtemp) ;
      if (dtemp >= 0.0)
	     rfRegLambda = dtemp ;
	  else 
		 error("rfRegLambda (regularization parameter lambda) should be nonnegative", "") ;
	}
    else if (strcmp(keyword, "rfRndSeed")==0) {            
       // Random seed for random forests
       sscanf(key,"%ld", &rfRndSeed) ;
       if (rfRndSeed == 0)
          rfRndSeed = -(long)time(NULL) ;
	}
    
	// Other options 
    
    else if (strcmp(keyword, "mEstPrediction")==0) {            
	  // m - estimate for prediction
	  sscanf(key,"%lf", &dtemp) ;
      if (dtemp>=0)
        mEstPrediction = dtemp ;
      else
        error("mEstPrediction (m-estimate for prediction) should be nonnegative","") ;
	}
    else if (strcmp(keyword, "printTreeInDot")==0) {            
 	  // print tree in dot format as well  
	  if (key[0] == 'y' || key[0] == 'Y')
		 printTreeInDot = TRUE ;
	  else if (key[0] == 'n' || key[0] == 'N')
		 printTreeInDot = FALSE ;
	  else 
		 error("printTreeInDot (print tree also in dot format) should be on or off (Y or N)", "") ;
	}
    else if (strcmp(keyword, "outProbDistr")==0) {            
	  // output class probability distribution   
	  if (key[0] == 'y' || key[0] == 'Y')
		 outProbDistr = TRUE ;
	  else if (key[0] == 'n' || key[0] == 'N')
		 outProbDistr = FALSE ;
	  else 
		 error("outProbDistr (output class probability distribution) should be on or off (Y or N)", "") ;
	}
    else if (strcmp(keyword, "defaultEditor")==0) {            
      //  default editor
      if (strlen(key) > 0)
		 strcpy(defaultEditor, key) ;
	}
    else if (strcmp(keyword, "NAstring")==0) {            
	  //  missing value indicator
      if (strlen(key) > 0)
		strcpy(NAstring, key) ;
	}
	else {
  	   error("unrecognized option", keyword) ;
	}
	}
    }
}

//************************************************************
//
//                      readConfig
//                      ----------
//
//      reads parameters for feature tree from given file
//
//************************************************************
/*
int Options::readConfigOld(char* ConfigName)
{
    FILE *from ;
    if ((from=fopen(ConfigName,"r"))==NULL)
    {
        error("Cannot open configuration file ",ConfigName) ;
        return 0 ;
    }

    int temp;
    double dtemp ;
    char buf[MaxNameLen]  ;

    // File options
    
    // Domain name
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    strcpy(domainName, buf) ;

    // Data directory
    fgetStrIgnoreTill(from,buf,'=',"#%") ; 
    strcpy(dataDirectory,buf) ;
    temp = strlen(dataDirectory) ;
    if (dataDirectory[temp-1] != DirSeparator)
    {
       dataDirectory[temp] = DirSeparator ;
       dataDirectory[temp+1] = '\0' ;
    }

    // Results directory
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    strcpy(resultsDirectory, buf) ;
    temp = strlen(resultsDirectory) ;
    if (resultsDirectory[temp-1] != DirSeparator)
    {
       resultsDirectory[temp] = DirSeparator ;
       resultsDirectory[temp+1] = '\0' ;
    }

     // Definiton of train/test data splits 
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%d", &temp) ;
     if (temp >= 0 && temp <=5)
        splitSelection = (splitSelectionType)temp ;
      else
          error("Definiton of train/test data splits should be one of supported (0-5) in file ", ConfigName) ;

    // Number of iteraions (data splits) to work on
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp > 0)
 		numberOfSplits = temp ;
    else
       error("Number of iterations (data splits) should be positive in file", ConfigName) ;

    // Train proportion
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%lf", &dtemp) ;
    if (dtemp > 0.0 && dtemp < 1.0)
        trainProportion = dtemp ;
    else
        error("The proportion of training instances in random split should be between 0 and 1 in file",ConfigName) ;

    // Random seed for data splits
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%ld", &rndSeedSplit) ;
    if (rndSeedSplit == 0)
        rndSeedSplit = -(long)time(NULL) ;

    // Split index
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf, "%d", &temp) ;
    if (temp>=0)
       splitIdx = temp ;
    else
       error("Split index should be positive in file",ConfigName) ;



     // Estimator options

	 // Treat all attributes as binary  
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     if (buf[0] == 'y' || buf[0] == 'Y')
         binaryAttributes = TRUE ;
      else
        if (buf[0] == 'n' || buf[0] == 'N')
         binaryAttributes = FALSE ;
        else 
          error("The switch \"treat attributes as binary\" should be on or off (Y or N) in file ", ConfigName) ;
     	 
     // Treat continuous attribute splits as binary in applicable measures 
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     if (buf[0] == 'y' || buf[0] == 'Y')
         binarySplitNumericAttributes = TRUE ;
      else
        if (buf[0] == 'n' || buf[0] == 'N')
         binarySplitNumericAttributes = FALSE ;
        else 
          error("The switch \"treat continuous attributes' splits as binary\" should be on or off (Y or N) in file ", ConfigName) ;

     // Maximal number of examples  for all estimations
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%d", &temp) ;
     if (temp >= 0)
       attrEvaluationInstances = temp ;
     else
       error("Maximal number of examples for estimation should be non-negative in file", ConfigName) ;


     // switches for estimation
     for (int estIdx = 1 ; estIdx <= NoEstimators ; estIdx++)  {
        fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
        if (buf[0] == 'y' || buf[0] == 'Y')
           estOn[estIdx] = TRUE ;
        else
            if (buf[0] == 'n' || buf[0] == 'N')
              estOn[estIdx] = FALSE ;
            else {
                sprintf(buf, "Estimator %d (%s) should be on (y, Y) or off (n, N) in file", estIdx, estName[estIdx].brief) ;
              error(buf, ConfigName) ;
            }
     }

     //  ReliefF options

     // Number of iterations in ReliefF's main loop
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= -2)
       ReliefIterations = temp ;
     else
       error("Number of iterations for ReliefF's estimations should be larger or equal -2 in file", ConfigName) ;

     // numAttrProportionEqual
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%lf", &dtemp) ;
     if (dtemp >= 0.0 && dtemp <= 1.0)
        numAttrProportionEqual = dtemp ;
     else
        error("Default proportion of continuous attribute range to consider value equal should be between 0 and 1 in file",ConfigName) ;

     // numAttrProportionDifferent
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%lf", &dtemp) ;
     if (dtemp >= 0 && dtemp <= 1.0)
        if (dtemp >= numAttrProportionEqual)
          numAttrProportionDifferent = dtemp ;
        else
        {
           numAttrProportionDifferent = numAttrProportionEqual ;
           error("Default proportion of continuous attribute range to consider value different should be greater or equal to  equal proportion in file",ConfigName) ;
        }
     else
        error("Default proportion of continuous attribute range to consider value different should be between equal proportion and 1 in file",ConfigName) ;

     // Number of neighbours to consider - k
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0)
       kNearestEqual = temp ;
     else
       error("Number of neighbours to consider in all-equal estimation", ConfigName) ;

     // Number of neighbours at density estimation
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0)
       kNearestExpRank = temp ;
     else
       error("Number of neighbours in distance density estimation should be positive in file", ConfigName) ;

     // Quotient in Gaussian function at distance density estimation
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%lf", &dtemp) ;
     if (dtemp > 0.0 )
        quotientExpRankDistance = dtemp ;
     else
        error("Quotient in Gaussian function at distance density estimation should be positive in file",ConfigName) ;

     // stoping options

     // Minimal weight of a node to split
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%lf", &dtemp) ;
     if (dtemp >= 0.0)
        minNodeWeight = dtemp ;
     else
        error("Minimal weight of a node should be non-negative in file",ConfigName) ;

     // Proportion of all examples in a node to stop
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%lf", &dtemp) ;
     if (dtemp >= 0.0 && dtemp <=1.0)
        relMinNodeWeight = dtemp ;
     else
        error("Proprtion of examples in a node should be between 0 and 1 in file", ConfigName) ;

    
     // Majority class proportion in a node
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%lf", &dtemp) ;
     if (dtemp >= 0.0 && dtemp <=1.0)
        majorClassProportion = dtemp ;
     else
        error("\nMajority class proportion in a node should be between 0 and 1 in file", ConfigName) ;


     // Building options

	// selection estimator
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp > 0 && temp <= NoEstimators)
       selectionEstimator = temp ;
     else
       error("Estimator for selection of attributes and binarization should be one of existing (1..NoEstimators) in file", ConfigName) ;
     
    // Minimal ReliefF's estimate of attribute to consider it further
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%lf", &dtemp) ;
     if (dtemp >= -1.0 && dtemp <= 1.0)
        minReliefEstimate = dtemp ;
     else
        error("Minimal ReliefF's estimate of attribute should be non-negative in file",ConfigName) ;

	 // Minimal probabillity of example to consider it
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%lf", &dtemp) ;
     if (dtemp > 0.0 && dtemp <= 1.0)
        minInstanceWeight = dtemp ;
     else
        error("Minimal probabillity of example should be between 0 and 1 in file",ConfigName) ;

     // Type of models used in the leafs (1-majority class, 2-kNN, 3-kNN with kernel, 4-simple Bayes):
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 1 && temp <= 4)
       modelType = temp ;
    else
       error("Type of models used in the leafs should be 1, 2, 3, or 4 in file", ConfigName) ;

   // k in kNN models
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0 )
       kInNN = temp ;
    else
       error("Number of neighbours in k-nearest neighbours models should be nonnegative in file", ConfigName) ;

   // kernel in kNN models
   fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
   sscanf(buf,"%lf", &dtemp) ;
    if (dtemp > 0.0)
        nnKernelWidth = dtemp ;
    else
       error("Kernel in k-nearest neighbours models should be positive in file", ConfigName) ;

    // type or discretizationn for simple bayes
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 1 && temp <= 2)
       bayesDiscretization = temp ;
     else
       error("Discretization for simple Bayes should be one of existing (1 or 2) in file", ConfigName) ;

  	// number of intervals for equal frequency discretization for simple Bayes models
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp > 1)
       bayesEqFreqIntervals = temp ;
     else
       error("Number of intervals in equal frequency discretization for simple Bayes models should be greater than 1 in file", ConfigName) ;


    // Constructive induction options

    // which constructive operators to use
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0 && temp <= cSINGLEattribute+cCONJUNCTION+cSUM+cPRODUCT)
    {
       constructionMode = temp | cSINGLEattribute ;  // cSINGLEattribute MUST be included
    }
    else
       error("Unknown construction operators  in file", ConfigName) ;

    // where to perform  constructive induction
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0)
       constructionDepth = temp ;
    else
       error("Depth of the tree to do the construction should be non-negative in file", ConfigName) ;

    // how many attributes to cache in each node
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0)
       noCachedInNode = temp ;
    else
       error("Number of cached constructs in eaach construction node should be non-negative in file", ConfigName) ;

    // construction estimator
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp > 0 && temp <= NoEstimators)
       constructionEstimator = temp ;
     else
       error("Estimator for constructive induction should be one of existing (1..NoEstimators) in file", ConfigName) ;

    // beam size for beam search
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp > 0 )
       beamSize = temp ;
    else
       error("Size of the beam in constructive induction should be greater than 0 in file", ConfigName) ;

    // maximal size of constructs
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp > 0 )
       maxConstructSize = temp ;
    else
       error("Maximal size of the constructs should be greater than 0 in file", ConfigName) ;
    

    // Number of times current discretization can be worse than the best
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0)
       discretizationLookahead = temp ;
     else
       error("Number of times current discretization can be worse than the best should be non-negative in file", ConfigName) ;
    
     // Maximal number of points to try discretization with RReliefF
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0)
       discretizationSample = temp ;
     else
       error("Maximal number of points to try discretization with RReliefF should be non-negative in file", ConfigName) ;


    // Pruning options

    // selected pruner (0-none, 1-MDL, 2-m-estimate)
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0 && temp <= 2)
       selectedPruner = temp ;
     else
       error("Selected pruning method should be one of existing (0-2) in file", ConfigName) ;

    
     // Precision of the model coefficients in MDL 
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%lf", &dtemp) ;
     if (dtemp > 0.0 )
        mdlModelPrecision = dtemp ;
     else
        error("Precision of the model coefficients in MDL should be positive in file",ConfigName) ;

     // Precision of the error in MDL 
     fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
     sscanf(buf,"%lf", &dtemp) ;
     if (dtemp > 0.0 )
        mdlErrorPrecision = dtemp ;
     else
        error("Precision of the error in MDL should be positive in file",ConfigName) ;


     // m - estimate for pruning
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%lf", &dtemp) ;
     if (dtemp>=0)
        mEstPruning = dtemp ;
     else
        error("m-estimate for pruning should be non-negative in file",ConfigName) ;


    // Random forests options
    
    // number of trees in the forest
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp>0)
        rfNoTrees = temp ;
    else
        error("Number of trees in the forest should be positive in file ",ConfigName) ;

    // Number of randomly selected attributes in the node
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp>=-2)
        rfNoSelAttr = temp ;
    else
        error("Number of randomly selected attributes in the node should be >=-2 in file ",ConfigName) ;

    // Use multiple estimators
	fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
	if (buf[0] == 'y' || buf[0] == 'Y')
	   rfMultipleEst = TRUE ;
	else if (buf[0] == 'n' || buf[0] == 'N')
	   rfMultipleEst = FALSE ;
	else error("The use of multiple estimators in the forest should be on or off (Y or N) in file ", ConfigName) ;
    
    // Number of nearest instances for weighted rf classification
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%d", &temp) ;
    if (temp>=0)
        rfkNearestEqual = temp ;
    else
        error("Number of nearest instances for random forest weighting should be nonnegative in file ",ConfigName) ;

    // proportion of trees where attribute probabilities are weighted with ReliefF
	fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
	sscanf(buf,"%lf", &dtemp) ;
    if (dtemp >=0 && dtemp <= 1.0)
	    rfPropWeightedTrees = dtemp ;
	else   error("Proportion of trees where attribute probabilities are weighted should be between 0 and 1 in file ", ConfigName) ;

    // Predict with majority class, otherwise use class distribution
	fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
	if (buf[0] == 'y' || buf[0] == 'Y')
	   rfPredictClass = TRUE ;
	else if (buf[0] == 'n' || buf[0] == 'N')
	   rfPredictClass = FALSE ;
	else error("The switch \"Predict with majority class\" should be on or off (Y or N) in file ", ConfigName) ;

    // Evaluate attributes with out-of-bag evaluation
	fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
	if (buf[0] == 'y' || buf[0] == 'Y')
	   rfAttrEvaluate = TRUE ;
	else if (buf[0] == 'n' || buf[0] == 'N')
	   rfAttrEvaluate = FALSE ;
	else error("The switch \"Evaluate attributes with out-of-bag evaluation\" should be on or off (Y or N) in file ", ConfigName) ;

	// Proportion of the training examples to be used in learning (0.0-bootstrap replication)
	fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
	sscanf(buf,"%lf", &dtemp) ;
    if (dtemp >= 0.0 && dtemp <= 1.0)
	    rfSampleProp = dtemp ;
	else error("Proportion of the the training examples to be used in learning should be between 0.0 and 1.0 in file ", ConfigName) ;
    
    // Number of leaves in the individual trees (0-build a whole tree)
    fgetStrIgnoreTill(from,buf,'=',"#%") ; 
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0)
        rfNoTerminals = temp ;
    else error("Number of leaves in the individual trees should be nonnegative in file ",ConfigName) ;

	// Type of regularization (0-no regularization, 1-global regularization, 2-local regularization)
    fgetStrIgnoreTill(from,buf,'=',"#%") ; 
    sscanf(buf,"%d", &temp) ;
    if (temp >= 0 && temp <= 2)
        rfRegType = temp ;
    else error("Type of regularization should be 0, 1, or 2 in file ",ConfigName) ;

	// Regularization parameter Lambda
	fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
	sscanf(buf,"%lf", &dtemp) ;
    if (dtemp >= 0.0)
	    rfRegLambda = dtemp ;
	else error("Regularization parameter lambda should be larger or equal 0.0 in file ", ConfigName) ;

    // Random seed for random forests
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%ld", &rfRndSeed) ;
    if (rfRndSeed == 0)
        rfRndSeed = -(long)time(NULL) ;

    // Other options 
    // m - estimate for prediction
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    sscanf(buf,"%lf", &dtemp) ;
     if (dtemp>=0)
        mEstPrediction = dtemp ;
     else
        error("m-estimate for prediction should be non-negative in file",ConfigName) ;

	 // print tree in dot format as well  
	 fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
	 if (buf[0] == 'y' || buf[0] == 'Y')
		 printTreeInDot = TRUE ;
	  else
		if (buf[0] == 'n' || buf[0] == 'N')
		 printTreeInDot = FALSE ;
		else 
		  error("The switch \"print tree also in dot format\" should be on or off (Y or N) in file ", ConfigName) ;

	 // output class probability distribution   
	 fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
	 if (buf[0] == 'y' || buf[0] == 'Y')
		 outProbDistr = TRUE ;
	  else
		if (buf[0] == 'n' || buf[0] == 'N')
		 outProbDistr = FALSE ;
		else 
		  error("The switch \"output class probability distribution\" should be on or off (Y or N) in file ", ConfigName) ;

    //  default editor
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    if (strlen(buf) > 0)
		strcpy(defaultEditor, buf) ;

	//  missing value indicator
    fgetStrIgnoreTill(from,buf,'=',"#%") ; ;
    if (strlen(buf) > 0)
		strcpy(NAstring, buf) ;

    fclose(from) ;
    return 1 ;
}


void Options::outConfigOld(FILE *to) const
{
    fprintf(to, "# Options file for %s", VersionString) ;
	fprintf(to, "\n# Note the conventions: ");
    fprintf(to, "\n# each options is on a separate line, the order of options matters") ;
	fprintf(to, "\n# if # is the first character, line is ignored") ;
    fprintf(to, "\n# each line is ignored up to the = character, representing only comment") ;
    fprintf(to, "\n# the ~ character is used instead of = in comments") ;
	fprintf(to, "\n#") ;

    fprintf(to, "\n# ---------- File and data options ---------- ") ;
    
    // Domain name
    fprintf(to,"\nDomain name =%s",domainName) ;
    
    // Data directory
    fprintf(to,"\nData directory =%.*s", strlen(dataDirectory)-1, dataDirectory) ;

    // Results directory
    fprintf(to,"\nResults directory =%.*s", strlen(resultsDirectory)-1, resultsDirectory) ;

    // Definiton of train/test data splits 
    fprintf(to, "\n# Types of supported splits to training/testing data:  ") ;
    fprintf(to,"\n# 0~read from files, 1~cross validation, 2~stratified cross-validation,") ;
    fprintf(to,"\n# 3~leave one out CV, 4~all data is for training, 5~random split to train/test") ;
    fprintf(to, "\nDefiniton of train/test data splits =%d",splitSelection) ; ;

    // Number of of iterations (data split to work on)
    fprintf(to, "\nNumber of data splits (iterations, cross-validations) to work on =%d", numberOfSplits) ;

    // Train proportion
    fprintf(to,"\nIn case of random split to train/test, the proportion of training instances =%f",trainProportion) ;

    // random seed for split
    fprintf(to,"\nRandom seed for data split determination (0~take from clock) =%ld",rndSeedSplit) ;

    // Split index
    fprintf(to,"\nIn case of work on single split, the index of that split=%d",splitIdx) ;
    
    
    // estimators

    fprintf(to, "\n# ---------- Estimation of attributes options ---------- ") ;

    // Treat all attributes as binary 
    fprintf(to,"\nTreat all attributes as binary =%s", (binaryAttributes ? "Y" : "N")) ;

    // Treat continuous attribute splits as binary in applicable measures 
    fprintf(to,"\nTreat continuous attributes' splits as binary in applicable measures =%s", (binarySplitNumericAttributes ? "Y" : "N")) ;

    // Number of examples  for estimation
    fprintf(to,"\nNumber of examples to use for estimation (0 means all) =%d", attrEvaluationInstances) ;


     // switches for estimation
     for (int estIdx = 1 ; estIdx <= NoEstimators ; estIdx++)  
         fprintf(to, "\n%s =%s" , estName[estIdx].dsc, (estOn[estIdx] ? "Y" : "N")) ;


    
    fprintf(to, "\n# ---------- ReliefF options ----------") ;

     // number of iterations for ReliefF's estimation
    fprintf(to,"\nNumber of iterations for ReliefF's estimation (0~TrainSize, -1~ln(TrainSize), -2~sqrt(TrainSize)) =%d",ReliefIterations) ;

    // Default proportion of continuous attribute to consider value equal
    fprintf(to,"\nDefault proportion of continuous attribute to consider value equal =%.5f",numAttrProportionEqual) ;

    // Default proportion of continuous attribute to consider value different
    fprintf(to,"\nDefault proportion of continuous attribute to consider value different =%.5f",numAttrProportionDifferent) ;

    // Number of neighbours to consider - k
    fprintf(to, "\nNumber of neighbours to consider in k-equal estimation =%d",kNearestEqual) ;

    // Number of neighbours in  distance density estimation
    fprintf(to, "\nNumber of neighbours to consider in other estimators (except k-equal) =%d",kNearestExpRank) ;

    // Quotient in Gaussian function of distance density
    fprintf(to, "\nQuotient in Gaussian function of distance density =%f",quotientExpRankDistance) ;

    
    
    fprintf(to, "\n# ---------- Stopping options ---------- ") ;

    // minimal alowable leaf's weight 
    fprintf(to,"\nMinimal weight of a node =%.2f", minNodeWeight) ;

    // Proportion of all examples in a node to stop
    fprintf(to,"\nProportion of all examples in a node to stop =%.3f",relMinNodeWeight) ;

    // Majority class proportion in a node
    fprintf(to,"\nMajority class proportion in a node =%.4f",majorClassProportion) ;

    
    
    fprintf(to, "\n# ---------- Building  options ---------- ") ;

	// selected estimator
    fprintf(to, "\n# Available estimators: 1~ReliefF k-equal, 2~Relief with distance density, 3~ReliefF best of K,") ;
    fprintf(to, "\n#\t4~Kira's Relief, 5~Information gain, 6~Gain ratio, 7~Mdl, 8~Gini, 9~Myopic Relief, 10~Accuracy, \n#\t11~Binarized accuracy, 12~ReliefF with merit, 13~ReliefF with distance, 14~ReliefF with squared distance, \n#\t15~DKM, 16~ReliefF with cost and equal K nearest, 17~Relief with cost and densities") ;
    fprintf(to, "\nEstimator for selection of attributes and binarization (1-15) =%d" , selectionEstimator) ;

    // Minimal ReliefF's estimate of attribute to consider it further
    fprintf(to,"\nIn case of any Relief's variant the minimal evaluation of attribute to considerd it useful =%.4f",minReliefEstimate) ;

	// Minimal probabillity of example to consider it
    fprintf(to,"\nMinimal probabillity of example to consider it =%.2f",minInstanceWeight) ;

    // Type of models used in the leafs (0~point, 1~linear by MSE, 2~linear by MDL, 3~linear as in M5)
    fprintf(to, "\n# Available models: 1~majority class, 2~k-nearest neighbours, 3~k-nearest neighbors with kernel, 4~simple Bayes") ;
    fprintf(to,"\nType of models used in the leafs =%d", modelType) ;

    // k in k nearest neighbour models
    fprintf(to,"\nNumber of neighbours (k) in k-nearst neighbours models  (0~all) =%d", kInNN) ;

    // kernel  in kNN models
    fprintf(to,"\nKernel in k-nearest neighbour models with kernel =%.2f", nnKernelWidth) ;

    // type of discretization for simple Bayes
    fprintf(to, "\nType of discretization for simple Bayes models (1~greedy with selection estimator, 2~equal frequency) =%d", bayesDiscretization) ;

	// number of intervals for equal frequency discretization for simple Bayes models
    fprintf(to, "\nNumber of intervals in equal frequency discretization for simple Bayes models =%d", bayesEqFreqIntervals) ;



    fprintf(to, "\n# ---------- Constructive induction options ---------- ") ;

    // which constructive operators to use
    fprintf(to,"\nConstructive operators sum (1~single, 2~conjunction, 4~addition, 8~multiplication, e.g., all~1+2+4+8 i.e. 15) =%d", constructionMode) ;

    // depth to which to perform  constructive induction
    fprintf(to,"\nMaximal depth (height) of the tree to do construction (0~do not do construction, 1~only at root, ...) =%d", constructionDepth) ;

    // depth to which to perform  constructive induction
    fprintf(to,"\nNumber of cached attributes in each node where construction was performed =%d", noCachedInNode) ;

    // construction estimator
    fprintf(to, "\nEstimator for constructive induction (1-15) =%d" , constructionEstimator) ;

    // beam size for beam search
    fprintf(to,"\nSize of the beam =%d",beamSize) ;

    // maximal size of constructs
    fprintf(to,"\nMaximal size of constructs =%d", maxConstructSize) ;


    // Number of times current discretization can be worse than the best
    fprintf(to,"\nNumber of times current discretization can be worse than the best (0 means try all posibillities) =%d",discretizationLookahead) ;

    // Maximal number of points to try discretization (binarization)    
    fprintf(to,"\nMaximal number of points to try discretization (0 means all sensible) =%d",discretizationSample) ;


    fprintf(to, "\n# ---------- Pruning  options ---------- ") ;

    // selected pruner
    fprintf(to, "\nPruning method used (0~none, 1~MDL, 2~with m-estimate) =%d" , selectedPruner) ;
   
    // Precision of model coefficients in MDL pruning procedure
    fprintf(to, "\nPrecision of the model coefficients in the MDL =%f",mdlModelPrecision) ;

    // Precision of error coefficients in MDL 
    fprintf(to, "\nPrecision of the error in the MDL =%f",mdlErrorPrecision) ;

    // Proportion of equal class values in MDL pruning
    // fprintf(to, "\nProportion of equal class values in MDL pruning procedure =%f",mdlnumAttrProportionEqual) ;

    // m - estimate for pruning
    fprintf(to,"\nm - estimate for pruning =%.4f",mEstPruning) ;


    fprintf(to, "\n# ---------- Random forest options ---------- ") ;

    // number of trees in forest
    fprintf(to,"\nNumber of trees in forest =%d",rfNoTrees) ;

    // Number of randomly selected attributes in the node
	fprintf(to,"\nNumber of randomly selected attributes in the node (0~sqrt(numOfAttr), -1~log_2(numOfAttr)+1, -2~all)=%d",rfNoSelAttr) ;

    // Use multiple estimators
    fprintf(to,"\nUse multiple estimators in the forest =%s",(rfMultipleEst ? "Y" : "N")) ;

    // Number of nearest instances for weighted rf classification
    fprintf(to,"\nNumber of nearest intances for weighted random forest classification (0~no weighting)=%d",rfkNearestEqual) ;

    // Proportion of trees where attribute probabilities are weighted with ReliefF
    fprintf(to,"\nProportion of trees where attribute probabilities are weighted =%f",rfPropWeightedTrees) ;

    // Predict with majority class, otherwise use class distribution
    fprintf(to,"\nPredict with majority class (otherwise with class distribution) =%s",(rfPredictClass ? "Y" : "N")) ;

    // Evaluate attributes with out-of-bag evaluation
    fprintf(to,"\nEvaluate attributes with out-of-bag evaluation =%s",(rfAttrEvaluate ? "Y" : "N")) ;

	// Proportion of the training examples to be used in learning (0.0~bootstrap replication)
    fprintf(to,"\nProportion of the training examples to be used in learning (0.0 ~ bootstrap replication) =%f",rfSampleProp) ;
    
	// Number of leaves in the individual trees (0-build a whole tree)
    fprintf(to,"\nNumber of leaves in the individual trees (0~build a whole tree)=%d",rfNoTerminals) ;

	// Type of regularization (0~no regularization, 1~global regularization, 2~local regularization)
    fprintf(to,"\nType of regularization (0~no regularization, 1~global regularization, 2~local regularization) =%d",rfRegType) ;

	// Regularization parameter Lambda
    fprintf(to,"\nRegularization parameter lambda =%f",rfRegLambda) ;
	
    // random seed for forest
    fprintf(to,"\nRandom seed for random forest (0~take from clock) =%ld", rfRndSeed) ;


    fprintf(to, "\n# ---------- Other  options ---------- ") ;
    
	// m - estimate for prediction
    fprintf(to,"\nm - estimate for prediction =%.4f",mEstPrediction) ;
    
    // print tree also in dot format
	fprintf(to,"\nPrint tree also in dot format =%s", (printTreeInDot ? "Y" : "N")) ;

    // output probability distribution
	fprintf(to,"\nOutput class probability distribution for predicted instances =%s", (outProbDistr ? "Y" : "N")) ;

	// Editor for options
    fprintf(to,"\nEditor for options =%s",defaultEditor) ;

    // Missing values indicator
    fprintf(to,"\nString indicating missing value =%s",NAstring) ;

 }
*/