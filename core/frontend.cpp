/********************************************************************
*
*      Project:  Constructive induction with ReliefF
*
*      Author:    Marko Robnik
*      Date:    december 1994 -
*
*
*********************************************************************/

/********************************************************************
*
*   Name:                      Main modul
*
*   Description:    calls other modules, initializes the program
*                    and structures,  deals with menus
*
*********************************************************************/

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "general.h"  // general constants and data type definitions
                      // here you specify weather to compile for
                      // Windows, OS2 (C Set++ 2.1, BC 1.5) or UNIX
#if defined(DEBUG)
#if defined(BORLAND)
#include <alloc.h>  // for heapcheck and coreleft
#endif
#if defined(MICROSOFT)
#include <malloc.h>  // for heapcheck and coreleft
#endif
#endif

#if defined(DEBUG_NEW)
  extern long SetSize ;
#endif


#if defined(CSET)
  #include <float.h> // to mask doubleing point underflow exception
#endif


#include "error.h"    // joint method of reporting errors
#include "menu.h"     // functions for menu handling
#include "dectree.h"  // frame for decision trees
#include "ftree.h"    // decision tree with feature construction
#include "rndforest.h"  // random forests 
#include "utils.h"    // various utillities eg. computing of std. dev.
#include "frontend.h"      // header for this file 
#include "estimator.h"    // various utillities eg. computing of std. dev.
#include "utils.h"
#include "options.h"
#include "randomForestClass.hpp"
Options *opt ;

featureTree *gFT ;
extern int NoEstimators ;
extern marray<int> splitTable ;

char VersionString[]="CORE, classification version 0.9.15, built on " __DATE__ " at " __TIME__  ;

int main(int argc, char *argv[]) {
    outVersion(stdout) ;
	//RandomForestLearner* ucenec = new RandomForestLearner();
	//RandomForestClassifier* klasifikator = ucenec->GetClassifier();
	//printf("\nSt Attr: %d",klasifikator->Forest->NoAttr);
	printf("\nKonec");
//    gFT = new featureTree ;
//    opt = new Options() ;
//
//	char  keyword[MaxNameLen], key[MaxNameLen] ;
//    if (argc == 1)  
//		mainMenu() ;
//	else {
//		// first option must be the name of the option file
// 		opt->parseOption(argv[1], keyword, key) ;
//  	    if (strcmp(keyword, "optionFile")==0 || strcmp(keyword, "o")==0) {
//             printf("\nReading configuration file %s . . .", key) ;
//             if (opt->readConfig(key))    
//                printf(" done.") ;
//  	 		 fflush(stdout) ;
//		}
//		else  {
//			error("Unrecognized option (first option should be configuration file):",keyword) ;
//			fprintf(stderr, "Usage: %s # runs in iteractive mode\n", argv[0]) ;
//			fprintf(stderr, "       %s o=optionFile # runs in iteractive mode, first reads configuration file \n", argv[0]) ;
//			fprintf(stderr, "       %s o=optionFile a=action [keyword=value ...] # batch mode executing action,\n", argv[0]) ; 
//			fprintf(stderr, "       # first reads configuration file, options given in command line override the file\n") ;
//			fprintf(stderr, "       # where action can be any of the\n") ;
//			fprintf(stderr, "       #   {none, estOnce, estAll, treeOnce, treeAll, rfOnce, rfAll, data, ordEval3cl}\n") ;
//			fprintf(stderr, "       # and keywords are the same as in configuration file\n") ;
//            exit(1) ;
//		}
//		
//		// now process command line options
//		for (int i=2 ; i < argc ; i++) 
// 		    opt->assignOption(argv[i]) ;
//
//		if (strcmp(opt->action,"none") == 0) 
//			mainMenu() ;
//		else if (strcmp(opt->action,"estOnce") == 0) 
//    	    singleEstimation(gFT) ;
//	    else if (strcmp(opt->action,"estAll") == 0)
//            allSplitsEstimation(gFT) ;
//		else if (strcmp(opt->action, "treeOnce") == 0)
//            singleTree(gFT) ;
//        else if (strcmp(opt->action, "treeAll") == 0)
//            allSplitsTree(gFT) ;
//		else if (strcmp(opt->action,"rfOnce") == 0)
//            singleRF(gFT) ;
//		else if (strcmp(opt->action,"rfAll") == 0)
//            allSplitsRF(gFT) ;
//		else if (strcmp(opt->action,"data") == 0)
//            domainCharacteristics(gFT) ;
//		else if (strcmp(opt->action,"avReliefF") == 0)
//            evalAttrVal(gFT, avReliefF) ;
//		else if (strcmp(opt->action,"avRF") == 0)
//            evalAttrVal(gFT, avRF) ;
//		else if (strcmp(opt->action,"ordEval") == 0)
//            evalOrdAttrVal(gFT, ordEval) ;
//		else if (strcmp(opt->action,"ordEval3cl") == 0)
//			evalOrdAttrVal(gFT,ordEval3cl) ;
//	    else 
//			error("Unrecognized action:",opt->action) ;
//	}
//
//    fflush(stdout) ;
//    delete gFT ;
//    delete opt ; 
//   	splitTable.destroy() ;
//
//#if defined(DEBUG)
//#if defined(BORLAND)
//    // if program behaves correctly, heap should be OK now,
//    // and all allocated memory should be released
//    if (heapcheck()<0 )
//       fprintf(stderr, "\nWARNING: Heap is not OK !!") ;
//#endif
//#if defined(MICROSOFT)	
//   /* Check heap status */
//   int heapstatus = _heapchk();
//   if (heapstatus!= _HEAPOK)
//       fprintf(stderr, "\nWARNING: Heap is not OK !!") ;
//   // _HEAPOK, _HEAPEMPTY, _HEAPBADBEGIN, _HEAPBADNODE
//#endif
//#endif
//
//#if defined(DEBUG_NEW)
//   printf("Still alocated memory blocks: %ld\n",SetSize) ;
//#endif

    return 0 ;
}


void mainMenu(void) {
       char *MainMenu[] = { "Load the domain",
                            "Estimate attributes on single split" ,
                            "Estimate attributes on all splits" ,
                            "Learning trees on single data split",
                            "Learning trees on all data splits",
                            "Learning random forests on single data split",
                            "Learning random forests on all data splits",
                            "Summarize data characteristics",
                            "Options" ,
                            "Load parameters",
                            "Save parameters",
                            "Exit"
     
	   } ;
       int choice ;
       do
       {
    	 printf("\n\n Current domain: ") ;
         if (opt->domainName[0])
           printf("%s\n", opt->domainName) ;
         else
           printf("<none>\n") ;
         fflush(stdout) ;
         char tempName[MaxPath] ;
         switch (choice=textMenu("Choose the number:", MainMenu,12))  {
             // Load domain data
             case 1 : gFT->readProblem() ;
                      break ;

             // attribute estimation on single data split
             case 2 : singleEstimation(gFT) ;
                      break ;

             // attribute estimation on all data splits
			 case 3 : allSplitsEstimation(gFT);
                      break ;

             // learning tree on single data split
             case 4 : singleTree(gFT) ;
                      break ;

             // learning tree on all data split
             case 5 : allSplitsTree(gFT);
                      break ;
     
             // learning random forst on single data split
             case 6 : singleRF(gFT) ;
                      break ;

             // learning random forests on all data split
             case 7 : allSplitsRF(gFT);
                      break ;

             // data characteristics
 			 case 8: domainCharacteristics(gFT) ;
				     break ;

             // Options menu
             case 9 : opt->processOptions() ;
					  gFT->state = empty ;
                      break ;

             // Load parameters
             case 10: printf("\nConfiguration file name: ") ;
                      fflush(stdout) ;
                      scanf("%s",tempName) ;
                      printf("\nReading configuration file %s . . .", tempName ) ;
                      opt->readConfig(tempName) ;
					  gFT->state = empty ;
                      printf(" done.") ;
                      break ;

             // Save parameters
             case 11: printf("\nConfiguration file name: ") ;
                      fflush(stdout) ;
                      scanf("%s", tempName) ;
                      printf("\nWritting configuration file %s . . .", tempName );
                      opt->writeConfig(tempName) ;
                      printf(" done.") ;
                      break ;

             // Exit
             case 12: break ;

             default: error("Non existing menu option.","") ;
         }
       }  while (choice > 0 && choice != 12) ;
}

//**********************************************************************
//
//                      singleEstimation
//                      ----------
//
//      dealing wih single split estimation
//
//**********************************************************************
void singleEstimation(featureTree* const Tree){
   if (Tree->state < data  && !Tree->readProblem()) 
      return ;
   Tree->setDataSplit(opt->splitIdx) ;

   marray<double> weight(Tree->NoTeachCases,1.0) ;
   estimation Estimator(Tree, Tree->DTeach,weight,Tree->NoTeachCases) ;
   FILE *fout ;
   char path[MaxPath] ;
   sprintf(path,"%s%s.%02dest", opt->resultsDirectory, opt->domainName, opt->splitIdx) ;
   if ((fout = fopen(path,"w"))==NULL)
   {
      error("singleEstimation: cannot open results file: ", path)  ;
   }

   outVersion(fout) ;
   Tree->printEstimationHead(fout) ;
   Tree->printEstimationHead(stdout) ;

   marray<marray<double> > Result(NoEstimators+1) ; 
   int estIdx ;
   for (estIdx = 1 ; estIdx <= NoEstimators ; estIdx++)
      if (opt->estOn[estIdx])
         Result[estIdx].create(Tree->NoAttr+1, 0.0) ;
        
   int i;
   attributeCount attrType ;
   double estStart = timeMeasure() ;

   for (estIdx = 1 ; estIdx <= NoEstimators ; estIdx++)
      if (opt->estOn[estIdx])
      {
         Estimator.estimate(estIdx, 0,Tree->NoContinuous,1,Tree->NoDiscrete, attrType) ;

         for (i=1 ; i <= Tree->NoAttr; i++)
           if (Tree->AttrDesc[i].continuous)
             Result[estIdx][i] = Estimator.ContEstimation[Tree->AttrDesc[i].tablePlace] ;
           else
             Result[estIdx][i] =  Estimator.DiscEstimation[Tree->AttrDesc[i].tablePlace] ;
      }

   double estEnd = timeMeasure() ;

   Tree->printEstimations(fout,  opt->splitIdx, Result) ;
   Tree->printEstimations(stdout, opt->splitIdx, Result) ;

   fprintf(fout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(estStart, estEnd)) ;
   fprintf(stdout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(estStart, estEnd)) ;

   printLine(fout,"-",23+11 * Tree->NoAttr)  ;
   printLine(stdout,"-",23+11 * Tree->NoAttr)  ;
   
   //Tree->outConfig(fout) ;

   fclose(fout) ;

}


 //**********************************************************************
//
//                      allSplitsEstimation
//                      --------------------
//
//      dealing wih single split estimation
//
//**********************************************************************
void allSplitsEstimation(featureTree* const Tree)
{
   if (!Tree->readProblem())
	   return ;

   marray<double> weight ;
   estimation *pEstimator ;
   FILE *fout ;
   char path[MaxPath] ;
   sprintf(path,"%s%s.est", opt->resultsDirectory, opt->domainName) ;
   if ((fout = fopen(path,"w"))==NULL)   {
      error("allSplitsEstimation: cannot open results file: ", path)  ;
      exit(1) ;
   }

   outVersion(fout) ;
   Tree->printEstimationHead(fout) ;
   Tree->printEstimationHead(stdout) ;

   int estIdx ;
   marray<marray<double> > result(NoEstimators+1), sumResult(NoEstimators+1) ;
   for (estIdx=1 ; estIdx <= NoEstimators ; estIdx++)
     if (opt->estOn[estIdx])   {
        result[estIdx].create(Tree->NoAttr+1, 0.0) ;
        sumResult[estIdx].create(Tree->NoAttr+1, 0.0) ;
     }
   int i,  iter ;
   attributeCount attrType ;
   double estStart = timeMeasure() ;

   for (iter = 0 ; iter < opt->numberOfSplits ; iter++)   {
      Tree->setDataSplit(iter) ;
      weight.create(Tree->NoTeachCases,1.0) ;
      pEstimator = new estimation(Tree, Tree->DTeach,weight,Tree->NoTeachCases) ;

      for (estIdx = 1 ; estIdx <= NoEstimators ; estIdx++)
        if (opt->estOn[estIdx])  {
           pEstimator->estimate(estIdx, 0,Tree->NoContinuous,1,Tree->NoDiscrete, attrType) ;

           for (i=1 ; i <= Tree->NoAttr; i++)  {
             if (Tree->AttrDesc[i].continuous)
               result[estIdx][i] = pEstimator->ContEstimation[Tree->AttrDesc[i].tablePlace] ;
             else
               result[estIdx][i] =  pEstimator->DiscEstimation[Tree->AttrDesc[i].tablePlace] ;

             sumResult[estIdx][i] += result[estIdx][i] ;
           }
        }
      
      Tree->printEstimations(fout,   iter, result) ;
      Tree->printEstimations(stdout, iter, result) ;
 
      fflush(fout) ;
      fflush(stdout) ;

      delete pEstimator ;
    }

    double estEnd = timeMeasure() ;

   for (i=1 ; i <= Tree->NoAttr; i++)   {
      for (estIdx = 1 ; estIdx <= NoEstimators ; estIdx++)
        if (opt->estOn[estIdx])
           sumResult[estIdx][i] /= double(opt->numberOfSplits) ;
   }
   printLine(fout,"-",23+11 * Tree->NoAttr)  ;
   printLine(stdout,"-",23+ 11 * Tree->NoAttr)  ;
   Tree->printEstimations(fout,   -1, sumResult) ;
   Tree->printEstimations(stdout, -1, sumResult);
 
   Tree->printEstimationsInColumns(fout,   -1, sumResult) ;
   Tree->printEstimationsInColumns(stdout, -1, sumResult) ;
   
   fprintf(fout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(estStart, estEnd)) ;
   fprintf(stdout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(estStart, estEnd)) ;

   fflush(stdout) ;
   fclose(fout) ;
}


//**********************************************************************
//
//                      singleTree
//                      ----------
//
//      dealing wih single tree construction and testing
//
//**********************************************************************
void singleTree(featureTree* const Tree) {
   if (Tree->state < data && !Tree->readProblem())
      return ;
   double buildStart = timeMeasure() ;
   Tree->learnRF = FALSE ;
   Tree->setDataSplit(opt->splitIdx) ;
   if (Tree->constructTree())   {
       Tree->printResultsHead(stdout) ;
       // only after pruning prediction matrix is needed
       int PMxSize = Tree->NoClasses+1 ;
       mmatrix<int> TrainPMx(PMxSize,PMxSize) ;
       mmatrix<int> TestPMx(PMxSize,PMxSize) ;
       int Leaves = Tree->noLeaves() ;
       int freedom = Tree->degreesOfFreedom() ;  
       double TrainAccuracy, TestAccuracy ;
       double TrainInf, TestInf ;
       double TrainCost, TestCost ;
       double TrainAuc, TestAuc ;
	   double TrainSens, TrainSpec, TestSens, TestSpec ;
	   FILE *distrFile = prepareDistrFile(opt->splitIdx) ;
 	   if (distrFile != NULL)
	  	 fprintf(distrFile, "# Training instances \n") ;
       Tree->test(Tree->DTeach, Tree->NoTeachCases, TrainAccuracy, TrainCost, TrainInf, TrainAuc, TrainPMx, TrainSens, TrainSpec, distrFile);
 	   if (distrFile != NULL)
	 	 fprintf(distrFile, "# Testing instances \n") ;
       Tree->test(Tree->DTest, Tree->NoTestCases, TestAccuracy, TestCost, TestInf, TestAuc, TestPMx, TestSens, TestSpec, distrFile) ;
       if (distrFile != NULL)
          fclose(distrFile) ;

       double buildEnd = timeMeasure() ;   
       Tree->printResultLine(stdout, opt->splitIdx,
                            Leaves, freedom, 
                            TrainAccuracy, TrainCost, TrainInf, TrainAuc, 
                            TestAccuracy, TestCost, TestInf, TestAuc, TestSens, TestSpec) ;
       fflush(stdout) ;

       char OutName[MaxFileNameLen] ;
       sprintf(OutName, "%s%s.%02dtree", opt->resultsDirectory, opt->domainName, opt->splitIdx) ;
       Tree->printFTreeFile(OutName, opt->splitIdx,
                            Leaves, freedom, 
                            TrainAccuracy, TrainCost, TrainInf,TrainAuc,
                            TestAccuracy, TestCost, TestInf, TestAuc,
                            TrainPMx, TestPMx, TestSens, TestSpec) ;
       fprintf(stdout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(buildStart, buildEnd)) ;
       fflush(stdout) ;
   }
}

//**********************************************************************
//
//                      allSplitsTree
//                      -------------
//
//          constructs and tests the whole domain
//
//**********************************************************************
void allSplitsTree(featureTree* const Tree) {
   if (!Tree->readProblem())
	   return ;

   Tree->learnRF = FALSE ;

   char path[MaxPath] ;
   sprintf(path,"%s%s.treeResults", opt->resultsDirectory, opt->domainName);
   FILE *to, *distrFile ;
   if ((to=fopen(path,"w"))==NULL)
   {
       error("Cannot open descision tree output file",path) ;
       exit(1) ;
   }

   outVersion(to) ;
   fprintf(to,"Parameters:\n" ) ;
   fprintf(to,"-----------\n" ) ;
   fprintf(stdout,"Parameters:\n" ) ;
   fprintf(stdout,"-----------\n" ) ;
   opt->outConfig(to) ;
   opt->outConfig(stdout) ;
   Tree->outDomainSummary(to) ;
   Tree->outDomainSummary(stdout) ;
   Tree->printResultsHead(to) ;
   Tree->printResultsHead(stdout) ;
   fflush(to) ;
   fflush(stdout) ;
   marray<double> TrainAccuracy(opt->numberOfSplits) ;
   marray<double> TestAccuracy(opt->numberOfSplits) ;
   marray<double> TrainInf(opt->numberOfSplits) ;
   marray<double> TestInf(opt->numberOfSplits) ;
   marray<double> TrainCost(opt->numberOfSplits) ;
   marray<double> TestCost(opt->numberOfSplits) ;
   marray<double> TrainAuc(opt->numberOfSplits) ;
   marray<double> TestAuc(opt->numberOfSplits) ;
   marray<double> TrainSens(opt->numberOfSplits) ;
   marray<double> TrainSpec(opt->numberOfSplits) ;
   marray<double> TestSens(opt->numberOfSplits) ;
   marray<double> TestSpec(opt->numberOfSplits) ;
   marray<int> Leaves(opt->numberOfSplits) ;
   marray<int> freedom(opt->numberOfSplits) ;

   int sizePMx = Tree->NoClasses +1;
   mmatrix<int> TrainPMx(sizePMx,sizePMx), TestPMx(sizePMx, sizePMx) ;

   double buildStart = timeMeasure() ;
   for (int i = 0 ; i < opt->numberOfSplits ; i++)
   {
      Tree->setDataSplit(i) ; 

      if (Tree->constructTree())
      {
		 distrFile = prepareDistrFile(i) ;

         Leaves[i] = Tree->noLeaves() ;
         freedom[i] = Tree->degreesOfFreedom() ;

		 if (distrFile != NULL)
			fprintf(distrFile, "# Training instances \n") ;
         Tree->test(Tree->DTeach, Tree->NoTeachCases,  TrainAccuracy[i], TrainCost[i], TrainInf[i], TrainAuc[i], TrainPMx, TrainSens[i], TrainSpec[i], distrFile) ;
		 if (distrFile != NULL)
			fprintf(distrFile, "# Testing instances \n") ;
         Tree->test(Tree->DTest, Tree->NoTestCases, TestAccuracy[i], TestCost[i], TestInf[i], TestAuc[i], TestPMx, TestSens[i], TestSpec[i], distrFile );

         if (distrFile != NULL)
            fclose(distrFile) ;

         sprintf(path,"%s%s.%02dtree", opt->resultsDirectory, opt->domainName,i) ;
         Tree->printFTreeFile(path,i, Leaves[i], freedom[i],
                               TrainAccuracy[i], TrainCost[i], TrainInf[i],TrainAuc[i],
                               TestAccuracy[i], TestCost[i], TestInf[i],TestAuc[i],
                               TrainPMx, TestPMx, TestSens[i], TestSpec[i]) ;

         Tree->printResultLine(to, i,  Leaves[i], freedom[i],
                               TrainAccuracy[i], TestCost[i], TrainInf[i],TrainAuc[i],
                               TestAccuracy[i], TestCost[i], TestInf[i], TestAuc[i], TestSens[i], TestSpec[i] ) ;

         Tree->printResultLine(stdout, i, Leaves[i], freedom[i],
                               TrainAccuracy[i], TrainCost[i], TrainInf[i], TrainAuc[i],
                               TestAccuracy[i], TestCost[i], TestInf[i], TestAuc[i], TestSens[i], TestSpec[i] ) ;
         fflush(to) ;
         fflush(stdout) ;
      }
   }
   double buildEnd = timeMeasure() ;

   Tree->printResultSummary(to, Leaves, freedom, 
                               TrainAccuracy, TrainCost, TrainInf, TrainAuc,
                               TestAccuracy, TestCost, TestInf, TestAuc, TestSens, TestSpec) ;
   Tree->printResultSummary(stdout, Leaves, freedom, 
                               TrainAccuracy, TrainCost, TrainInf, TrainAuc,
                               TestAccuracy, TestCost, TestInf, TestAuc, TestSens, TestSpec) ;
   fprintf(to,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(buildStart, buildEnd)) ;
   fprintf(stdout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(buildStart, buildEnd)) ;

   fclose(to) ;
   fflush(stdout) ;
}


//**********************************************************************
//
//                      singleRF
//                      ----------
//
//      dealing wih single random forest construction and testing
//
//**********************************************************************
void singleRF(featureTree* const Tree) {
   if (Tree->state < data && !Tree->readProblem())
      return ;
   char path[MaxPath] ;
   FILE *to ;
   sprintf(path,"%s%s.rfResult", opt->resultsDirectory, opt->domainName);
   if ((to=fopen(path,"w"))==NULL) {
       error("Cannot open random forests report file", path) ;
   }
   outVersion(to) ;
   fprintf(to,"Parameters:\n" ) ;
   fprintf(to,"-----------\n" ) ;
   opt->outConfig(to) ;
   Tree->outDomainSummary(to) ;
   Tree->outDomainSummary(stdout) ;
   Tree->rfResultHead(to) ;
   Tree->rfResultHead(stdout) ;
   fflush(to) ;
   fflush(stdout) ;

   double buildStart = timeMeasure() ;
   Tree->learnRF = TRUE ;
   Tree->setDataSplit(opt->splitIdx) ;
   randSeed(opt->rfRndSeed) ;
   if (Tree->buildForest()) {
       int PMxSize = Tree->NoClasses+1 ;
       mmatrix<int> TrainPMx(PMxSize,PMxSize) ;
       mmatrix<int> TestPMx(PMxSize,PMxSize) ;
       
	   double trainAccuracy, testAccuracy ;
       double trainInf, testInf ;
       double trainCost, testCost ;
       double trainAuc, testAuc ;
	   double TrainSens, TrainSpec, TestSens, TestSpec ;

	   FILE *distrFile = prepareDistrFile(opt->splitIdx) ;

	   if (distrFile != NULL)
	  	 fprintf(distrFile, "# Training instances \n") ;
	   Tree->test(Tree->DTeach, Tree->NoTeachCases, trainAccuracy, trainCost, trainInf, trainAuc, TrainPMx, TrainSens, TrainSpec, distrFile);
 	   if (distrFile != NULL)
	  	 fprintf(distrFile, "# Testing instances \n") ;
       Tree->test(Tree->DTest, Tree->NoTestCases, testAccuracy, testCost, testInf, testAuc, TestPMx, TestSens, TestSpec, distrFile) ;
       if (distrFile != NULL)
          fclose(distrFile) ;

       double buildEnd = timeMeasure() ;   
       Tree->rfResultLine(stdout, opt->splitIdx,
		   trainAccuracy, trainCost, trainInf, trainAuc,
           Tree->avgOobAccuracy, Tree->avgOobMargin,Tree->avgOobCorrelation,
           testAccuracy, testCost, testInf, testAuc,TestSens, TestSpec ) ;
       Tree->rfResultLine(to, opt->splitIdx,
		   trainAccuracy, trainCost, trainInf, trainAuc,
           Tree->avgOobAccuracy, Tree->avgOobMargin,Tree->avgOobCorrelation,
           testAccuracy, testCost, testInf, testAuc,TestSens, TestSpec ) ;
	   if (opt->rfAttrEvaluate) {
		  marray<marray<double> > attrEval(1) ;
		  attrEval[0].create(Tree->NoAttr+1) ;
		  attrEval.setFilled(1) ;
          marray<int> idx(1,opt->splitIdx) ;
	      Tree->varImportance(attrEval[0]) ;
		  // for attribute values evaluatioon
		  marray<marray<double> > avEval(Tree->NoAttr+1) ;
		  for (int iA=1 ; iA <= Tree->NoAttr ; iA++) 
			  if (Tree->AttrDesc[iA].continuous)
  			     avEval[iA].create(1) ;
			  else 
			     avEval[iA].create(Tree->AttrDesc[iA].NoValues+1) ;
		  Tree->avImportance(avEval) ;
		  Tree->printAttrEval(stdout,idx,attrEval) ;
		  Tree->printAttrEval(to,idx,attrEval) ;
	   }
       fflush(to) ;
       fprintf(stdout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(buildStart, buildEnd)) ;
       fflush(stdout) ;
	   fclose(to) ;
   }
}



//**********************************************************************
//
//                      allSplitsRF
//                      -------------
//
//          constructs and tests random forest on all splits
//
//**********************************************************************
void allSplitsRF(featureTree* const Tree) {
   if (!Tree->readProblem())
	   return ;
   Tree->learnRF = TRUE ;

   char path[MaxPath] ;
   FILE *to, *distrFile ;
   sprintf(path,"%s%s.rfResult", opt->resultsDirectory, opt->domainName);
   if ((to=fopen(path,"w"))==NULL) {
       error("Cannot open random forests report file",path) ;
       exit(1) ;
   }

   outVersion(to) ;
   fprintf(to,"Parameters:\n" ) ;
   fprintf(to,"-----------\n" ) ;
   fprintf(stdout,"Parameters:\n" ) ;
   fprintf(stdout,"-----------\n" ) ;
   opt->outConfig(to) ;
   opt->outConfig(stdout) ;
   Tree->outDomainSummary(to) ;
   Tree->outDomainSummary(stdout) ;
   Tree->rfResultHead(to) ;
   Tree->rfResultHead(stdout) ;
   fflush(to) ;
   fflush(stdout) ;
   marray<double> TrainAccuracy(opt->numberOfSplits) ;
   marray<double> TestAccuracy(opt->numberOfSplits) ;
   marray<double> TrainInf(opt->numberOfSplits) ;
   marray<double> TestInf(opt->numberOfSplits) ;
   marray<double> TrainCost(opt->numberOfSplits) ;
   marray<double> TestCost(opt->numberOfSplits) ;
   marray<double> TrainAuc(opt->numberOfSplits) ;
   marray<double> TestAuc(opt->numberOfSplits) ;
   marray<double> TrainSens(opt->numberOfSplits) ;
   marray<double> TrainSpec(opt->numberOfSplits) ;
   marray<double> TestSens(opt->numberOfSplits) ;
   marray<double> TestSpec(opt->numberOfSplits) ;
   marray<double> oobAccuracy(opt->numberOfSplits) ;
   marray<double> oobMargin(opt->numberOfSplits) ;
   marray<double> oobCorrelation(opt->numberOfSplits) ;
   marray<marray<double> > attrEval(opt->numberOfSplits+1) ; 
   attrEval[0].create(Tree->NoAttr+1, 0.0) ; // for averages
   attrEval.setFilled(opt->numberOfSplits+1) ;
   marray<int> idx(opt->numberOfSplits+1) ;
   idx[0] = -1 ;
   int sizePMx = Tree->NoClasses +1;
   mmatrix<int> TrainPMx(sizePMx,sizePMx), TestPMx(sizePMx, sizePMx) ;
   randSeed(opt->rfRndSeed) ;

   double buildStart = timeMeasure() ;
   int i ;
   for (i = 0 ; i < opt->numberOfSplits ; i++)  {
      Tree->setDataSplit(i) ;

      if (Tree->buildForest()) {
		 oobAccuracy[i] = Tree->avgOobAccuracy ;
		 oobMargin[i] = Tree->avgOobMargin ;
		 oobCorrelation[i] = Tree->avgOobCorrelation ;
 	     distrFile = prepareDistrFile(i) ;
 	     if (distrFile != NULL)
	  	   fprintf(distrFile, "# Training instances \n") ;
         Tree->test(Tree->DTeach, Tree->NoTeachCases,  TrainAccuracy[i], TrainCost[i], TrainInf[i], TrainAuc[i], TrainPMx, TrainSens[i], TrainSpec[i], distrFile) ;
 	     if (distrFile != NULL)
	  	    fprintf(distrFile, "# Testing instances \n") ;
         Tree->test(Tree->DTest, Tree->NoTestCases, TestAccuracy[i], TestCost[i], TestInf[i], TestAuc[i], TestPMx, TestSens[i], TestSpec[i], distrFile);
     
 	     if (distrFile != NULL)
            fclose(distrFile) ;

         Tree->rfResultLine(to, i, TrainAccuracy[i], TrainCost[i], TrainInf[i],TrainAuc[i], 
                               oobAccuracy[i],oobMargin[i],oobCorrelation[i], 
                               TestAccuracy[i], TestCost[i], TestInf[i], TestAuc[i], TestSens[i], TestSpec[i]) ;
         Tree->rfResultLine(stdout, i, TrainAccuracy[i], TrainCost[i], TrainInf[i],TrainAuc[i], 
                               oobAccuracy[i],oobMargin[i],oobCorrelation[i], 
                               TestAccuracy[i], TestCost[i], TestInf[i],TestAuc[i], TestSens[i], TestSpec[i]) ;
		 if (opt->rfAttrEvaluate) {
		    attrEval[i+1].create(Tree->NoAttr+1) ;
  	        Tree->varImportance(attrEval[i+1]) ;
			for (int iA=1 ; iA <= Tree->NoAttr ; iA++) // averages
				attrEval[0][iA] += attrEval[i+1][iA] ;
			idx[i+1] = i ;
		 }
         fflush(to) ;
         fflush(stdout) ;
      }
   }
   double buildEnd = timeMeasure() ;

   double avgAccTrain, stdAccTrain, avgCostTrain, stdCostTrain, avgInfTrain, stdInfTrain,
          avgOobAcc, stdOobAcc, avgOobMg, stdOobMg, avgOobRo, stdOobRo, 
          avgAccTest, stdAccTest, avgCostTest, stdCostTest, avgInfTest, stdInfTest,
          avgAucTrain, stdAucTrain, avgAucTest, stdAucTest, avgSensTest, stdSensTest, avgSpecTest, stdSpecTest ;
   AvgStd(TrainAccuracy, opt->numberOfSplits, avgAccTrain, stdAccTrain) ;
   AvgStd(TrainCost, opt->numberOfSplits, avgCostTrain, stdCostTrain) ;
   AvgStd(TrainAuc, opt->numberOfSplits, avgAucTrain, stdAucTrain) ;
   AvgStd(TrainInf, opt->numberOfSplits, avgInfTrain, stdInfTrain) ;
   AvgStd(oobAccuracy, opt->numberOfSplits, avgOobAcc, stdOobAcc) ;
   AvgStd(oobMargin, opt->numberOfSplits, avgOobMg, stdOobMg) ;
   AvgStd(oobCorrelation, opt->numberOfSplits, avgOobRo, stdOobRo) ;
   AvgStd(TestAccuracy, opt->numberOfSplits, avgAccTest, stdAccTest) ;
   AvgStd(TestCost, opt->numberOfSplits, avgCostTest, stdCostTest) ;
   AvgStd(TestInf, opt->numberOfSplits, avgInfTest, stdInfTest) ;
   AvgStd(TestAuc, opt->numberOfSplits, avgAucTest, stdAucTest) ;
   AvgStd(TestSens, opt->numberOfSplits, avgSensTest, stdSensTest) ;
   AvgStd(TestSpec, opt->numberOfSplits, avgSpecTest, stdSpecTest) ;

   printLine(to,"-",85)  ;
      
   Tree->rfResultLine(to, -1, avgAccTrain, avgCostTrain, avgInfTrain, avgAucTrain, avgOobAcc, avgOobMg, avgOobRo, avgAccTest, avgCostTest, avgInfTest, avgAucTest, avgSensTest, avgSpecTest) ;
   Tree->rfResultLine(to, -2, stdAccTrain, stdCostTrain, stdInfTrain, stdAucTrain, stdOobAcc, stdOobMg, stdOobRo, stdAccTest, stdCostTest, stdInfTest, stdAucTest, stdSensTest, stdSpecTest) ;
   Tree->rfResultLine(stdout, -1, avgAccTrain, avgCostTrain, avgInfTrain, avgAucTrain, avgOobAcc, avgOobMg, avgOobRo, avgAccTest, avgCostTest, avgInfTest, avgAucTest, avgSensTest, avgSpecTest) ;
   Tree->rfResultLine(stdout, -2, stdAccTrain, stdCostTrain, stdInfTrain, stdAucTrain, stdOobAcc, stdOobMg, stdOobRo, stdAccTest, stdCostTest, stdInfTest, stdAucTest, stdSensTest, stdSpecTest) ;

   if (opt->rfAttrEvaluate) {
	   for (int iA=1 ; iA <= Tree->NoAttr ; iA++)
		   attrEval[0][iA] /= double(opt->numberOfSplits) ;

	   Tree->printAttrEval(to, idx, attrEval) ;
       Tree->printAttrEval(stdout, idx, attrEval) ;
   }

   fprintf(to,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(buildStart, buildEnd)) ;
   fprintf(stdout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(buildStart, buildEnd)) ;

   fclose(to) ;
   fflush(stdout) ;
}



//**********************************************************************
//
//                      domainCharacteristics
//                      ---------------------
//
//      computes concept variation of estimation data
//
//**********************************************************************
void domainCharacteristics(featureTree* const Tree)
{
   if (Tree->state < data && !Tree->readProblem())
      return ;
   char path[MaxPath] ;
   FILE *to ;
   sprintf(path,"%s%s.dataCharacteristics", opt->resultsDirectory, opt->domainName);
   if ((to=fopen(path,"w"))==NULL) {
       error("Cannot open data characteristics report file",path) ;
       exit(1) ;
   }

   Tree->setDataSplit(opt->splitIdx) ;
   outVersion(stdout) ;
   Tree->outDomainSummary(stdout) ;
   outVersion(to) ;
   Tree->outDomainSummary(to) ;
   fprintf(to, "\nNumber of training examples for statistics below: %d\n",Tree->NoTeachCases) ;
   fprintf(to, "\nCharacteristics of attributes:\n") ;
   fprintf(stdout, "\nNumber of training examples for statistics below: %d\n",Tree->NoTeachCases) ;
   fprintf(stdout, "\nCharacteristics of attributes:\n") ;
   int i, j, k ;
   int missing, attrIdx, missingSum=0, valSum=0 ;
   for (i=0 ; i <= Tree->NoAttr ; i++)
   {
       if (i==0) {
	    fprintf(stdout, "Class:") ;
	    fprintf(to, "Class:") ;
       }
       else {
		 fprintf(stdout, "%d. ",i) ;
		 fprintf(to, "%d. ",i) ;
       }
	  fprintf(to, "%s",Tree->AttrDesc[i].AttributeName) ;
	  fprintf(stdout, "%s",Tree->AttrDesc[i].AttributeName) ;
      if (Tree->AttrDesc[i].continuous)
	  {
          missing = 0 ;
		  attrIdx = Tree->AttrDesc[i].tablePlace ;
		  for (k=0 ; k < Tree->NoTeachCases; k++)
             if (Tree->ContData(Tree->DTeach[k], attrIdx) == NAcont)
				 missing++ ;
		 fprintf(stdout, " (numeric, missing: %d values = %7.4f%%)\n", missing, double(missing)/Tree->NoTeachCases*100.0);
		 fprintf(to, " (numeric, missing: %d values = %7.4f%%)\n", missing, double(missing)/Tree->NoTeachCases*100.0);
	     missingSum += missing ;
      }
	  else {
		 attrIdx =  Tree->AttrDesc[i].tablePlace ;
		 marray<int> valCount(Tree->AttrDesc[i].NoValues+1, 0) ;
         if ( i>0 )
             valSum += Tree->AttrDesc[i].NoValues ;
         for (k=0 ; k < Tree->NoTeachCases ; k++)
           valCount[Tree->DiscData(Tree->DTeach[k], attrIdx)] ++ ;
	  	 fprintf(stdout, " (%d values, missing: %d = %7.4f%%)\n",Tree->AttrDesc[i].NoValues,valCount[0], double(valCount[0])/Tree->NoTeachCases*100.0) ;
         fprintf(to, " (%d values, missing: %d = %7.4f%%)\n",Tree->AttrDesc[i].NoValues,valCount[0], double(valCount[0])/Tree->NoTeachCases*100.0) ;
         missingSum += valCount[0]  ;
         for (j=0 ; j < Tree->AttrDesc[i].NoValues ; j++) {
			 fprintf(stdout, "\t%s (%d=%7.4f%%)\n",Tree->AttrDesc[i].ValueName[j], valCount[j+1], double(valCount[j+1])/Tree->NoTeachCases*100.0) ;
			 fprintf(to, "\t%s (%d=%7.4f%%)\n",Tree->AttrDesc[i].ValueName[j], valCount[j+1], double(valCount[j+1])/Tree->NoTeachCases*100.0) ;
         }
	  }
   }
   double avgVal = 0 ;
   if (Tree->NoDiscrete > 1)
       avgVal = double(valSum)/double(Tree->NoDiscrete-1) ;
   fprintf(to,"Average number of values per discrete attribute: %.2f\n",avgVal) ;
   fprintf(to,"Number of missing values: %d = %.2f%%\n", missingSum, double(missingSum)/Tree->NoTeachCases/Tree->NoAttr*100.0) ;
   fprintf(stdout,"Average number of values per discrete attribute: %.2f\n",avgVal) ;
   fprintf(stdout,"Number of missing values: %d = %.2f%%)\n", missingSum, double(missingSum)/Tree->NoTeachCases/Tree->NoAttr*100.0) ;
   printLine(to,"-",50)  ;
      
   // conceptVariation
   marray<double> weight(Tree->NoTeachCases,1.0) ;
   estimation Estimator(Tree, Tree->DTeach,weight,Tree->NoTeachCases) ;

   double ConVar ;
   //ConVar = Estimator.CVVilalta(0,Tree->NoContinuous,1,Tree->NoDiscrete) ;
   //fprintf(stdout,"\nConcept variation (Vilalta) for %d examples is %10.4f\n", Estimator.TrainSize, ConVar) ;
   //fprintf(to,"\nConcept variation (Vilalta) for %d examples is %10.4f\n", Estimator.TrainSize, ConVar) ;

   ConVar = Estimator.CVmodified(0,Tree->NoContinuous,1,Tree->NoDiscrete) ;
   fprintf(stdout,"\nConcept variation (Robnik Sikonja variant): %10.4f\n", ConVar) ;
   fprintf(to,"\nConcept variation (Robnik Sikonja variant): %10.4f\n", ConVar) ;
   fclose(to) ;
}



//**********************************************************************
//
//                      outVersion
//                      ----------
//
//                prints version information
//
//**********************************************************************
void outVersion(FILE *fout) 
{
    fprintf(fout,"%s\n",VersionString) ;
}


//**********************************************************************
//
//                      rf
//                      --------
//
//      random forest built from feature trees
//
//**********************************************************************
void rf(featureTree*  Tree) {
   Tree->buildForest();
}

//**********************************************************************
//
//                      evalAttrVal
//                      -----------
//
//      evaluate attribute values with ReliefF
//
//**********************************************************************
void evalAttrVal(featureTree*  Tree, demandType demand)
{
   if (!Tree->readProblem())
	   return ;

   marray<double> weight ;
   estimation *pEstimator=0 ;
   FILE *fout ;
   char path[MaxPath] ;
   sprintf(path,"%s%s.AVest", opt->resultsDirectory, opt->domainName) ;
   if ((fout = fopen(path,"w"))==NULL)   {
      error("evalAttrVal: cannot open results file: ", path)  ;
      exit(1) ;
   }
   char methodStr[32] ;
   if (demand==avReliefF)								 
	 strcpy(methodStr,"avReliefF") ;
   else if (demand == avRF) 
  	 strcpy(methodStr,"avRF") ;

   int attrIdx ;
   marray<marray<double> > sumResult(Tree->NoDiscrete), result(Tree->NoDiscrete) ;
   for (attrIdx=1 ; attrIdx < Tree->NoDiscrete ; attrIdx++) {
       sumResult[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
       result[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
     }
   int i,  j, iter ;
   Tree->printAVestimationHead(fout,methodStr) ;
   Tree->printAVestimationHead(stdout,methodStr) ;
   double estStart = timeMeasure() ;

   for (iter = 0 ; iter < opt->numberOfSplits ; iter++)   {
      Tree->setDataSplit(iter) ;
      
	  if (demand==avReliefF) {
		  // for ReliefF estiamtion
	      weight.create(Tree->NoTeachCases,1.0) ;
          pEstimator = new estimation(Tree, Tree->DTeach,weight,Tree->NoTeachCases) ;
          pEstimator->aVReliefF(1,Tree->NoDiscrete, result, estReliefFkEqual) ;
	  }
	  else if (demand == avRF) {
	     // for rf estimation
	     Tree->learnRF = TRUE ;
         Tree->buildForest() ;
         Tree->avImportance(result) ;
	  }
	  else error("evalAttrVal", "unrecognized demand") ;

      for (i=1 ; i < Tree->NoDiscrete; i++)  
		  for (j=0 ; j <= Tree->AttrDesc[Tree->DiscIdx[i]].NoValues ; j++)
             sumResult[i][j] += result[i][j] ;
               
      Tree->printAVestimations(fout,  iter, result) ;
      Tree->printAVestimations(stdout, iter, result) ;
 
      fflush(fout) ;
      fflush(stdout) ;

      delete pEstimator ;
    }
    double estEnd = timeMeasure() ;

    for (i=1 ; i < Tree->NoDiscrete; i++)  
		  for (j=0 ; j <= Tree->AttrDesc[Tree->DiscIdx[i]].NoValues ; j++)
             sumResult[i][j] /= double(opt->numberOfSplits) ;

   printLine(fout,"-",60)  ;
   printLine(stdout,"-",60)  ;
   Tree->printAVestimations(fout,   -1, sumResult) ;
   Tree->printAVestimations(stdout, -1, sumResult);

   Tree->printAVestInColumns(fout, sumResult, methodStr) ;
   Tree->printAVestInColumns(stdout, sumResult, methodStr) ;
 
   fprintf(fout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(estStart, estEnd)) ;
   fprintf(stdout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(estStart, estEnd)) ;

   fflush(stdout) ;
   fclose(fout) ;
}




void evalOrdAttrVal(featureTree*  Tree, demandType demand)  {
   if (!Tree->readProblem())
	   return ;
   Tree->setDataSplit(opt->splitIdx) ;

   marray<double> weight ;
   estimation *pEstimator=0 ;
   FILE *fout ;
   char path[MaxPath] ;
   sprintf(path,"%s%s.ordAV", opt->resultsDirectory, opt->domainName) ;
   if ((fout = fopen(path,"w"))==NULL)   {
      error("evalOrdAttrVal: cannot open results file: ", path)  ;
      exit(1) ;
   }
   char methodStr[32] ;
   int attrIdx ;
   marray<marray<double> > resultCpAe(Tree->NoDiscrete), resultCpAp(Tree->NoDiscrete),
	                       resultCpAn(Tree->NoDiscrete), resultCnAe(Tree->NoDiscrete), 
						   resultCnAp(Tree->NoDiscrete), resultCnAn(Tree->NoDiscrete),
						   resultCeAe(Tree->NoDiscrete), resultCeAp(Tree->NoDiscrete), 
						   resultCeAn(Tree->NoDiscrete);
   for (attrIdx=1 ; attrIdx < Tree->NoDiscrete ; attrIdx++) {
	   resultCpAe[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
	   resultCpAp[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
	   resultCpAn[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
	   resultCnAe[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
	   resultCnAp[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
	   resultCnAn[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
	   resultCeAe[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
	   resultCeAp[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
	   resultCeAn[attrIdx].create(Tree->AttrDesc[Tree->DiscIdx[attrIdx]].NoValues+1, 0.0) ;
     }

   double estStart = timeMeasure() ;
   weight.create(Tree->NoTeachCases,1.0) ;
   pEstimator = new estimation(Tree, Tree->DTeach,weight,Tree->NoTeachCases) ;
   
   if (demand == ordEval) {
	   pEstimator->ordAvReliefF(1,Tree->NoDiscrete, resultCpAp,resultCpAn, resultCpAe,
			  resultCnAp,resultCnAn, resultCnAe, resultCeAp, resultCeAn, resultCeAe,
			  estReliefFkEqual) ;
	   strcpy(methodStr, "ordAvReliefF") ;
   }
   else if (demand == ordEval3cl) {
	   pEstimator->ordAV3clReliefF(1,Tree->NoDiscrete, resultCpAp,resultCpAn, resultCpAe,
			  resultCnAp,resultCnAn, resultCnAe,resultCeAp,resultCeAn,resultCeAe, estReliefFkEqual) ;
	   strcpy(methodStr, "ord3clAvReliefF") ;
   }
   else error("evalOrdAttrVal", "unrecognized demand") ;

   Tree->printAVestIn9Columns(stdout, methodStr, resultCpAp,resultCpAn, resultCpAe,
			                  resultCnAp,resultCnAn,resultCnAe, resultCeAp,resultCeAn, resultCeAe) ;
   Tree->printAVestIn9Columns(fout, methodStr, resultCpAp,resultCpAn, resultCpAe,
             			       resultCnAp,resultCnAn, resultCnAe, resultCeAp,resultCeAn,resultCeAe) ;  

   delete pEstimator ;
   double estEnd = timeMeasure() ;

   fprintf(fout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(estStart, estEnd)) ;
   fprintf(stdout,"\nCPU time used: %.2f seconds\n", timeMeasureDiff(estStart, estEnd)) ;

   fflush(stdout) ;
   fclose(fout) ;
}


FILE* prepareDistrFile(int fileIdx) {
  FILE *distrFile = NULL ;
  if (opt->outProbDistr)     {
	 char distrPath[MaxPath] ;
     sprintf(distrPath, "%s%s.%03d.cpd", opt->resultsDirectory, opt->domainName, fileIdx) ;
	 if ((distrFile=fopen(distrPath,"w"))==NULL)
           error("Cannot write to distribution file", distrPath) ;
	 else {
	   fprintf(distrFile, "# Class probability distribution file generated by %s\n", VersionString) ;
	   fprintf(distrFile, "# Format: instance prob_of_class_1 ... prob_of_class_n\n# \n") ;
      }
  }
  return distrFile ;
}
