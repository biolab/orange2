/********************************************************************
*
*   Name:    MODULE dectree
*
*   Description: basic operations on decision trees, not regarding any
*                special additives and improvements;
*                 - data input
*
*********************************************************************/

#include <stdio.h>    // read and write functions
#include <string.h>    // dealing with attribute and value names
#include <stdlib.h>    
#include <float.h>

#include "utils.h"
#include "error.h"
#include "dectree.h"
#include "C5hooks.h"
#include "options.h"

extern Options *opt ;

int NoEstimators = 24;
estDsc estName[25]={ 
    {"", ""} , 
    {"ReliefFequalK",    "ReliefF with equal k-nearest" },           // 1
    {"ReliefFexpRank",   "ReliefF with exponential rank distance" }, // 2
    {"ReliefFbestK",     "ReliefF with best of equal k-nearest" },   // 3
    {"Relief",           "Original Relief" },                // 4 original 
    {"InfGain",          "Information gain" },               // 5
    {"GainRatio",        "Gain ratio" },                     // 6
    {"MDL",              "MDL" },                            // 7
    {"Gini",             "Gini index"},                      // 8
    {"ReliefFmyopic",    "Myopic Relief" },                  // 9
    {"Accuracy",         "Accuracy" },                       // 10
    {"BinAccuracy",      "Binarized accuracy" },             // 11
    {"ReliefFmerit",     "ReliefF with merit"},              // 12
    {"ReliefFdistance",  "ReliefF with direct distance" },   // 13
    {"ReliefFsqrDistance","ReliefF with direct squared distance"},  // 14
    {"DKM",              "Dietterich-Kearns-Mansour (DKM)"}, // 15
    {"ReliefFexpC","ReliefF with expected costs of missclassification"}, // 16
    {"ReliefFavgC","ReliefF with average costs of misclassification"}, // 17
    {"ReliefFpe","ReliefF with expected cost probabilities"},// 18
    {"ReliefFpa","ReliefF with average cost probabilities"},//19
    {"ReliefFsmp", "ReliefF with sampling by expected cost"},// 20
    {"GainRatioCost","Gain ration with cost information"},// 21
    {"DKMcost", "DKM with cost information"},// 22
    {"ReliefKukar","ReliefF with Kukar's variant of costs"}, // 23
    {"MDLsmp","MDL with expected cost sampling"} // 24
} ;

// char CommentChar = "%#" ;
char dataSeparators[] = " ,\t";
marray<int> splitTable ;


/********************************************************************
*
*   Name:   class attribute
*   Base:   / 
*
*   Description:   description of attribute
*
*********************************************************************/

// constructor
attribute::attribute()
{
    AttributeName = 0 ;
    continuous = TRUE ;
    NoValues = 0 ;
    tablePlace = -1 ;
    userDefinedDistance =FALSE ;
    DifferentDistance = EqualDistance = 0.0 ;
}

attribute::~attribute()
{
   destroy() ; 
}


void attribute::destroy()
{
    delete [] AttributeName ;
    AttributeName = 0 ;
    if ( ValueName.defined() && (! continuous) )
    {
       for(int j=0 ; j < NoValues  ; j++)
           delete [] ValueName[j] ;
       ValueName.destroy() ;
    }
    Boundaries.destroy() ;
    valueProbability.destroy() ;

    NoValues = 0  ;
}



/********************************************************************
*
*   Name:   class dectree
*   Base:   bintree
*
*   Description:   base for decision trees: data input, pruning
*
*********************************************************************/

// constructor
dectree::dectree()
{
    NoAttr =  NoOriginalAttr = NoCases = NoTeachCases = NoTestCases  = 0 ;
    NoContinuous = NoDiscrete = 0 ;
    state = empty ;
}

// destructor: clears dinamicaly reserved free store
dectree::~dectree()
{
    clearData();
    clearDescription();
}


// ************************************************************
//
//                    readProblem
//                    -----------
//
//     determine the data format and read in the description 
//     and data for the problem
//
// ************************************************************
int dectree::readProblem(void) //--//
{
   if (! opt->domainName[0])
   {
      error("Uninitialised domain name","") ;
      return 0 ;
   }

   // clear previous data
   clearData() ;
   clearDescription() ;

   // check the existance of .names and determine C5 or native format
   char FileName[MaxPath] ;
   
   sprintf(FileName, "%s%s.dsc", opt->dataDirectory, opt->domainName) ;
   FILE *from ;
   if ((from=fopen(FileName,"r"))!=NULL)
   {
      fclose(from) ;
	  // native format
	  if (opt->domainName[0])
	  {
		  printf("\nReading description, ") ;
          if (readDescription()) 
          {
             if (readCosts()) {
                printf("data, ") ;
			    if (readData())
                printf("done.") ;
			    else {
                    error("Reading a problem failed in data file.","") ;
                    return 0 ;
			    }
             }
		  }
          else {
              error("Reading a problem failed in description file.","") ;
              return 0 ;
		  }
	  }
	  else {
         error("\nSet domain name first.","") ;
         return 0;
	  }
   }
   else {
      sprintf(FileName, "%s%s.names", opt->dataDirectory, opt->domainName) ;
	  if ((from=fopen(FileName,"r"))!=NULL) {	  
		  // C5/See5/Cubist/C4.5 format
		  printf("\nReading C5 names,") ;
		  GetNames(from) ;
		  // get data
          sprintf(FileName, "%s%s.data", opt->dataDirectory, opt->domainName) ;
	      if ((from=fopen(FileName,"r"))!=NULL) 
		  {
             printf(" data,") ;
		     GetData(from, TRUE) ;
		  }
		  else {
			  error("Nonexistent data file",FileName) ;
			  return 0 ;
		  }
          // get cost matrix
          sprintf(FileName, "%s%s.costs", opt->dataDirectory, opt->domainName) ;
	      if ((from=fopen(FileName,"r"))!=NULL) 
		  {
             printf(" C5 costs,") ;
		     GetMCosts(from) ;
		  }
          
		  
		  // convert to internal representation
          printf(" converting, ") ;
          if (names2dsc())
 		     if (data2dat())
			 {
				 prepareDataSplits() ;
		         state=data ;
                 FreeC5() ;
                 printf(" done.\n") ;
			 }
			 else return 0;
		  else return 0 ;
	  } 
	  else {
        sprintf(FileName, "%s%s", opt->dataDirectory, opt->domainName) ;
        error("Description file (.dsc or .names) does not exist for problem",FileName) ;
	    return 0 ;
	  }
   }
   return 1 ;
}





//************************************************************
//
//                    readDescription
//                    ---------------
//
//          read description of attributes from file
//
//************************************************************
int dectree::readDescription(void)
{
   clearDescription();
   char DescFileName[MaxPath] ;
   sprintf(DescFileName, "%s%s.dsc", opt->dataDirectory, opt->domainName) ;
   FILE *from ;
   if ((from=fopen(DescFileName,"r"))==NULL)
   {
      error("Cannot open description file", DescFileName);
      return 0;
   }

   char buf[MaxNameLen+1] ;      // buffers for strings and
   int temp;                     // numbers
   int DescLen ;
   double fTemp1, fTemp2 ;
   // start reading data
   do {
       fgets(buf,MaxNameLen,from);
   } while  (buf[0] == '#' || buf[0] == '%') ;
   sscanf(buf,"%d",&DescLen) ;
   NoOriginalAttr = NoAttr = DescLen -1 ;
   if (NoAttr <= 0)
   {
      error("Data description contains no attributes in file", DescFileName) ;
      return 0 ;
   }
 
   NoContinuous = 0 ;
   NoDiscrete = 0 ;
   ContIdx.create(DescLen, -1) ;
   DiscIdx.create(DescLen, -1) ;

   AttrDesc.create(DescLen) ;
   int i, j ;
   for (i=0; i< DescLen ; i++)
   {
 
       do {
           fgets(buf,MaxNameLen,from);  // name of a attribute
      } while  (buf[0] == '#' || buf[0] == '%') ;
      buf[strlen(buf)-1] = '\0' ;
      strTrim(buf) ;
      strcpy(AttrDesc[i].AttributeName = new char[strlen(buf)+1], buf) ;
      do {
         fgets(buf,MaxNameLen,from);  // description line
      } while  (buf[0] == '#' || buf[0] == '%') ;
      sscanf(buf,"%d",&temp) ; // how many values attribute has
      if (temp == 0)  // continuous variable
      {
          AttrDesc[i].continuous = TRUE ;
          AttrDesc[i].NoValues = 0 ;
          AttrDesc[i].tablePlace = NoContinuous ;
          AttrDesc[i].userDefinedDistance = FALSE ;
          AttrDesc[i].EqualDistance = AttrDesc[i].DifferentDistance = -1.0 ;
              
          ContIdx[NoContinuous] = i ;
          // read distances if defined
          sscanf(buf,"%d%lf%lf",&temp, &fTemp1, &fTemp2) ; // equal and different distance
          if (fTemp1<=0.0 && fTemp2 <= 0.0) // meaning: not defined, use defaults
          {
             AttrDesc[i].userDefinedDistance = FALSE ;
             AttrDesc[i].EqualDistance = AttrDesc[i].DifferentDistance = -1.0 ;
          }
          else
          {
             AttrDesc[i].userDefinedDistance = TRUE ;
             AttrDesc[i].EqualDistance = double(fabs(fTemp1)) ;
             AttrDesc[i].DifferentDistance = double(fabs(fTemp2)) ;
          }
          NoContinuous ++ ;
      }
      else
      {
          AttrDesc[i].continuous = FALSE ;
          DiscIdx[NoDiscrete] = i ;
          AttrDesc[i].tablePlace = NoDiscrete ;
          NoDiscrete ++ ;

          if (temp < 0)
          {
             // Assistent format: continuous attribute with predefined boundaries
             AttrDesc[i].Boundaries.create(-temp, -FLT_MAX) ; 
             AttrDesc[i].NoValues = -temp+1 ;
          }
          else 
          {
             // normal discrete attribute
             AttrDesc[i].NoValues = temp ;
          }
  
          // read the values of the attribute
          AttrDesc[i].ValueName.create(AttrDesc[i].NoValues) ;
          AttrDesc[i].valueProbability.create(AttrDesc[i].NoValues+1) ;
          

          if (temp > 0)  // normal discrete attribute 
          {
             for (j=0  ;  j < temp ; j++)
             {
               fgets(buf,MaxNameLen,from) ;
               buf[strlen(buf)-1] = '\0' ;
               strTrim(buf) ;
               strcpy(AttrDesc[i].ValueName[j]=new char[strlen(buf)+1], buf) ;
             }
          }
          else
          {  
             // continuous attribute with predefined boundaries
             // first interval 
             fgets(buf,MaxNameLen,from) ;
             sscanf(buf,"%lf",&fTemp1) ; // first boundary
             AttrDesc[i].Boundaries[0] = fTemp1 ;
             sprintf(buf, "( <= %.3f )", fTemp1) ;
             strcpy(AttrDesc[i].ValueName[0] = new char[strlen(buf)+1], buf) ; 
             fTemp2 = fTemp1 ;
             // intervals with two boundaries
             for (j=1 ;  j < -temp ; j++)
             {
               fgets(buf,MaxNameLen,from) ;
               sscanf(buf,"%lf",&fTemp2) ; // how many values attribute has
               AttrDesc[i].Boundaries[j] = fTemp2 ;
               sprintf(buf, "( > %.3f  &  <= %.3f )", fTemp1, fTemp2) ;
               strcpy(AttrDesc[i].ValueName[j] = new char[strlen(buf)+1], buf) ;
               fTemp1 = fTemp2 ;
             }
             // last interval
             sprintf(buf, "( > %.3f )", fTemp2) ;
             strcpy(AttrDesc[i].ValueName[j] = new char[strlen(buf)+1], buf) ; 
          }
      }
   }
   if (feof(from) || !ferror(from) )
   {
   	  if (AttrDesc[0].continuous)
	  {
         error("dectree::readDescription","This program assumes classification problem.") ;
	     return 0 ;
	  }
	    NoClasses = AttrDesc[0].NoValues ;
      state=description;
      fclose(from) ;
      return 1;
   }
   else
   {
       clearDescription();
       error("Cannot read file",DescFileName) ;
       fclose(from)  ;
       return 0 ;
   }
}

//************************************************************
//
//                       clearDescription
//                       ----------------
//
//    free the store for description of attributes and classes
//
//************************************************************
void dectree::clearDescription(void)
{
   AttrDesc.destroy() ; 
   ContIdx.destroy() ;
   DiscIdx.destroy() ;
   CostMatrix.destroy() ;
   NoAttr = NoOriginalAttr = NoDiscrete = NoContinuous = 0 ;
}

//************************************************************
//
//                    readDescription (egen)
//                    ---------------
//
//          read description of attributes from file
//
//************************************************************
#include "table.hpp"

void dectree::readDescription(TExampleTable &egen)
{
  const nattrs = egen.domain->variables->size();
  NoOriginalAttr = NoAttr = nattrs -1 ;
  NoContinuous = 0 ;
  NoDiscrete = 1 ;
  ContIdx.create(nattrs, -1) ;
  DiscIdx.create(nattrs, -1) ;
  AttrDesc.create(nattrs);

  int i = 1;
  TVarList::const_iterator vi(egen.domain->attributes->begin()), ve(egen.domain->attributes->end());
  for(; vi != ve; vi++, i++) {
    AttrDesc[i].AttributeName = strcpy(new char[(*vi)->name.length()+1], (*vi)->name.c_str());
    if ((*vi)->varType == TValue::FLOATVAR) {
      AttrDesc[i].continuous = TRUE;
      AttrDesc[i].NoValues = 0;
      AttrDesc[i].tablePlace = NoContinuous ;
      AttrDesc[i].userDefinedDistance = FALSE ;
      AttrDesc[i].EqualDistance = AttrDesc[i].DifferentDistance = -1.0 ;
              
      ContIdx[NoContinuous] = i-1 ;
      NoContinuous++;
    }
    else if ((*vi)->varType == TValue::INTVAR) {
      AttrDesc[i].continuous = FALSE ;
      DiscIdx[NoDiscrete] = i-1 ;
      AttrDesc[i].tablePlace = NoDiscrete ;

      const TEnumVariable &evar = dynamic_cast<const TEnumVariable &>((*vi).getReference());
      AttrDesc[i].NoValues = evar.noOfValues();

      AttrDesc[i].ValueName.create(AttrDesc[i].NoValues) ;
      AttrDesc[i].valueProbability.create(AttrDesc[i].NoValues+1) ;
      
      int j = 0;
      const_PITERATE(TStringList, ai, evar.values)
        AttrDesc[i].ValueName[j++] = strcpy(new char[(*ai).length()+1], (*ai).c_str()) ;

      NoDiscrete++;
    }
  }
  
  PVariable classVar = egen.domain->classVar;
  if (classVar->varType != TValue::INTVAR)
    throw "discrete class expected";
  //AttrDesc[0].AttributeName = strcpy(new char[6], "peter");
  AttrDesc[0].AttributeName = strcpy(new char[classVar->name.length()+1], classVar->name.c_str());
  AttrDesc[0].continuous = FALSE ;
  DiscIdx[0] = ege.domain->variables->size()-1;
  AttrDesc[0].tablePlace = 0 ;

  const TEnumVariable &evar = dynamic_cast<const TEnumVariable &>(classVar);
  AttrDesc[0].NoValues = evar.noOfValues();

  AttrDesc[0].ValueName.create(AttrDesc[0].NoValues) ;
  AttrDesc[0].valueProbability.create(AttrDesc[0].NoValues+1) ;
  
  int j = 0;
  const_PITERATE(TStringList, ai, evar.values)
    AttrDesc[0].ValueName[j++] = strcpy(new char[(*ai).length()+1], (*ai).c_str()) ;

  NoClasses = AttrDesc[0].NoValues;

}
// ************************************************************
//
//                           readData (egen)
//                           --------
//
//                     read the data from file
//
// ************************************************************

void dectree::readData(TExampleTable &egen)
{
  clearData();

  NoCases = egen.numberOfExamples();

  if (NoDiscrete)
    DiscData.create(egen.numberOfExamples()+1, NoDiscrete) ;
  if (NoContinuous)
    ContData.create(egen.numberOfExamples()+1, NoContinuous) ;

  int i = 0;
  for(TExampleIterator ei(egen.begin()); ei; ++ei, i++) {
    int contJ = 0, discJ = 1;
    for(TExample::const_iterator eei((*ei).begin()), eee((*ei).end()-1); eei != eee; eei++)
      if ((*eei).varType == TValue::FLOATVAR)
        ContData.Set(i, contJ++, (*eei).isSpecial() ? NAcont : (*eei).floatV);
      else
        DiscData.Set(i, discJ++, (*eei).isSpecial() ? NAdisc : (*eei).intV+1);
    if ((*ei).getClass().isSpecial())
      throw "missing class value";
    DiscData.Set(i, 0, (*ei).getClass().intV+1);
  }
  state = data;
  // set data split  -> vsi exampli v Teach, en prazen prostor za Test
  int noOfExamp = egen.numberOfExamples();
  DTeach.create(noOfExamp);
  DTest.create(1);
  for (i=0;i<noOfExamp;i++){
	  DTeach[i]=i;
  }
  DTest[0]=noOfExamp;
  NoTeachCases=noOfExamp;
  NoTestCases=1;
  SetValueProbabilities();
  SetDistances();
}

// ************************************************************
//
//                           readData
//                           --------
//
//                     read the data from file
//
// ************************************************************
int dectree::readData(void)
{
   clearData() ;

   if (state<description) // we have to read description of attributes first
   {
      error("To read data, you have to read description first.","");
      return 0;
   }

   
   char DataFileName[MaxPath] ;
   sprintf(DataFileName, "%s%s.dat", opt->dataDirectory, opt->domainName) ;
 
   // check the existance of file
   FILE *dfrom ;
   if ((dfrom=fopen(DataFileName,"r"))==NULL)
   {
      error("Cannot open data file",DataFileName);
      return 0;
   }

   char strBuf[MaxNameLen+1] ;

   // read the data, skip comments
   do {
     fgets(strBuf, MaxNameLen, dfrom) ;
   } while  (strBuf[0] == '%' || strBuf[0] == '#') ;
   sscanf(strBuf,"%d",&NoCases) ;

   int temp, i ;
   double number;
   char buf[MaxIntLen] ;
   char msg[MaxPath] ;
   char *token ;
   if (NoDiscrete)
     DiscData.create(NoCases, NoDiscrete) ;
   if (NoContinuous)
     ContData.create(NoCases, NoContinuous) ;
   int contJ, discJ ;

   for (i = 0 ; i < NoCases; i++)
   {  

      do {
        fgets(strBuf, MaxNameLen, dfrom) ;
      } while  (strBuf[0] == '#' || strBuf[0] == '%') ;
      if (strBuf[strlen(strBuf)-1] == '\n')
         strBuf[strlen(strBuf)-1] = '\0' ;

	  token = strtok(strBuf, dataSeparators );

      contJ = 0, discJ = 0 ;
      for (int j=0 ; j<= NoAttr; j++ )
      {
        if (token == 0)
		 {
		     sprintf(buf,"%d",i+1) ;
          	 error("Not enough values at example",buf) ;
         }
         if (AttrDesc[j].continuous)
         {
			if (strcmp(token,opt->NAstring) == 0) {
                ContData.Set(i,contJ,NAcont) ;
                if (j==0) // missing classification
                {
                  sprintf(buf,"%d",i+1) ;
                  error("Missing class value at example ",buf) ;
                }
			}
			else {
			  sscanf(token,"%lf", &number) ;
              ContData.Set(i,contJ,number) ;
			}
            contJ ++ ;
         }
         else   // discrete attribute
         {
            if (AttrDesc[j].Boundaries.defined())
            {
               if (strcmp(token,opt->NAstring) == 0)
                  DiscData.Set(i,discJ,NAdisc) ;
			   else {
			       sscanf(token,"%lf", &number) ;
                   temp = 0 ;
                   while (temp < AttrDesc[j].Boundaries.len() && 
                         number < AttrDesc[j].Boundaries[temp])
                    temp ++ ;
                   DiscData.Set(i, discJ, temp + 1) ;
               }  
            }
            else
            { 
               // ordinary discrete attribute 
			   if (strcmp(token,opt->NAstring) == 0){
                  DiscData.Set(i,discJ,NAdisc) ;
                  if (j==0) // missing classification
                  {
                    sprintf(buf,"%d",i+1) ;
                    error("Missing class value at example ",buf) ;
                  }
			   }
			   else {
			      sscanf(token,"%d", &temp) ;
                  if ((temp<=0) || (temp>AttrDesc[j].NoValues))
                  {
                    DiscData.Set(i,discJ,NAdisc) ;
                    strcpy(msg, "Data file corrupted; example ") ;
                    sprintf(buf,"%d",i+1) ;
                    strcat(msg,buf) ;
                    strcat(msg, ", Attribute ") ;
                    sprintf(buf,"%d",j) ;
					strcat(msg,buf) ;
                    strcat(msg, ": unexisting value (") ;
                    sprintf(buf,"%d",temp) ;
					strcat(msg,buf) ;
                    strcat(msg, "). ") ;
                    error(msg,"") ;
                 }
                 else
                   DiscData.Set(i,discJ,temp) ;
               }
            }
            discJ ++ ;
         }
   		 token = strtok(0, dataSeparators );   

      }
   }
   if ( ferror(dfrom) )
   {
      clearData();
      error("Cannot read data from data file", DataFileName);
      fclose(dfrom) ;
      return 0;
   }
   fclose(dfrom) ;


   if (prepareDataSplits())
      state = data ;
   
   return 1 ;
}


//************************************************************
//
//                       clearData
//                       ---------
//
//                free the store reserved for data
//
//************************************************************
void dectree::clearData(void)
{
   DiscData.destroy();
   ContData.destroy() ;
   DTeach.destroy() ;
   DTest.destroy() ;
   state = description;
   NoCases = NoTeachCases = NoTestCases = 0 ;

}


//************************************************************
//
//                       prepareDataSplits
//                       -----------------
//
//  prepares splits to train/test data
//
//************************************************************
int dectree::prepareDataSplits(void) {
   if (state<description) // we have to read the data first
   {
      error("dectree::prepareDataSplit","split to train/test data demands that data is read first");
      return 0;
   }
   randSeed(opt->rndSeedSplit) ;
   if (opt->splitSelection != FROM_FILES && opt->splitSelection != RANDOM_SPLIT) {
       // if FROM_FILES or RANDOM_SPLIT everything will be done at the time of use
       // generate  cross validation splits
       splitTable.create(NoCases) ;
       switch (opt->splitSelection) {
                   case CROSS_VALIDATION: 
                           cvTable(splitTable, NoCases, opt->numberOfSplits) ;
                           break ;
                   case STRATIFIED_CV:
                        {
                           marray<int> classTable(NoCases) ;
                           for (int i=0 ; i < NoCases ; i++)
                               classTable[i] = DiscData(i, 0) ;
                           stratifiedCVtable(splitTable, classTable, NoCases, NoClasses, opt->numberOfSplits) ;
                        }
                        break ;
                   case LOOCV:
                       {
                           opt->numberOfSplits = NoCases ;
                           for (int i=0 ; i < NoCases ; i++)
                               splitTable[i] = i ;
                       }
                       break ;
                   case ALL_TRAINING:
                       splitTable.init(1) ; 
                       break ;
      }
   }
   return 1 ;
}


//************************************************************
//
//                       setDataSplit
//                       ------------
//
// sets the correct  data split in tables
//
//************************************************************
int dectree::setDataSplit(int splitIdx)
{
   if (state<data) 
   {
      error("dectree::setDataSplit", "To select train/test distribution of data, read data first");
      return 0;
   }

   int NoTempTeach = 0, NoTempTest = 0, i ;
   marray<int> TempTeach(NoCases);
   marray<int> TempTest(NoCases) ;
 
   // use predefined splits on files  
   if (opt->splitSelection == FROM_FILES) {
   	   // check the existance of file
	   char ChoiceFileName[MaxPath] ;
   
	   sprintf(ChoiceFileName, "%s.*%ds", opt->domainName, splitIdx) ;
	   char *FName = getWildcardFileName(opt->dataDirectory, ChoiceFileName);
	   if (FName == 0) {
		  error("Nonexistent choices file",ChoiceFileName);
		  return 0;
	   }
	   strcpy(ChoiceFileName, FName) ;
	   delete [] FName ;
  
	   FILE *cfrom ;
	   if ((cfrom=fopen(ChoiceFileName,"r"))==NULL) {
		  error("Cannot open choices file",ChoiceFileName);
		  return 0;
	   }

	   int choice ;
	   for (i=0; i<NoCases ; i++) {
		  fscanf(cfrom,"%d", &choice) ;
		  if (choice==0)
			TempTeach[NoTempTeach++] = i;
		  else
			if (choice==1)
			  TempTest[NoTempTest++] = i;
	   }

	   if (ferror(cfrom)) {
		  error("\nCannot read data from choices file", ChoiceFileName) ;
		  fclose(cfrom) ;
		  return 0;
	   }
	   fclose(cfrom) ;
   }
   else if (opt->splitSelection == RANDOM_SPLIT) {
       for (i=0; i<NoCases ; i++) 
           if (randBetween(0.0,1.0) <= opt->trainProportion)
  			 TempTeach[NoTempTeach++] = i;
		  else
			  TempTest[NoTempTest++] = i;
   }
   else { 
	   for (i=0; i<NoCases ; i++) {
		  if (splitTable[i]!=splitIdx)
			TempTeach[NoTempTeach++] = i;
		  else
			  TempTest[NoTempTest++] = i;
	   }
   }

   // set teach and test cases
   if (NoTempTeach == 0)     {
	   error("\nNo training instances", "") ;
	   return 0 ;
   }
   NoTeachCases = NoTempTeach ;
   NoTestCases = NoTempTest ;
   DTeach.create(NoTeachCases);
   DTest.create(NoTestCases);
   for (i=0; i<NoTeachCases ; i++)
      DTeach[i] = TempTeach[i] ;
   for (i=0; i<NoTestCases ; i++)
      DTest[i] = TempTest[i] ;
   DTeach.setFilled(NoTeachCases) ;
   DTest.setFilled(NoTestCases) ;

   SetValueProbabilities() ;

   SetDistances() ;

   return 1;
}


// ************************************************************
//
//                           readCosts
//                           --------
//
//                     read the costs from file
//
// ************************************************************
int dectree::readCosts(void)
{
    CostMatrix.destroy() ;
    
   if (state<description) // we have to read description of attributes first
   {
      error("dectree::readCosts","Cannot read costs unless description is read first.");
      return 0;
   }

   char CostFileName[MaxPath] ;
   sprintf(CostFileName, "%s%s.cm", opt->dataDirectory, opt->domainName) ;
 
   CostMatrix.create(NoClasses+1,NoClasses+1,0) ;

      // check the existance of file
   FILE *cfrom ;
   int i, j ;
   if ((cfrom=fopen(CostFileName,"r"))==NULL)
   {
       for (i=1; i <= NoClasses; i++)
          for (j=1; j <= NoClasses; j++)
              if (i==j) CostMatrix(i,j) = 0 ;
              else CostMatrix(i,j) = 1.0 ;
       return 1;
   }
   else
       printf("costs, ") ;

   char strBuf[MaxNameLen+1], *token ;
   double costValue ;
   char buf[16] ;
   for (i = 1 ; i <= NoClasses; i++)
   {  
      do {
        fgets(strBuf, MaxNameLen, cfrom) ;
      } while  (strBuf[0] == '#' || strBuf[0] == '%') ;
      if (strBuf[strlen(strBuf)-1] == '\n')
         strBuf[strlen(strBuf)-1] = '\0' ;

	  token = strtok(strBuf, dataSeparators );

      for (j=1 ; j<= NoClasses; j++ )
      {
        if (token == 0)	 {
		  sprintf(buf,"%d",j) ;
          error("Not enough values for class value ",buf) ;
        }
        sscanf(token,"%lf", &costValue) ;
        CostMatrix(i,j) = costValue ;

        token = strtok(0, dataSeparators );   
      }
   }
   fclose(cfrom) ;
   return 1 ;
}


//************************************************************
//
//                           writeDescription
//                           ----------------
//
//                     writes the description  to given file
//
//************************************************************
int dectree::writeDescription(const char* DescriptionFileName) const
{
   FILE *descOut ;
   
   if ((descOut=fopen(DescriptionFileName,"w"))==NULL)
   {
      error("Cannot create output description file", DescriptionFileName);
      return 0;
   }
   int i, j ;
   fprintf(descOut,"%d\n",NoAttr+1) ;
   for (i=0 ; i<=NoAttr ; i++)
   {
      fprintf(descOut,"%s\n",AttrDesc[i].AttributeName) ;
      if (AttrDesc[i].continuous)
      {
         fprintf(descOut,"0 \n") ;
      }
      else
      {
         fprintf(descOut,"%d\n",AttrDesc[i].NoValues) ;
         for (j=0 ; j < AttrDesc[i].NoValues ; j++)
              fprintf(descOut,"%s\n",AttrDesc[i].ValueName[j]) ;
      }
   }
   if (ferror(descOut))
   {
       error("Error at writing description file to ",DescriptionFileName) ;
       fclose(descOut) ;
       return 0 ;
   }
   fclose(descOut) ;
   return 1 ;
}


//************************************************************
//
//                           writeData
//                           ---------
//
//                     writes data to given file
//
//************************************************************
int dectree::writeData(const char* DataFileName) const
{
   FILE *dataOut ;
   if ((dataOut=fopen(DataFileName,"w"))==NULL)
   {
      error("Cannot create output data file", DataFileName);
      return 0;
   }
   fprintf(dataOut,"%d\n",NoCases) ;
   int i,j ;
   for (i=0 ; i < NoCases ; i++)
   {
      for (j=0 ; j <= NoAttr ; j++)
      {
          if (AttrDesc[j].continuous)
          {
             if (ContData(i, AttrDesc[j].tablePlace) == NAcont)
               fprintf(dataOut," %10s", opt->NAstring) ;
             else
               fprintf(dataOut," %10f",ContData(i, AttrDesc[j].tablePlace)) ;
          }
          else
          {
            if (DiscData(i, AttrDesc[j].tablePlace) == NAdisc)
              fprintf(dataOut," %4s",opt->NAstring) ;
            else
              fprintf(dataOut," %4d",int(DiscData(i,AttrDesc[j].tablePlace))) ;
           }
      }
      fprintf(dataOut,"\n") ;
   }

   if (ferror(dataOut))
   {
       error("Error at writing data file to ",DataFileName) ;
       fclose(dataOut) ;
       return 0 ;
   }

   fclose(dataOut) ;
   return 1 ;
}




//************************************************************
//
//                           SetValueProbabilities
//                           ---------------------
//
//            compute value probabilities for discrete attributes from training set
//
//************************************************************
void dectree::SetValueProbabilities(void)
{
   int i, j ;
   marray<int> valueProb ;
   for (i=0 ; i < NoDiscrete ; i++)
   {

     valueProb.create(AttrDesc[DiscIdx[i]].NoValues+1, 0) ;
     for (j=0 ; j < NoTeachCases ; j++)
       valueProb[DiscData(DTeach[j],i)] ++ ;
   
     for (j=0 ; j <= AttrDesc[DiscIdx[i]].NoValues ; j++)
        AttrDesc[DiscIdx[i]].valueProbability[j] = double(valueProb[j]+1.0)/double(NoTeachCases + AttrDesc[DiscIdx[i]].NoValues ) ;
   }
}


//************************************************************
//
//                           SetDistances
//                           ---------
//
//            compute non user-defined distances from training set
//
//************************************************************
void dectree::SetDistances(void)
{
   int i, j ;
   maxValue.create(NoContinuous) ;
   minValue.create(NoContinuous) ;
   valueInterval.create(NoContinuous) ;
   for (i=0 ; i < NoContinuous ; i++)
   {
     j=0 ; 
     while (j < NoTeachCases && ContData(DTeach[j], i) == NAcont)
       j++ ;
     if (j >= NoTeachCases)
       error("dectree::SetDistances", "all values of the attribute are missing") ;
     else
        minValue[i] = maxValue[i] = ContData(DTeach[j],i) ;

     for (j=j+1 ; j < NoTeachCases ; j++)
       if (ContData(DTeach[j], i) != NAcont)
       {
         if (ContData(DTeach[j], i) < minValue[i])
            minValue[i] = ContData(DTeach[j], i) ;
         else
           if (ContData(DTeach[j], i) > maxValue[i])
             maxValue[i] = ContData(DTeach[j], i) ;
       }
   }
   for (i=0 ; i < NoContinuous ; i++)
   {
      valueInterval[i] = maxValue[i] - minValue[i] ;

     if ( ! AttrDesc[ContIdx[i]].userDefinedDistance)
     {
        AttrDesc[ContIdx[i]].EqualDistance = valueInterval[i] * opt->numAttrProportionEqual ;
        AttrDesc[ContIdx[i]].DifferentDistance = valueInterval[i] * opt->numAttrProportionDifferent ;
     }
   }
}


//************************************************************
//
//                      printEstimationHead
//                      -------------------
//
//              prints head of estimation table
//
//************************************************************
void dectree::printEstimationHead(FILE *to) const
{
   fprintf(to,"\n\nidx%20s", "estimator") ;
   int i ;
   for (i=1 ; i <= NoAttr; i++)
     fprintf(to,"%10s ", AttrDesc[i].AttributeName) ;
   fprintf(to, "\n") ;
   for (i=1 ; i <= 23+11*NoAttr; i++)
   fprintf(to, "-") ;
   fprintf(to, "\n") ;

}



 //************************************************************
//
//                      printEstimations
//                      ----------------
//
//        prints estimations for one split
//
//************************************************************
void dectree::printEstimations(FILE *to, int splitIdx, marray<marray<double> > &Result) const
{
  int i, estIdx ;
     
  for (estIdx=1 ; estIdx <= NoEstimators ; estIdx++)
    if (opt->estOn[estIdx])
     {
       fprintf(to, "%02d %20s", splitIdx, estName[estIdx]) ;
       for (i=1 ; i <= NoAttr; i++)
          if (Result[estIdx][i] == -FLT_MAX)
            fprintf(to, "%10s ", "NA") ;
          else
            fprintf(to, "%10.5f ", Result[estIdx][i]) ;
       fprintf(to, "\n") ;
     }
 }



// ************************************************************
//
//                      printEstimationsInColumns
//                      -------------------------
//
//        prints estimations for one split
//
// ************************************************************
void dectree::printEstimationsInColumns(FILE *to, int splitIdx, marray<marray<double> > &Result) const
{
     int estIdx ;
     // print header
     fprintf(to, "\n\n%02d\n%17s, ", splitIdx, "Attr.name") ;
     for (estIdx=1 ; estIdx <= NoEstimators ; estIdx++)
       if (opt->estOn[estIdx])
         fprintf(to, "%17s ", estName[estIdx]) ;
     fprintf(to, "\n") ;


     // print rows, one for each attribute
     for (int i=1 ; i <= NoAttr; i++)
     {
        fprintf(to, "%17s, ", AttrDesc[i].AttributeName) ;
        for (estIdx=1 ; estIdx <= NoEstimators ; estIdx++)
          if (opt->estOn[estIdx])
             fprintf(to, "%17.6f ", Result[estIdx][i]) ;

        fprintf(to, "\n") ;
     }
}

//************************************************************
//
//                      printAVestimationHead
//                      -------------------
//
//              prints head of attribute-value estimation table
//
//************************************************************
void dectree::printAVestimationHead(FILE *to,char *methodStr) const
{
   fprintf(to,"\n%s\nidx",methodStr) ;
   int i, j ;
   for (i=1 ; i < NoDiscrete; i++) {
	   fprintf(to,"%10s ", AttrDesc[DiscIdx[i]].AttributeName) ;
	   for (j=0 ; j < AttrDesc[DiscIdx[i]].NoValues; j++)
		   fprintf(to,"%10s ", AttrDesc[DiscIdx[i]].ValueName[j]) ;
   }
   fprintf(to, "\n-------------------------------------------------------------------------------------------------------\n") ;
}


//************************************************************
//
//                      printAVestimations
//                      ----------------
//
//        prints estimations for one split
//
//************************************************************
void dectree::printAVestimations(FILE *to, int splitIdx, marray<marray<double> > &Result) const
{
  int i, j ;
     
  fprintf(to, "%02d ", splitIdx) ;
  for (i=1 ; i < NoDiscrete; i++) {
     for (j=0 ; j <= AttrDesc[DiscIdx[i]].NoValues; j++)
		   fprintf(to,"%10.5f ", Result[i][j]) ;
  }
  fprintf(to, "\n") ;
}

//************************************************************
//
//                      printAVestInColumns
//                      ----------------
//
//        prints estimations for one split
//
//************************************************************
void dectree::printAVestInColumns(FILE *to, marray<marray<double> > &Result, char *methodStr) const  {
  int i, j ;
  
  fprintf(to, "\n%10s %10s\n", "Attr.value",methodStr) ;

  for (i=1 ; i < NoDiscrete; i++) {
     fprintf(to,"%10s %10.5f\n", AttrDesc[DiscIdx[i]].AttributeName, Result[i][0]) ;
	 for (j=1 ; j <= AttrDesc[DiscIdx[i]].NoValues; j++) 
		 fprintf(to,"%10s:%10.5f\n", AttrDesc[DiscIdx[i]].ValueName[j-1], Result[i][j]) ;
  }
  fprintf(to, "\n") ;
}



void dectree::printAVestIn9Columns(FILE *to, char *methodStr,
	  marray<marray<double> > &ResultCpAp,
	  marray<marray<double> > &ResultCpAn, marray<marray<double> > &ResultCpAe,
	  marray<marray<double> > &ResultCnAp,
	  marray<marray<double> > &ResultCnAn, marray<marray<double> > &ResultCnAe,
	  marray<marray<double> > &ResultCeAp,
	  marray<marray<double> > &ResultCeAn, marray<marray<double> > &ResultCeAe
	  ) const  {
  int i, j ;
  
  fprintf(to, "\n%s", methodStr) ;
  fprintf(to, "\n%10s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", 
	          "Attr.value","CpAp","CpAn","CpAe","CnAp","CnAn","CnAe", "CeAp","CeAn","CeAe") ;

  for (i=1 ; i < NoDiscrete; i++) {
     fprintf(to,"%10s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f \n", 
		 AttrDesc[DiscIdx[i]].AttributeName, 
		 ResultCpAp[i][0], ResultCpAn[i][0], ResultCpAe[i][0],
		 ResultCnAp[i][0], ResultCnAn[i][0], ResultCnAe[i][0],
		 ResultCeAp[i][0], ResultCeAn[i][0], ResultCeAe[i][0]) ;
	 for (j=1 ; j <= AttrDesc[DiscIdx[i]].NoValues; j++) 
		 fprintf(to,"%10s:%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n", 
		 AttrDesc[DiscIdx[i]].ValueName[j-1], 
		 ResultCpAp[i][j], ResultCpAn[i][j], ResultCpAe[i][j],
		 ResultCnAp[i][j], ResultCnAn[i][j], ResultCnAe[i][j],
		 ResultCeAp[i][j], ResultCeAn[i][j], ResultCeAe[i][j]) ;
  }
  fprintf(to, "\n") ;

  fprintf(to, "\nVisualization data:\n%10s %10s %10s %10s\n", 
	          "Attr.value","CpAp/Cp","CnAn/Cn","CeAe/Ce") ;

  double reinfPos, reinfNeg, anchor, denom ;
  for (i=1 ; i < NoDiscrete; i++) {
	 for (j=0 ; j <= AttrDesc[DiscIdx[i]].NoValues; j++) {
	     if (j==0)
			fprintf(to,"%10s ", AttrDesc[DiscIdx[i]].AttributeName) ;
		 else fprintf(to,"%10s:", AttrDesc[DiscIdx[i]].ValueName[j-1]) ;
 	     denom = ResultCpAp[i][j] + ResultCpAn[i][j] + ResultCpAe[i][j] ;
		 if (denom > 0)
			reinfPos = ResultCpAp[i][j] / denom ;
		 else reinfPos = 0.0 ;
         denom = ResultCnAn[i][j] + ResultCnAp[i][j]+ ResultCnAe[i][j] ;
		 if (denom > 0)
			reinfNeg = ResultCnAn[i][j] / denom ;
		 else reinfNeg = 0.0 ;
         denom = ResultCeAn[i][j] + ResultCeAp[i][j]+ ResultCeAe[i][j] ;
		 if (denom > 0)
			anchor = ResultCeAe[i][j] / denom ;
		 else anchor = 0.0 ;
		 fprintf(to,"%8.3f %8.3f %8.3f\n", reinfPos, reinfNeg, anchor) ;
	 }
  }
  fprintf(to, "\n") ;

}


//************************************************************
//
//                           writeTree
//                           ---------
//
//                     writes Tree to given file
//
//************************************************************
/*
int dectree::writeTree(const char* TreeFileName) const
{
   FILE *treeOut ;
   if ((treeOut=fopen(TreeFileName,"w"))==NULL)
   {
      error("Cannot create output tree file", TreeFileName);
      return 0;
   }

   if (root)
     writeSubTree(treeOut,root,0) ;

   if (ferror(treeOut))
   {
       error("Error at writing tree to file ",TreeFileName) ;
       fclose(treeOut) ;
       return 0 ;
   }

   fclose(treeOut) ;
   return 1 ;
}


// ************************************************************
//
//                           writeSubTree
//                           ---------
//
//                     writes subtree to given file
//
// ************************************************************
void dectree::writeSubTree(FILE *treeOut, binnode* Node, int tab) const
{
    int i ;
    switch (Node->Identification)
    {
         case leaf:
                   fprintf(treeOut, "%*s 1 %f %f %f %f %f \n",
                         tab, "",
                         Node->weight, Node->averageClassValue,
                         Node->stdDevClass, Node->squaresClass, Node->code) ;
                    break ;
         case continuousAttribute:
                   fprintf(treeOut, "%*s 2 %d %f %f %f %f %f %f %f \n",
                         tab, "", Node->AttrIdx,
                         Node->splitValue, Node->weight, Node->weightLeft,
                         Node->averageClassValue, Node->stdDevClass,
                         Node->squaresClass, Node->code) ;
                   writeSubTree(treeOut, Node->left, tab+4) ;
                   writeSubTree(treeOut, Node->right, tab+4) ;
                   break ;
          case discreteAttribute:
                   fprintf(treeOut, "%*s 3 %d %f %f %f %f %f %f ",
                         tab, "", Node->AttrIdx,
                         Node->weight, Node->weightLeft,
                         Node->averageClassValue, Node->stdDevClass,
                         Node->squaresClass, Node->code) ;
                   for (i=0 ; i < Node->leftValues.len() ; i++)
                       if (Node->leftValues[i])
                           fprintf(treeOut, "1 ") ;
                       else
                           fprintf(treeOut, "0 ") ;
                   fprintf(treeOut,"\n") ;
                   writeSubTree(treeOut, Node->left, tab+4) ;
                   writeSubTree(treeOut, Node->right, tab+4) ;
                   break;
          default:
                   error("dectree::wrteSubTree","invalid node identification") ;
     }
}


// ************************************************************
//
//                           readTree
//                           ---------
//
//                     reads Tree from given file
//
// ************************************************************
int dectree::readTree(const char* TreeFileName) const
{
   FILE *treeIn ;
   if ((treeIn=fopen(TreeFileName,"r"))==NULL)
   {
      error("Cannot read tree from file", TreeFileName);
      return 0;
   }

   destroy(root) ;
   root = 0 ;
   readSubTree(treeIn,root) ;

   if (ferror(treeIn))
   {
       error("Error reading tree from file ",TreeFileName) ;
       fclose(treeIn) ;
       return 0 ;
   }

   fclose(treeIn) ;
   return 1 ;
}

// ************************************************************
//
//                           readSubTree
//                           ---------
//
//                     reads Tree from given file
//
// ************************************************************
void dectree::readSubTree(FILE *treeIn, binnode* &Node) const
{
    Node = new binnode ;
    int dTemp, d1Temp ;
    char buf[MaxNameLen+1] ;
    char buf1[MaxNameLen] ;
    fgets(buf, MaxNameLen, treeIn);
    int i ;
    sscanf(buf, "%d", &dTemp)  ;

    switch (dTemp)
    {
       case 1:  // leaf
               Node->Identification = leaf ;
               sscanf(buf,"%d%lf%lf%lf%lf%lf", &d1Temp, &(Node->weight), &(Node->averageClassValue), &(Node->stdDevClass), &(Node->squaresClass), &(Node->code) ) ;
               Node->left = Node->right = 0 ;
               break ;
       case 2: // node with continuous attribute
               Node->Identification = continuousAttribute ;
               sscanf(buf, "%d%d%lf%lf%lf%lf%lf%lf%lf",
                            &d1Temp, &(Node->AttrIdx),
                            &(Node->splitValue), &(Node->weight),
                            &(Node->weightLeft), &(Node->averageClassValue),
                            &(Node->stdDevClass), &(Node->squaresClass),
                            &(Node->code) ) ;
               readSubTree(treeIn, Node->left) ;
               readSubTree(treeIn, Node->right) ;
               break ;
       case 3: // node with discrete attribute
               Node->Identification = discreteAttribute ;
               sscanf(buf, "%d%d%lf%lf%lf%lf%lf%lf%s",
                           &d1Temp, &(Node->AttrIdx),
                           &(Node->weight), &(Node->weightLeft),
                           &(Node->averageClassValue), &(Node->stdDevClass),
                           &(Node->squaresClass), &(Node->code), buf1 ) ;
               Node->leftValues.create(AttrDesc[DiscIdx[Node->AttrIdx]].NoValues+1) ;
               strTrim(buf1) ;
               if (strlen(buf1) != AttrDesc[DiscIdx[Node->AttrIdx]].NoValues )
                  error("dectree::readSubTree","number of discrete attribute values do not match") ;
               for (i=1; i < Node->leftValues.len(); i++)
               {
                  if (buf1[i-1] == '1')
                    Node->leftValues[i] = TRUE ;
                  else
                    if (buf1[i-1] == '0')
                       Node->leftValues[i] = FALSE ;
                    else
                      error("dectree::readSubTree","invalid indicator of discrete attribute value") ;
               }
               readSubTree(treeIn, Node->left) ;
               readSubTree(treeIn, Node->right) ;
               break ;
      default: error("dectree::readSubTree","invalid type of node") ;

    }
}
*/


