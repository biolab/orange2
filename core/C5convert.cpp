#include "general.h"
#include "error.h"
#include "dectree.h"
#include "utils.h"
#include "C5defns.h"


extern Attribute	ClassAtt,	/* attribute to use as class */
			LabelAtt;	/* attribute used as case label */
extern	int		MaxAtt,	/* max att number */
			MaxClass,	/* max class number */
			ErrMsgs,      /* errors found */
			AttExIn ;	/* attribute exclusions/inclusions */
extern  	DiscrValue	*MaxAttVal;	/* number of values for each att */
extern	char		*SpecialStatus;	/* special att treatment */
extern	Definition	*AttDef;	/* attribute definitions */
extern	String		*ClassName,	/* class names */
		  	*AttName,	/* att names */
		  	**AttValName;	/* att value names */
extern	Boolean		*SomeMiss,	/* att has missing values */
			*SomeNA;	/* att has N/A values */
extern	ItemNo	MaxItem;	/* max data item number */
extern	Description	*Item;		/* data items */
extern double **MCost ;

// ************************************************************
//
//                      names2dsc
//                      ---------
//
//        converts C5 description to native format
//
// ************************************************************
int dectree::names2dsc(void)
{
   int iA,iV, iN ;
   NoAttr = MaxAtt ;
 
   NoContinuous = 0 ;
   NoDiscrete = 0 ;
   int iT = 0 ; // index in description table
   ContIdx.create(NoAttr+1, -1) ;
   DiscIdx.create(NoAttr+1, -1) ;
   AttrDesc.create(NoAttr+1) ;
   if (AttName[ClassAtt])
      strcpy(AttrDesc[0].AttributeName=new char[1+strlen(AttName[ClassAtt])], AttName[ClassAtt]) ;
   else
      strcpy(AttrDesc[0].AttributeName=new char[1],"") ;
   AttrDesc[0].continuous = FALSE ;  // should be discrete
   AttrDesc[0].NoValues = NoClasses = MaxClass ;
   AttrDesc[0].ValueName.create(AttrDesc[0].NoValues) ;
   AttrDesc[0].valueProbability.create(AttrDesc[0].NoValues+1) ;
   for (iV=1 ; iV <= AttrDesc[0].NoValues ; iV++)
	   strcpy(AttrDesc[0].ValueName[iV-1]=new char[1+strlen(ClassName[iV])], ClassName[iV]) ;
   DiscIdx[NoDiscrete] = 0 ;
   AttrDesc[0].tablePlace = NoDiscrete ;
   NoDiscrete ++ ;
   for (iA=1, iT=1 ; iA <= NoAttr ; iA++)
   {
	   if (iA == ClassAtt || Skip(iA))
		   continue ;

       if (AttName[iA])
         strcpy(AttrDesc[iT].AttributeName=new char[1+strlen(AttName[iA])], AttName[iA]) ;
        else
         strcpy(AttrDesc[iT].AttributeName=new char[1],"") ;
 
       if (Discrete(iA) || Ordered(iA))
	   {
	      AttrDesc[iT].continuous = FALSE ;  // should be discrete
		  AttrDesc[iT].ValueName.create(MaxAttVal[iA]) ;
          AttrDesc[iT].valueProbability.create(MaxAttVal[iA]+1) ;
		  for (iV=1 , iN=0 ; iV <= MaxAttVal[iA] ; iV++)
            if (AttValName[iA][iV] && strcmp(AttValName[iA][iV],"N/A"))
			{
				strcpy(AttrDesc[iT].ValueName[iN]=new char[1+strlen(AttValName[iA][iV])], AttValName[iA][iV]) ;
                iN++ ;
			}
		  AttrDesc[iT].NoValues = iN ;
 
		  DiscIdx[NoDiscrete] = iT ;
          AttrDesc[iT].tablePlace = NoDiscrete ;
          NoDiscrete ++ ;
	   }
	   else if (Continuous(iA) || DateVal(iA) ||  TimeVal(iA) || TStampVal(iA))
	   {
    	  AttrDesc[iT].continuous = TRUE ;  
          AttrDesc[iT].NoValues = 0 ;
          AttrDesc[iT].tablePlace = NoContinuous ;
          AttrDesc[iT].userDefinedDistance = FALSE ;
          AttrDesc[iT].EqualDistance = AttrDesc[iT].DifferentDistance = -1.0 ;
          ContIdx[NoContinuous] = iT ;
          NoContinuous ++ ;
	   }
	   else {
		   error("Unknown data type.","") ;
		   return 0 ;
	   }
	   iT++ ;
   }
   NoOriginalAttr = NoAttr = NoContinuous+NoDiscrete-1 ;

  costsToCostMatrix() ;

  return 1 ;   
}

// ************************************************************
//
//                      costsToCostMatrix
//                      -----------------
//
//        converts C5 costs to native format
//
// ************************************************************
void dectree::costsToCostMatrix(void)
{
   // if missclassification cost matrix exist
    if ( MCost )
    {
        CostMatrix.create(NoClasses+1,NoClasses+1,0) ;
        int i, j ;
        for (i=1 ; i <= NoClasses; i++)
            for (j=1 ; j <= NoClasses ; j++)
                CostMatrix(i,j) = MCost[i][j] ; 
    }
    else  readCosts() ; // check if native costs matrix exists

}

// ************************************************************
//
//                      data2dat
//                      ---------
//
//        converts C5 data to native format
//
// ************************************************************
int dectree::data2dat(void)
{
    int i, j ;
    int iT, contJ, discJ ;

	NoCases= MaxItem+1 ;
    if (NoDiscrete)
      DiscData.create(NoCases, NoDiscrete) ;
    if (NoContinuous)
      ContData.create(NoCases, NoContinuous) ;

	for (i=0 ; i<=MaxItem ; i++)
	{
        contJ =  discJ = iT = 0 ;
        for (j=0, iT=0; j<=MaxAtt ; j++)
		{
		   if (Skip(j) || (j>0 && j==ClassAtt) )
			   continue ;

			if (AttrDesc[iT].continuous)
		   {
			   if (Unknown(Item[i],j))
				   ContData(i,contJ)=NAcont ;
			   else
				   ContData(i,contJ)=CVal(Item[i],j) ;
               contJ++ ;
		   }
		   else {
			   if (Unknown(Item[i],j))
				   DiscData(i,discJ)=NAdisc ;
			   else
				   DiscData(i,discJ)=DVal(Item[i],j)-1 ;
               discJ++ ;
           }
		   iT++ ;
		}
	} 
	
	for (i=0 ; i<=MaxItem ; i++) 
       DiscData(i,0)++ ;

    return 1 ;
}


// ************************************************************
//
//                      FreeC5
//                      ---------
//
//        frees C5 data and description
//
// ************************************************************
void dectree::FreeC5(void)
{
	FreeCases(Item, MaxItem) ;
    FreeGlobals() ;
}
