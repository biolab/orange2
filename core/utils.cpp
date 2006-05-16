/********************************************************************
*
*   Name:                 MODULE utils
*
*   Description: tools and utilities for other modules,
*
*
*               - quicksort
*               - integer to string
*               - logarithm of basis 2
*               - Rissanen's codes
*               - some maths
*
*********************************************************************/
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#include "general.h"

#if defined(UNIX)
#include <sys/times.h>
#include <glob.h>
#endif

#if defined(MICROSOFT)
#include <io.h>
#endif

#include "utils.h"
#include "error.h"
#include "contain.h"
#include "mathutil.h"

// used for quicksoort
int (*fcmp) (sortRec &X, sortRec& Y) ;

marray<sortRec> *heap ;



//************************************************************
//
//                        int2str
//                        -------
//
//        converts integer to string: compatibility sake
//
//************************************************************
/*
char* int2str(int Number, char* const Str)
{
   int Len = 0;
   for (int i=Number; i != 0; i /= 10) Len++ ;
   if (Number<0)
   {
      Str[0]='-';
      Len++;
   }
   if (Number==0)
   {
     Str[0]='0';
     Str[1]='\0' ;
   }
   else
   {
      Str[Len--] = '\0' ;
      while (Number != 0)
      {
         Str[Len--] = char('0' + Number % 10);
          Number /= 10 ;
      }
   }
   return Str;
}
*/


//************************************************************
//
//                        quicksort
//                        ---------
//
//  standard algorithm for sorting; sorts only sort records
//                defined in header file
//
//************************************************************
void quicksort(sortRec* const T, int left, int right)
{

   qsort((void *)T, right-left+1, sizeof(sortRec), ascSortComp);

}


int ascSortComp( const void *a, const void *b)
{
   return ( ( ((sortRec*)a)->key > ((sortRec*)b)->key ) ? 1 : (-1) ) ;
}


int descSortComp( const void *a, const void *b)
{
   return ( ( ((sortRec*)a)->key < ((sortRec*)b)->key ) ? 1 : (-1) ) ;
}



//************************************************************
//
//                      posCharStr
//                      ----------
//
//       returns the position of char in a string, 0
//                   if there is none
//
//************************************************************
int posCharStr(const char Chr, const char* Str)
{
   int i = 0 ;
   while ( Str[i] )
     if (Str[i++] == Chr)
     {
         return i ;
     }
   return 0 ;
}



//************************************************************
//
//                        strTrim
//                        -------
//
//     removes leading and trailing spaces from given string
//
//************************************************************
void strTrim(char* const Source)
{
  // trim from right
  int pos=0 ;
  while (Source[pos]) pos ++ ;
  pos-- ;
  while (pos >= 0 && Source[pos] == ' ')
    pos-- ;
  if (pos >=0)
     Source[pos+1] = '\0' ;
  else
  {
     Source[0] = '\0' ;
     return ;
  }

  // trim from left ;
  pos = 0 ;
  while (Source[pos]==' ')
    pos++ ;
  if (pos>0)
  {
     int i ;
     for (i = 0 ; Source[pos] != '\0' ; i++, pos++)
        Source[i] = Source[pos] ;
     Source[i] ='\0' ;
  }
}


//************************************************************
//
//                         multinomLog2
//                         ------------
//
//            computes logarithm of base 2 of a kind of multinom:
//              positive real values are used instead of integers
//
//************************************************************
double multinomLog2(marray<double> &selector)
{
   const double ln2 = 0.69314718055994528622 ;

   int noSelectors = selector.filled() ;
   int i ;
   double noAll = 0.0 ;
   for (i = 0 ; i < noSelectors ; i ++ )
      noAll += selector[i] ;

//   int selMax = 0 ;
//   for (i = 1 ; i < noSelectors ; i ++ )
//     if (selector[i] > selector[selMax])
//        selMax = i ;

   // log2(N!)
   double lgNf = gammaLn(noAll+double(1.0))/ln2  ;

   // log2(n_i !)
   marray<double> lgnFac(noSelectors) ;

   for (i=0 ; i< noSelectors ; i++)
   {
     if ((selector[i] == 0) || (selector[i] == 1) )
       lgnFac[i] = 0.0 ;
     else
       if (selector[i] == 2)
          lgnFac[i] = 1.0 ;
        else
          if (selector[i] == noAll)
             lgnFac[i] = lgNf ;
          else
             lgnFac[i] = gammaLn(selector[i]+double(1.0))/ln2  ;
   }

//   double temp = log2(noAll) ;
//   if (lgNf - lgnFac[selMax] < temp )
//   {
//      double sumLg = 0.0 ;
//      for (i=0 ; i < noSelectors ; i++)
//        if (i != selMax)
//           sumLg += lgnFac[i] ;
//
//      delete [] lgnFac ;
//      return selector[selMax] * temp - sumLg ;
//   }
//   else
//   {
      for (i=0 ; i < noSelectors ; i++)
        lgNf -= lgnFac[i] ;

      return lgNf ;
//   }
}


//************************************************************
//
//                         L2 function
//                         ------------
//
//            computes L function in logarithm of base 2 (a modified multinom function)
//      as defined in M. Mehta, J. Rissanen, R. Agrawal: MDL-based Decision Tree Pruning
//         (Proceedings of KDD-95)
//              positive real values are used instead of integers
//
//************************************************************
double L2(marray<double> &selector)
{
   const double ln2 = 0.69314718055994528622 ;
   const double lnPi = 1.144729885849 ;

   int noSelectors =  selector.filled() ;
   double noAll = 0.0 ;
   int i ;
   for (i = 0 ; i < noSelectors ; i ++ )
      noAll += selector[i] ;

   double L = 0.0 ;
   for (i=0 ; i< noSelectors ; i++)
   {
     if (selector[i] != 0.0)
        L += selector[i] * log(noAll/selector[i])  ;
     // else L += 0.0 ;
   }

   L += double(noSelectors-1)/double(2.0) * log(noAll/double(2.0)) + (double(noSelectors)/double(2.0))*lnPi -
          gammaLn(noSelectors/double(2.0)) ;

   L /= ln2 ;
   return double(L) ;
}


//*********************************************************************
//
//                          gammaLn
//                          -------
//
//       computes natural logarithm of gamma function
//              for argument xx > 0;
//        taken from William H. Press, Saul A. Teukolsky,
//         William T. Vetterling, Brian P. Flannery:
//         NUMERICAL RECIPES IN C, The Art od Scientific Computing,
//         Second edition,  Cambridge University Press, 1992
//          approximation by Lanczos:
//
//                                  (z+1/2)    -(z + g +1/2)
//          GAMA(z+1) = (z + g + 1/2)^      * e^              *
//                        _____
//                    * \/ 2*PI * [ c0 + c1/(z+1) + c2/(z+2) + ....
//
//                                  cN/(z+N) + epsilon    ]   ;  Re(z)  > 0
//
//          function bellow works for g=6, N=6, error is smaller than
//           |epsilon| < 2e-10
//
//**********************************************************************
double gammaLn(double xx)
{

   // internal arithmetic will be done in double precision ;
   // single precision will be enough if 5 figure precision is OK

   double x, y, tmp, ser ;

   static double cof[6] = { 76.18009172947146, -86.50532032941677,
                            24.01409824083091, -1.231739572450155,
                            0.1208650973866179e-2, -0.5395239384953e-5
   } ;

   int j ;


   y = x = xx ;
   tmp = x + 5.5 ;
   tmp -=  (x + 0.5) * log(tmp) ;
   ser = 1.000000000190015 ;

   for (j=0 ; j <= 5 ; j++)
      ser += cof[j] / ++y ;

   return double(-tmp + log(2.5066282746310005 * ser / x)) ;

}


//************************************************************
//
//                         mdlIntEncode
//                         ------------
//
//        lengthh of Rissanen's coding of natural numbers:
//     code(0) = 1
//     code(n) = 1 + log2(n) + log2(log2(n)) + ... + log2(2.865064..)
//       where the sum includes only the term that are positive
//
//************************************************************
double mdlIntEncode(long int number)
{
   if (number==0)
     return 1.0 ;

   double code = double(1.0) + log2(double(2.865064)) ;

   double logarithm = log2(number) ;

   while (logarithm > 0)
   {
      code += logarithm ;
      logarithm = log2(logarithm) ;
   }

   return code ;
}


double mdlIntEncode(double number)
{
 
   number = fabs(number) ;

   if (number==0.0)
     return 1.0 ;

   double code = double(1.0) + log2(double(2.865064)) ;

   double logarithm = log2(number) ;

   while (logarithm > 0)
   {
      code += logarithm ;
      logarithm = log2(logarithm) ;
   }

   return code ;
}


//************************************************************
//
//                         fgetStrIgnoreTill
//                         -----------------
//
//       reads line from file, ignoring contens until given character, skiping some lines
//
//**********************************************************
char* fgetStrIgnoreTill(FILE *from, char *Str, char Ignore, char* SkipChars)
{
    char bufWhole[MaxNameLen] ;
    do {
      fgets(bufWhole,MaxNameLen,from) ;
      while (bufWhole[strlen(bufWhole)-1] == '\n' || bufWhole[strlen(bufWhole)-1] == '\r')
         bufWhole[strlen(bufWhole)-1] = '\0' ;
    }  while (strchr(SkipChars, bufWhole[0])) ;
                                      
    char *buf=bufWhole;
    while (buf[0] != '\0' && buf[0] != Ignore )
       buf++ ;
    if (buf[0] != '\0')
    {
       buf++ ;
       strTrim(buf) ;
    }
    strcpy(Str,buf) ;
    return Str ;
}


//************************************************************
//
//                         sgetStrIgnoreTill
//                         -----------------
//
//           reads line from string, ignoring contens until given character
//
//************************************************************
char* sgetStrIgnoreTill(char *stringFrom, char *Str, char Ignore)
{
    char *buf=stringFrom ;
    while (buf[0] != '\0' && buf[0] != Ignore )
       buf++ ;
    if (buf[0] != '\0')
    {
       buf++ ;
       strTrim(buf) ;
    }
    strcpy(Str,buf) ;
    return Str ;
}


//************************************************************
//
//                         erfcc
//                         ------
//
//      complement of error function errf, computed by
//   aproximation based on Chebyshev fitting
//      (Numerical Recipes in C)
//
//************************************************************
double erfcc(double x)
{
   double t,z,ans;

   z=double(fabs(x));
   t=1.0/(1.0 + 0.5*z);
   ans=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
      t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
      t*(-0.82215223+t*0.17087277)))))))));
   return x >= 0.0 ? ans : 2.0-ans;
}


//************************************************************
//
//                         round
//                         ------
//
//           rounds to nearest integer
//
//
//************************************************************
int intRound(double x)
{
   return (x>=0 ? int(x+0.5) : int(x-0.5)) ;
}

long int longRound(double x)
{
   return (x>=0 ? long(x+0.5) : long(x-0.5)) ;
}



//************************************************************
//
//                         binom
//                         ------
//
//           simple implementation of binom value
//
//
//************************************************************
long int binom(int N, int selector)
{
 
  long int value = 1 ;
  selector = Mmin(selector, N-selector) ;
  int j ;
  for (j=N ; j > N-selector ; j--)
    value *= j ;
  for (j=2 ; j <= selector ; j++)
     value /= j ;
  return value ; 
}



//************************************************************
//
//                         no1bits
//                         -------
//
//     number of set (1) bits in number
//
//
//************************************************************
int no1bits(unsigned long int number) 
{
  int no1 = 0 ;
  while (number)
  {
    no1 += number % 2 ;
    number /= 2 ;
  }
  return no1 ;
}



// ************************************************************
//
//                         Correlation
//                         -----------
//
//     returns standard linear correlation coefficient between two arrays
//
//
// ************************************************************
double Correlation(marray<double> &X, marray<double> &Y, int From, int To) 
{
   double sumX=0, sumY=0, sumXY=0, sumX2=0, sumY2=0 ;
   for (int i=From ; i < To ; i++)
   {
      sumX += X[i] ;
      sumY += Y[i] ;
      sumXY += X[i] * Y[i] ;
      sumX2 += sqr(X[i]) ;
      sumY2 += sqr(Y[i]) ;
   }
   int  N = To - From ;
   double divisor = 0, temp ;
   temp = N*sumX2 - sqr(sumX) ;
   if (temp > 0)
      divisor += sqrt(temp) ;
   temp = N*sumY2 - sqr(sumY) ;
   if (temp > 0)
      divisor *= sqrt(temp) ;
   else
      divisor = 0.0 ;

   if (divisor > 0 )
      return (N*sumXY - sumX*sumY)/divisor ;
   else
      return 0.0 ;
}

//************************************************************
//
//                         cvTable
//                         -------
//
//     returns the table filled with the indexes of the splits
//      according to the degree of cross validation 
//
//
//************************************************************
void cvTable(marray<int> &splitTable, int NoCases, int cvDegree) 
{
   marray<int> scrambledTable(NoCases) ;
   int selected, upper, noElem, alreadyUsed, i, j ;

   for (i=0 ; i < NoCases; i++)
     splitTable[i] = i ;

   // scramble the examples
   upper = NoCases ;
   for (i=0 ; i < NoCases ; i++)
   {
      selected = randBetween(0, upper) ;
      scrambledTable[i] = splitTable[selected] ;
      splitTable[selected] = splitTable[--upper] ;
   }

   // determine how many files define split with one element more than NoCases/NumberOfFiles
   upper = NoCases % cvDegree ;
   noElem = NoCases / cvDegree ;
   for (i=0; i<upper ; i++)
   {
     for (j=0 ; j < NoCases ; j++)
        if (scrambledTable[j] >= i*(noElem+1) && scrambledTable[j] < (i+1)*(noElem+1) )
          splitTable[j] = i ;
   }

   alreadyUsed = upper * (noElem+1) ;
   // splits with NoCases/NumberOfFiles
   for (i=upper; i<cvDegree ; i++)
   {
      for (j=0 ; j < NoCases ; j++)
        if (scrambledTable[j] >= alreadyUsed + (i-upper)*noElem && scrambledTable[j] < alreadyUsed +(i+1-upper)*noElem )
	       splitTable[j] = i ; 
   }
}   
   

void stratifiedCVtable(marray<int> &splitTable, marray<int> &classTable, int NoCases, int NoClasses, int cvDegree) {
  marray<marray<int> > clCase(NoClasses+1) ;
  int i, cl ; 
  for (cl=1 ; cl <= NoClasses ; cl++) 
    clCase[cl].create(NoCases) ;
  for (i=0 ; i < NoCases ; i++)
      clCase[classTable[i]].addEnd(i) ;
  int fold = 0, upper, pos ;
  for (cl=1 ; cl <= NoClasses ; cl++) {
      upper =  clCase[cl].filled() ;
      for (i=0 ; i < upper ; i++) {
          pos = randBetween(0, clCase[cl].filled()) ;
          splitTable[clCase[cl][pos]] = fold++ ;
          clCase[cl][pos] = clCase[cl][clCase[cl].filled()-1] ;
          clCase[cl].setFilled(clCase[cl].filled()-1) ;
          if (fold >= cvDegree)
             fold = 0 ;
      }
  }
}



//************************************************************
//
//                       timeMeasure
//                       -----------
//
//     measures the current time as precisely as possible, independend on 
//   the operating system
//
//
//************************************************************
double timeMeasure(void) 
{
   #if defined(MICROSOFT) || defined(BORLAND)
      return (double)clock() ;
   #endif
   #if defined(UNIX)
     struct tms timeMes ;
     times(&timeMes) ;
     return (double) (timeMes.tms_utime + timeMes.tms_stime + 
                      timeMes.tms_cutime + timeMes.tms_cstime) ;
   #endif
}


double timeMeasureDiff(double Start, double Finish) 
{
     return (Finish-Start)/double(CLOCKS_PER_SEC) ;
}


// ************************************************************
//
//                  getWildcardFileName
//                  -------------------
//
//     checks if file exists and returns first matching filename 
//  
//
//
// ************************************************************
char* getWildcardFileName(char *Path, char *WildcardFileName)
{
   char fullName[MaxPath] ;
   sprintf(fullName, "%s%s" ,Path,WildcardFileName) ;

#if defined(MICROSOFT)
   struct _finddata_t choiceF ;
   long hFile ;
   if( (hFile = _findfirst(fullName, &choiceF)) == -1L )
      return 0;
   
   char *FName = new char[strlen(Path)+strlen(choiceF.name)+1] ; 
   sprintf(FName,"%s%s",Path,choiceF.name) ;
   _findclose(hFile) ;
   return FName ;
#endif
#if defined(UNIX)
  glob_t vecP;
  glob(fullName,GLOB_NOSORT,0, &vecP) ;
  char *FName = 0 ;
  if (vecP.gl_pathc >0)
  {
	  FName = new char[strlen(vecP.gl_pathv[0])+1] ; 
      strcpy (FName,vecP.gl_pathv[0]) ;
  }
  globfree(&vecP) ;
  return FName ;
#endif

}
// get indexes of sampleSize samples from 0 to domainSize, used in ReliefF
void randomizedSample(marray<int> &sampleIdx, int sampleSize, int domainSize) {
    int i ;
    if (sampleSize >= domainSize)  {
	   for (i=0 ; i < sampleSize ; i++)
		   sampleIdx[i] = i % domainSize ;
    } else {
	   marray<int> samplePrep(domainSize) ;
	   for (i=0 ; i < domainSize ; i++)
		   samplePrep[i] = i ;
	   int idx, size = domainSize ;
	   for (i=0 ; i < sampleSize ; i++) {
		  idx =  randBetween(0, size) ;
          sampleIdx[i] = samplePrep[idx] ;
		  size -- ;
		  samplePrep[idx] = samplePrep[size] ;
	   }
   }
}


static long rseed = -1 ;

// exclusive of the endpoint values
double randBetween(double From, double To) {
    return From + ran1(&rseed) * (To-From) ;
}

// exclusive of the To value
int randBetween(int From, int To) {
    return From + (int)(ran1(&rseed) * (To-From)) ;
}


// seed equal to 0 is replaced by -1 in ran1
void randSeed(long seed) {
    if (seed > 0)
       rseed = -seed ;
   else
       rseed  = seed ;
}

void printLine(FILE *to, char *what, int times) {
   for (int i=0 ; i < times ; i++)
	  fprintf(to, "%s", what) ;
   fprintf(to, "\n") ;
}


// tokenizer which works similarly to strtoken, but does not skips multiple delimiters 
// and uses variable idx as an indicator for original string
char *myToken(char *inStr, int &idx, char *delimiters) {
  if (idx == -1)
	  return 0 ;
  char *token = &(inStr[idx]) ;
  size_t delimIdx = strcspn(token, delimiters) ;
  if (delimIdx > strlen(token))
	  idx = -1 ;
  else {
	  token[delimIdx]='\0' ;
	  idx += int(delimIdx+1) ;
  }
  
  return token ; 
}
