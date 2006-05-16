#if !defined(UTILS_H)
#define UTILS_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "general.h"
#include "contain.h"
#include "error.h"


struct sortRec
{
   int value ;
   double key;
   inline void operator= (sortRec& X) { value=X.value; key=X.key; }
   inline int operator== (sortRec &X) { if (key==X.key) return 1 ; else return 0 ; }
   inline int operator> (sortRec& X) { if (key > X.key)  return 1 ; else return 0 ; }
   inline int operator< (sortRec& X) { if ( key < X.key)  return 1 ; else return 0 ; }
};


void quicksort(sortRec* const T,int left, int right);
int ascSortComp(const void *a, const void *b) ;
int descSortComp(const void *a, const void *b) ;

// char* int2str(int Number, char* const Str);

//   logarithm of basis 2: compatibility sake
inline double log2(double x) { return double( log(x) / 0.69314718055994528622) ; }

#define sqrt2 1.414213562373

#define Phi 3.14159265359

inline long int sqr(int x) { return long(x)*long(x) ; }
inline double sqr(double x) { return x*x ; }
//inline int abs(int x) { return (x>=0 ? x : -x) ; }
#if !defined(MICROSOFT)
inline double abs(double x) { return (x>=0 ? x : -x) ; }
#endif
inline double sign(double x) { if (x>0) return 1.0; else if (x<0) return -1.0; else return 0.0; }
int posCharStr(const char Chr, const char* Str) ;

void strTrim(char* const Source) ;

double multinomLog2(marray<double> &selector) ;
double L2(marray<double> &selector) ;
double gammaLn(double xx) ;
double erfcc(double x) ;

double mdlIntEncode(long int number) ;
double mdlIntEncode(double number) ;

char* fgetStrIgnoreTill(FILE *from, char *Str, char Ignore, char *SkipChars) ;
char* sgetStrIgnoreTill(char *stringFrom, char *Str, char Ignore) ;

long int binom(int N, int selector) ;

int intRound(double x) ;
long int longRound(double x) ;
int no1bits(unsigned long int number) ;

void cvTable(marray<int> &splitTable, int NoCases, int cvDegree)  ;
void stratifiedCVtable(marray<int> &splitTable, marray<int> &classTable, int NoCases, int NoClasses, int cvDegree) ;

void randomizedSample(marray<int> &sampleIdx, int sampleSize, int domainSize) ;
double Correlation(marray<double> &X, marray<double> &Y, int From, int To) ;

double timeMeasure(void) ;
double timeMeasureDiff(double Start, double Finish) ;

char* getWildcardFileName(char* Path, char *WildcardFileName);
double randBetween(double From, double To) ; 
int randBetween(int from, int to) ;
void randSeed(long seed) ;
void printLine(FILE *to, char *what, int times) ;
char *myToken(char *inStr, int &idx, char *delimiters) ;


//template <class T> void AvgStd(marray<T> &Number, int NoNumbers, double &Avg, double &Std) ;
//void intAvgStd(marray<int> &Number, int NoNumbers, double* const Avg, double* const Std) ;
//************************************************************
//
//                        AvgStd
//                        -------
//
//     computes average and standard deviation for int table
//
//************************************************************
template <class T> void AvgStd(marray<T> &Number, int NoNumbers, double &Avg, double &Std)
{
    int i ;
    Avg = Std = 0.0 ;
    for (i=0; i<NoNumbers ; i++)
    {
       Avg += Number[i] ;
       Std += sqr(Number[i]) ;
    }
    Avg /= double(NoNumbers) ;
    Std = sqrt(Std/double(NoNumbers) - sqr(Avg)) ;  
}



#endif

