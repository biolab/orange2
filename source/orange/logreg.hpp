// logreg.hpp
//
// automatically generated from logreg.cpp


#ifndef logreg_hpp
#define logreg_hpp


#include <math.h>
#include <stdlib.h>
#include "lsq.h"
//#include <malloc.h>

double lngamma (double z);
double alnorm (double x, bool upper);
double gammad (double x, double p);
double chi_squared (int ndf, double chi2);
void disaster ();
void logistic (int & ier, int ngroups, double * * x, int k, double * s, double * n, double & chisq, double & devnce, int & ndf, double * beta, double * se_beta, double * fit, double * * cov_beta, double * stdres, int * dependent);

#endif
