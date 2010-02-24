/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "root.hpp"
#include "orvector.hpp"
#include "examplegen.hpp"

#ifndef __LOGFIT_HPP
#define __LOGFIT_HPP

// WRAPPERS
WRAPPER(LogRegFitter)

// abstract class for LR fitters. 
// New fitters should be derived from this one
class ORANGE_API TLogRegFitter : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  // Don't change the order (<= Divergence means that model is fitted, > means error)
  CLASSCONSTANTS(ErrorCode) enum {OK, Infinity, Divergence, Constant, Singularity};

  // main function call, fits LR, returns coefficients and their 
  // corres. standard errors
  virtual PAttributedFloatList operator()(PExampleGenerator, const int &, PAttributedFloatList &, float &, int &, PVariable &)=0;

  // transforms orange PExampleGenerator attributes in a classic C double 2D array
  // returns number of examples and number of attributes as well
  virtual double** generateDoubleXMatrix(PExampleGenerator, long &nexamples, long &nattr);

  // transforms orange PExampleGenerator class in a classic C double array 
  virtual double* generateDoubleYVector(PExampleGenerator, const int &);
  virtual double* generateDoubleYVector_cont(PExampleGenerator, const int &);
  virtual double* generateDoubleTrialsVector(PExampleGenerator, const int &);
};


// output values computed in logistic fitter
class ORANGE_API LRInfo {
public:
  LRInfo();
  ~LRInfo();

   int nn, k;
   double chisq;      // chi-squared
   double devnce;     // deviance
   int    ndf;        // degrees of freedom
   double *beta;      // fitted beta coefficients
   double *se_beta;   // beta std.devs
   double *fit;       // fitted probabilities for groups
   double **cov_beta; // approx covariance matrix
   double *stdres;    // residuals
   int    *dependent; // dependent/redundant variables
   int	  error;
};

// input values for logistic fitter
class ORANGE_API LRInput {
public:
  LRInput();
  ~LRInput();

  long nn;
  long k;
  double **data;    //nn*k
  double *success;     //nn
  double *trials;
};


// Logistic regression fitter via minimization of log-likelihood
// orange integration of Aleks Jakulin version of LR
// based on Alan Miller's(1992) F90 logistic regression code
class ORANGE_API TLogRegFitter_Cholesky : public TLogRegFitter {
public:
  __REGISTER_CLASS

/*  int maxit; //maximum no. iterations
  double offset; //offset on the logit scale
  double tol; //  tolerance for matrix singularity
  double eps; //difference in `-2  log' likelihood for declaring convergence.
  double penalty; //penalty (scalar), substract from ML beta'×penalty×beta. Set if 
     //model doesnt converge */

  // constructor
  TLogRegFitter_Cholesky();
  TLogRegFitter_Cholesky(bool showErrors);

  // Public main function, use it for fitting LR
  virtual PAttributedFloatList operator()(PExampleGenerator, const int &, PAttributedFloatList &, float &, int &, PVariable &);


private:
  static const char *errors[];
};

#endif
