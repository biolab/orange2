/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/

#ifndef __LINREG_HPP
#define __LINREG_HPP

#include "vars.hpp"
#include "learn.hpp"
#include "classify.hpp"
#include "transval.hpp"

#include "r_imports.hpp"

WRAPPER(Imputer)
WRAPPER(ImputerConstructor)

class ORANGE_API TLinRegLearner : public TLearner
{
public:
  __REGISTER_CLASS

  int iterativeSelection; //P 0 all, 1 forward, 2 backward, 3 stepwise
  float Fin; //P significance limit for an attribute to enter the model
  float Fout; //P significance limit for the attribute to be removed
  int maxIterations; //P maximal number of iterations for stepwise

  PImputerConstructor imputerConstructor; //P if present, it constructs an imputer for unknown values
  PDomainContinuizer continuizer; //P if present, specifies the way in which discrete attributes are converted to continuous

  TLinRegLearner();

  void Fselection(double *X, double*y, double *w, const int &rows, const int &columns,
             bool forward, bool backward,
             int *&pivot, int &rank, double *&coeffs, double *&coeffs_se, double *&cov,
             double &SSres, double &SStot, double &N);

  virtual PClassifier operator()(PExampleGenerator, const int &weight = 0);

  static double *unmix(double *mixed, vector<int> columnOrder, int k);
  static void sort_inPlace(double *mixed, vector<int> columnOrder);
};


class ORANGE_API TLinRegClassifier : public TClassifierFD
{
public:
  __REGISTER_CLASS

	PAttributedFloatList coefficients; //P coefficients of regression plane
  PAttributedFloatList coefficients_se; //P standard errors of coefficients
  float N; //P number of examples
  float SStot; //P total sum of squares
  float SSres; //P residual sum of squares
  float SSreg; //P sum of squares due to regression
  float MStot; //P total mean squares
  float MSres; //P residual mean square
  float MSreg; //P mean square regression
  float F; //P F statistics for the model
  float Fprob; //P significance of the model (F)
  float R2; //P determination
  float adjR2; //P adjusted determination

  PImputer imputer; //P if present, it imputes unknown values
  float threshold; //P classification threshold (for discrete classes)

  TLinRegClassifier();
  TLinRegClassifier(PDomain, PAttributedFloatList, PAttributedFloatList, const float &SSres, const float &SStot, const float &N);

  TValue operator()(const TExample &ex);

  void setStatistics(const float &aSSres, const float &aSStot, const float &N);
};

WRAPPER(LinRegLearner)
WRAPPER(LinRegClassifier)

#endif
