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

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"

class TLinRegLearner : public TLearner
{
public:
  __REGISTER_CLASS

  int multinomialTreatment; //P treatment of multinomial variables (0 ignore, 1 treat as ordinal, 2 report an error)
  int iterativeSelection; //P 0 all, 1 forward, 2 backward, 3 stepwise
  float Fin; //P significance limit for an attribute to enter the model
  float Fout; //P significance limit for the attribute to be removed
  int maxIterations; //P maximal number of iterations for stepwise

  TLinRegLearner();

  void Fselection(gsl_matrix *X, gsl_vector *y, gsl_vector *w, const int &rows, const int &columns,
                  bool forward, bool backward,
                  gsl_vector *&c, gsl_vector *&c_se, double &SSres, double &SStot);

  virtual PClassifier operator()(PExampleGenerator, const int &weight = 0);
};


class TLinRegClassifier : public TClassifierFD
{
public:
  __REGISTER_CLASS

	PAttributedFloatList coefficients; //P coefficients of regression plane
  PAttributedFloatList coefficients_se; //P standard errors of coefficients
  float SSres; //P residual sum of squares
  float SStot; //P total sum of squares

  TLinRegClassifier();
  TLinRegClassifier(PDomain, PAttributedFloatList, PAttributedFloatList, const double &SSres, const double &SStot);

  TValue operator()(const TExample &ex);
};

WRAPPER(LinRegLearner)
WRAPPER(LinRegClassifier)

#endif