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

#include <vector>
#include "gsl/gsl_multifit.h"

#include "examplegen.hpp"
#include "stat.hpp"

#include "linreg.ppp"

void exampleGenerator2gsl(PExampleGenerator egen, const int &weightID, bool addConstant, const int &multiTreatment,
                          gsl_matrix *&X, gsl_vector *&y, gsl_vector *&w, int &rows, int &columns);


TLinRegLearner::TLinRegLearner()
: multinomialTreatment(0),
  iterativeSelection(0),
  Fin(float(0.05)),
  Fout(float(0.1)),
  maxIterations(60000)
{}


#define FREE_IF(O,func) { if (O) { func(O); O = NULL; }}

void TLinRegLearner::Fselection(gsl_matrix *X, gsl_vector *y, gsl_vector *w, const int &rows, const int &columns,
                                bool forward, bool backward,
                                gsl_vector *&c, gsl_vector *&c_se, double &SSres, double &SStot)
{
  gsl_matrix *cov = NULL;
  gsl_multifit_linear_workspace *work = NULL;
  gsl_vector *bestc = NULL, *bestc_se = NULL, *cnew = NULL;
  c = gsl_vector_calloc(columns);
  c_se = gsl_vector_calloc(columns);

  double Sy = 0.0, SSy = 0.0, Sw = 0.0;
  int i;
  for(i = 0; i<rows; i++) {
    const double el = gsl_vector_get(y, i);
    const double we = gsl_vector_get(w, i);
    Sy += we * el;
    SSy += we * el * el;
    Sw += we;
  }
  if (Sw == 0.0)
    raiseError("total weight of examples equals 0.0");
  
  SStot = SSy - Sy*Sy/Sw;

  try {
    int iterations, nSelected;
    double SSminimal;

    if (forward) {
      nSelected = 1;

      if (backward) {
        if (Fout <= Fin)
          raiseError("Fout should be higher than Fin");
        iterations = (maxIterations > columns) ? maxIterations : columns;
        if (iterations > columns*columns)
          iterations = columns*columns*columns;
      }
      else
        iterations = columns;

      SSres = SStot;
      SSminimal = Sy/Sw*Sy/Sw*1e-20; // SSres below 1e-20th of average is low enough
      if (SSres < SSminimal) {
        gsl_vector_set(c, 0, Sy/Sw);
        gsl_vector_set(c_se, 0, sqrt(SSres/(rows-2)/rows));
        return;
      }
    }
    else {
      nSelected = iterations = columns;

      work = gsl_multifit_linear_alloc(rows, columns);
      cov = gsl_matrix_alloc(columns, columns);
      gsl_multifit_wlinear(X, w, y, c, cov, &SSres, work);
      FREE_IF(work, gsl_multifit_linear_free);
      if (!backward) {
        for(i=0; i<columns; i++)
          gsl_vector_set(c_se, i, sqrt(gsl_matrix_get(cov, i, i)));
        FREE_IF(cov, gsl_matrix_free)
        return;
      }

      FREE_IF(cov, gsl_matrix_free)
      gsl_vector_set_zero(c);
      SSminimal = 1e-20;
    }

    vector<int> colAttribute, bestOrder;
    for(i = 0; i<columns; colAttribute.push_back(i++));

    /* That's ugly and can cause troubles on future GSLs
       I should use gsl_matrix_submatrix, but it causes stack problems
       with MS VC++ (see commented parts of the code; use them on Linux & Mac?!) */
    gsl_matrix mview = *X;
    mview.owner = 0;

    int changed = 1;
    while(iterations-- && changed-- && (SSres > SSminimal)) {
      for(int direction = forward ? 1 : -1; direction >= -1; direction -= (backward && (!forward || changed)) ? 2 : 4) {
        if ((direction == 1) ? (nSelected==columns) : (nSelected <= 2))
          continue;

        int p = nSelected + direction;
        int swapplace = nSelected - ((direction == -1) ? 1 : 0);
        // mview = gsl_matrix_submatrix(X, 0, 0, rows, p);
        mview.size2 = p;
        work = gsl_multifit_linear_alloc(rows, p);
        cnew = gsl_vector_alloc(p);
        cov = gsl_matrix_alloc(p, p);

        double bestSS = 1e30, SSnew;
        int bestplace = 0;
        bool prevbest = false;
        vector<int>::iterator swapi(colAttribute.begin()+swapplace);
        for(int candidate = swapplace - (forward && backward && (direction==-1) ? 1 : 0); // won't remove the one which we just added
            candidate && (candidate<columns); candidate+=direction) {
          if (swapplace != candidate) {
            gsl_matrix_swap_columns(X, swapplace, candidate);
            swap(*swapi, colAttribute[candidate]);
            if (prevbest) {
              // remember the column with the best candidate so far
              bestplace = candidate;
              prevbest = false;
            }
          }
          gsl_multifit_wlinear(&mview, w, y, cnew, cov, &SSnew, work);

          double F = (SSres - SSnew)*direction / (SSnew/(rows-nSelected));
          double Fprob = (F>1e-10) ? fprob(1.0, double(rows-nSelected), F) : 0.0;

          if (((direction==1) ? (Fprob < Fin) : (Fprob > Fout)) && (SSnew < bestSS)) {
            if (!bestc || (bestc->size != p)) {
              FREE_IF(bestc, gsl_vector_free);
              FREE_IF(bestc_se, gsl_vector_free);
              bestc = gsl_vector_alloc(p);
              bestc_se = gsl_vector_alloc(p);
            }
            gsl_vector_memcpy(bestc, cnew);
            // should use that, but crashes MS VS
            // gsl_vector_view gsl_matrix_diagonal (gsl_matrix * m)
            for(i=0; i<p; i++)
              gsl_vector_set(bestc_se, i, gsl_matrix_get(cov, i, i));

            bestOrder = vector<int>(colAttribute.begin(), colAttribute.begin()+p);

            // remember this candidate at the next swap (a few lines above)
            prevbest = true;
            bestSS = SSnew;
          }
        }

        FREE_IF(cnew, gsl_vector_free)
        FREE_IF(cov, gsl_matrix_free)
        FREE_IF(work, gsl_multifit_linear_free)

        // put the column with the best candidate to the swap place
        if (bestplace) {
            gsl_matrix_swap_columns(X, swapplace, bestplace);
            swap(*swapi, colAttribute[bestplace]);
        }
        else
          // if there is no best candidate place remembered and it was not the last one...
          if (!prevbest)
            continue;

        nSelected += direction;
        SSres = bestSS;
        changed = 1;
      }
    }

    i = 0;
    for(vector<int>::const_iterator seli(bestOrder.begin()), sele(bestOrder.end()); seli!=sele; seli++, i++) {
      gsl_vector_set(c, *seli, gsl_vector_get(bestc, i));
      gsl_vector_set(c_se, *seli, sqrt(gsl_vector_get(bestc_se, i)));
    }
    if (!i) {
      gsl_vector_set(c, 0, Sy/Sw);
      gsl_vector_set(c_se, 0, sqrt(SSres/(rows-2)/rows));
    }


    FREE_IF(bestc, gsl_vector_free)
  }
  catch (...) {
    FREE_IF(c, gsl_vector_free)
    FREE_IF(cnew, gsl_vector_free)
    FREE_IF(cov, gsl_matrix_free)
    FREE_IF(work, gsl_multifit_linear_free)
    FREE_IF(bestc, gsl_vector_free)
    throw;
  }
}


PClassifier TLinRegLearner::operator()(PExampleGenerator gen, const int &weightID)
{
  gsl_matrix *X = NULL;
  gsl_vector *y = NULL, *w = NULL;
  gsl_vector *c = NULL, *c_se = NULL;
  int rows, columns;

  if ((iterativeSelection < 0) || (iterativeSelection > 3))
    raiseError("invalid 'iterativeSelection'");

  try {
    exampleGenerator2gsl(gen, weightID, true, multinomialTreatment, X, y, w, rows, columns);

    if (columns==1)
      raiseError("no useful attributes");

    double SSres, SStot;
    Fselection(X, y, w, rows, columns, (iterativeSelection & 1) == 1, (iterativeSelection & 2) == 2, c, c_se, SSres, SStot);

    FREE_IF(X, gsl_matrix_free)
    FREE_IF(y, gsl_vector_free)
    FREE_IF(w, gsl_vector_free)

    PVarList attributes = mlnew TVarList(); 
    attributes->push_back(gen->domain->classVar);
    PITERATE(TVarList, vl, gen->domain->attributes) 
      attributes->push_back(*vl);

    PAttributedFloatList coeffs = mlnew TAttributedFloatList(attributes, attributes->size());
    PAttributedFloatList coeffs_se = mlnew TAttributedFloatList(attributes, attributes->size());
    TAttributedFloatList::iterator ci(coeffs->begin()), ce(coeffs->end());
    TAttributedFloatList::iterator ci_se(coeffs_se->begin());
    for(int i = 0; ci!=ce; ci++, ci_se++, i++) {
      *ci = float(gsl_vector_get(c, i));
      *ci_se = float(gsl_vector_get(c_se, i));
    }

    FREE_IF(c, gsl_vector_free)

    return mlnew TLinRegClassifier(gen->domain, coeffs, coeffs_se, SSres, SStot);
  }
  catch (...) {
    FREE_IF(X, gsl_matrix_free)
    FREE_IF(y, gsl_vector_free)
    FREE_IF(w, gsl_vector_free)
    FREE_IF(c, gsl_vector_free)
    throw;
  }
}


TLinRegClassifier::TLinRegClassifier()
: SSres(0.0),
  SStot(0.0)
{}


TLinRegClassifier::TLinRegClassifier(PDomain dom, PAttributedFloatList coeffs, PAttributedFloatList coeffs_se, const double &aSSres, const double &aSStot)
: TClassifierFD(dom),
  coefficients(coeffs),
  coefficients_se(coeffs_se),
  SSres(aSSres),
  SStot(aSStot)
{}


TValue TLinRegClassifier::operator()(const TExample &ex)
{ 
  checkProperty(domain)
  TExample exam = TExample(domain, ex);
  TAttributedFloatList::const_iterator ci(coefficients->begin()), ce(coefficients->end());
  TExample::const_iterator ei(exam.begin());
  TVarList::const_iterator vi(domain->attributes->begin());

  float prediction = *(ci++);
  for(; ci!=ce; ci++, ei++, vi++) {
    if ((*ei).isSpecial())
      raiseError("attribute '%s' has unknown value", (*vi)->name.c_str());
    prediction += *ci * ( (*vi)->varType==TValue::INTVAR ? (*ei).intV : (*ei).floatV);
  }

  return TValue(prediction);
}
