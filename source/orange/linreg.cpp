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
#include "gslconversions.hpp"

#include "stat.hpp"

#include "examplegen.hpp"
#include "table.hpp"
#include "imputation.hpp"

#include "linreg.ppp"


TLinRegLearner::TLinRegLearner()
: multinomialTreatment(0),
  iterativeSelection(0),
  Fin(float(0.05)),
  Fout(float(0.1)),
  maxIterations(60000)
{}


void TLinRegLearner::Fselection(gsl_matrix *X, gsl_vector *y, gsl_vector *w, const int &rows, const int &columns,
                                bool forward, bool backward,
                                vector<int> &bestOrder, gsl_vector *&best_c, gsl_vector *&best_se, double &SSres, double &SStot, double &N)
{
  gsl_matrix *cov = NULL;
  gsl_multifit_linear_workspace *work = NULL;
  gsl_vector *bestc = NULL, *bestc_se = NULL, *cnew = NULL;

  double Sy = 0.0, SSy = 0.0;
  N = 0.0;
  int i;
  for(i = 0; i<rows; i++) {
    const double el = gsl_vector_get(y, i);
    const double we = gsl_vector_get(w, i);
    Sy += we * el;
    SSy += we * el * el;
    N += we;
  }
  if (N == 0.0)
    raiseError("total weight of examples equals 0.0");
  
  SStot = SSy - Sy*Sy/N;

  try {
    int iterations, nSelected;
    double SSminimal;

    if (forward) {
      nSelected = 1;

      if (backward) {
        if (Fout <= Fin)
          raiseError("Fout should be higher than Fin");
        iterations = (maxIterations > columns) ? maxIterations : columns;
        float columns3 = float(columns)*columns*columns;
        if (iterations > columns3)
          iterations = columns3;
      }
      else
        iterations = columns;

      SSres = SStot;
      SSminimal = Sy/N*Sy/N*1e-20; // SSres below 1e-20th of average is low enough
      if (SSres < SSminimal) {
        bestOrder = vector<int>(1, 0);
        best_c = gsl_vector_alloc(1);
        best_se = gsl_vector_alloc(1);
        gsl_vector_set(best_c, 0, Sy/N);
        gsl_vector_set(best_se, 0, sqrt(SSres/(rows-2)/rows));
        return;
      }
    }

    else { // !forward
      nSelected = iterations = columns;

      work = gsl_multifit_linear_alloc(rows, columns);
      cov = gsl_matrix_alloc(columns, columns);
      best_c = gsl_vector_alloc(columns);
      gsl_multifit_wlinear(X, w, y, best_c, cov, &SSres, work);
      GSL_FREE(work, multifit_linear);

      if (!backward) {
        best_se = gsl_vector_alloc(columns);
        for(i=0; i<columns; i++)
          gsl_vector_set(best_se, i, sqrt(gsl_matrix_get(cov, i, i)));
        GSL_FREE(cov, matrix)

        bestOrder = vector<int>();
        for(i = 0; i<columns; bestOrder.push_back(i++));
        return;
      }

      GSL_FREE(cov, matrix)
      GSL_FREE(best_c, vector)
      SSminimal = 1e-20;
    }

    vector<int> columnOrder;
    for(i = 0; i<columns; columnOrder.push_back(i++));

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
        vector<int>::iterator swapi(columnOrder.begin()+swapplace);
        for(int candidate = swapplace - (forward && backward && (direction==-1) ? 1 : 0); // won't remove the one which we just added
            candidate && (candidate<columns); candidate+=direction) {
          if (swapplace != candidate) {
            gsl_matrix_swap_columns(X, swapplace, candidate);
            swap(*swapi, columnOrder[candidate]);
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
              GSL_FREE(best_c, vector);
              GSL_FREE(best_se, vector);
              best_c = gsl_vector_alloc(p);
              best_se = gsl_vector_alloc(p);
            }
            gsl_vector_memcpy(best_c, cnew);
            // should use that, but crashes MS VC
            // gsl_vector_view gsl_matrix_diagonal (gsl_matrix * m)
            for(i=0; i<p; i++)
              gsl_vector_set(best_se, i, gsl_matrix_get(cov, i, i));

            bestOrder = vector<int>(columnOrder.begin(), columnOrder.begin()+p);

            // remember this candidate at the next swap (a few lines above)
            prevbest = true;
            bestSS = SSnew;
          }
        }

        GSL_FREE(cnew, vector)
        GSL_FREE(cov, matrix)
        GSL_FREE(work, multifit_linear)

        // put the column with the best candidate to the swap place
        if (bestplace) {
            gsl_matrix_swap_columns(X, swapplace, bestplace);
            swap(*swapi, columnOrder[bestplace]);
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

    if (nSelected==1) {
      GSL_FREE(best_c, vector);
      GSL_FREE(best_se, vector);
      bestOrder = vector<int>(1, 0);
      best_c = gsl_vector_alloc(1);
      best_se = gsl_vector_alloc(1);
      gsl_vector_set(best_c, 0, Sy/N);
      gsl_vector_set(best_se, 0, sqrt(SSres/(rows-2)/rows));
    }

  }
  catch (...) {
    GSL_FREE(cnew, vector)
    GSL_FREE(cov, matrix)
    GSL_FREE(work, multifit_linear)
    GSL_FREE(best_c, vector)
    GSL_FREE(best_se, vector)
    throw;
  }
}


gsl_vector *TLinRegLearner::unmix(gsl_vector *mixed, vector<int> columnOrder, int N)
{
  gsl_vector *straight = gsl_vector_calloc(N);
  int i = 0;
  for(vector<int>::const_iterator seli(columnOrder.begin()), sele(columnOrder.end()); seli!=sele; seli++, i++)
    gsl_vector_set(straight, *seli, gsl_vector_get(mixed, i));
  return straight;
}


#include <algorithm>

template<class T>
class TSortedIndexVector : public vector<int> {
public:
  const vector<T> &vect;

  TSortedIndexVector(vector<T> &v)
  : vector<int>(),
    vect(v)
  { 
    for(int i(0), e(vect.size()); i<e; push_back(i++));
    sort(begin(), end(), *this);
  }

  bool operator ()(const int &ind1, const int &ind2) const
  { return less<T>()(vect[ind1], vect[ind2]); }
};


template<class T, class _Pr>
class TSortedIndexVectorPred : public vector<int> {
public:
  const vector<T> &vect;

  TSortedIndexVectorPred(vector<T> &v, _Pr P)
  : vector<int>(),
    vect(v)
  { 
    for(int i(0), e(vect.size()); i<e; push_back(i++));
    sort(begin(), end(), *this);
  }

  bool operator ()(const int &ind1, const int &ind2) const
  { return P(vect[ind1], vect[ind2]); }
};


void TLinRegLearner::sort_inPlace(gsl_vector *mixed, vector<int> columnOrder)
{
  TSortedIndexVector<int> sorted(columnOrder);
  gsl_vector *temp = gsl_vector_alloc(columnOrder.size());
  gsl_vector_memcpy(temp, mixed);
  int i = 0;
  ITERATE(vector<int>, si, sorted)
    gsl_vector_set(mixed, *si, gsl_vector_get(temp, i++));
  GSL_FREE(temp, vector);
}


PClassifier TLinRegLearner::operator()(PExampleGenerator origen, const int &weightID)
{
  gsl_matrix *X = NULL;
  gsl_vector *y = NULL, *w = NULL;
  gsl_vector *c = NULL, *c_se = NULL;
  int rows, columns;

  if ((iterativeSelection < 0) || (iterativeSelection > 3))
    raiseError("invalid 'iterativeSelection'");

  try {
    PImputer imputer = imputerConstructor ? imputerConstructor->call(origen, weightID) : PImputer();
    PExampleGenerator gen = imputer ? imputer->call(origen, weightID) : origen;

    if (gen->domain->hasDiscreteAttributes(true)) {
      PDomain regDomain = TDomainContinuizer()(gen, weightID);
      gen = mlnew TExampleTable(regDomain, gen);
    }

    exampleGenerator2gsl(gen, weightID, true, multinomialTreatment, X, y, w, rows, columns);

    if (columns==1)
      raiseError("no useful attributes");

    double SSres, SStot, N;
    vector<int> columnOrder;
    Fselection(X, y, w, rows, columns, (iterativeSelection & 1) == 1, (iterativeSelection & 2) == 2, columnOrder, c, c_se, SSres, SStot, N);
    GSL_FREE(X, matrix)
    GSL_FREE(y, vector)
    GSL_FREE(w, vector)

    if (iterativeSelection) {
      sort_inPlace(c, columnOrder);
      sort_inPlace(c_se, columnOrder);
      sort(columnOrder.begin(), columnOrder.end());
    }

    const TVarList &origattr = gen->domain->attributes.getReference();
    PVarList dom_attributes = mlnew TVarList();
    PVarList attributes = mlnew TVarList();
    PAttributedFloatList coeffs = mlnew TAttributedFloatList(attributes, columnOrder.size());
    PAttributedFloatList coeffs_se = mlnew TAttributedFloatList(attributes, columnOrder.size());

    TAttributedFloatList::iterator ci(coeffs->begin()), ce(coeffs->end());
    TAttributedFloatList::iterator ci_se(coeffs_se->begin());
    vector<int>::const_iterator cli(columnOrder.begin()), cle(columnOrder.end());

    for(int i = 0; cli!=cle; cli++, ci++, ci_se++, i++) {
      if (i) {
        attributes->push_back(origattr[*cli-1]);
        dom_attributes->push_back(origattr[*cli-1]);
      }
      else
        attributes->push_back(gen->domain->classVar);

      *ci = float(gsl_vector_get(c, i));
      *ci_se = float(gsl_vector_get(c_se, i));
    }

    GSL_FREE(c, vector)
    GSL_FREE(c_se, vector)


    TLinRegClassifier *classifier = mlnew TLinRegClassifier(mlnew TDomain(origen->domain->classVar, dom_attributes.getReference()), coeffs, coeffs_se, SSres, SStot, rows);
    PClassifier wclassifier(classifier);
    classifier->imputer = imputer;

    if (origen->domain->classVar->varType == TValue::INTVAR)
      classifier->threshold = 0.5; //XXX compute the threshold!!!

    return wclassifier;
  }
  catch (...) {
    GSL_FREE(X, matrix)
    GSL_FREE(y, vector)
    GSL_FREE(w, vector)
    GSL_FREE(c, vector)
    GSL_FREE(c_se, vector)
    throw;
  }
}


TLinRegClassifier::TLinRegClassifier()
: N(0.0),
  SStot(-1.0),
  SSres(-1.0),
  SSreg(-1.0),
  MSres(-1.0),
  MSreg(-1.0),
  F(-1.0),
  Fprob(-1.0),
  R2(-1.0),
  adjR2(-1.0)
{}


TLinRegClassifier::TLinRegClassifier(PDomain dom, PAttributedFloatList coeffs, PAttributedFloatList coeffs_se, const float &SSres, const float &SStot, const float &N)
: TClassifierFD(dom),
  coefficients(coeffs),
  coefficients_se(coeffs_se),
  N(0.0),
  SStot(-1.0),
  SSres(-1.0),
  SSreg(-1.0),
  MSres(-1.0),
  MSreg(-1.0),
  F(-1.0),
  Fprob(-1.0),
  R2(-1.0),
  adjR2(-1.0)
{ setStatistics(SSres, SStot, N); }


TValue TLinRegClassifier::operator()(const TExample &ex)
{ 
  checkProperty(domain)
  TExample cexample = TExample(domain, ex);
  TExample *example = imputer ? imputer->call(cexample) : &cexample;

  try {
    TAttributedFloatList::const_iterator ci(coefficients->begin()), ce(coefficients->end());
    TExample::const_iterator ei(example->begin());
    TVarList::const_iterator vi(domain->attributes->begin());

    float prediction = *(ci++);
    for(; ci!=ce; ci++, ei++, vi++) {
      if ((*ei).isSpecial())
        raiseError("attribute '%s' has unknown value", (*vi)->name.c_str());
      prediction += *ci * ( (*vi)->varType==TValue::INTVAR ? (*ei).intV : (*ei).floatV);
    }

    if (classVar->varType == TValue::FLOATVAR)
      return TValue(prediction);
    else
      return TValue(prediction>threshold ? 1 : 0);
  }
  catch (...) {
    if (imputer)
      mldelete example;
    throw;
  }
}

void TLinRegClassifier::setStatistics(const float &aSSres, const float &aSStot, const float &aN)
{
  const int k = coefficients->size()-1;

  N = aN;
  SSres = aSSres;
  SStot = aSStot;
  SSreg = SSres+SStot;
  MSres = SSres / (N-k-1);
  MStot = SStot / (N-1);
  MSreg = k>0 ? SSreg / k : 0.0;
  
  if (MSres > 1e-10) {
    F = MSreg / MSres;
    Fprob = (F>1e-10) ? fprob(double(k), double(N-k-1), double(F)) : 0.0;
  }
  else {
    F = numeric_limits<float>::max();
    Fprob = 0;
  }
  R2 = 1 - SSres/SStot;
  adjR2 = 1 - MSres/MStot;
}

