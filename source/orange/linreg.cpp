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
#include "rconversions.hpp"

#include "stat.hpp"

#include "examplegen.hpp"
#include "table.hpp"
#include "imputation.hpp"

#include "linreg.ppp"

#include "r_imports.hpp"

TLinRegLearner::TLinRegLearner()
: multinomialTreatment(0),
  iterativeSelection(0),
  Fin(float(0.05)),
  Fout(float(0.1)),
  maxIterations(60000)
{}


// X is destroyed (it contains qr upon return)
void linreg(double *X, double *y, const double &meany, int rows, int columns, double *coeffs, int &rank, int *pivot, double &rss, double &F)
{
  int one = 1;
  double tol = 1e-07;
  
  double *residuals = (double *)malloc(rows * sizeof(double));
  double *effects = (double *)malloc(rows * sizeof(double));
  double *qraux = (double *)malloc(rows * sizeof(double));
  double *work = (double *)malloc(2 * rows * sizeof(double));
  dqrls(X, &rows, &columns, y, &one, &tol, coeffs, residuals, effects, &rank, pivot, qraux, work);
  free(work);
  free(qraux);
  free(effects);

  rss = 0.0;
  double mss = 0.0;
  for(double *ri = residuals, *re = residuals + rows, *yi = y; ri != re; ri++, y++) {
    rss += *ri * *ri;
    const double fted = *yi - meany;
    mss += fted * fted;
  }

  int rdf = rows - rank;
  double resvar = rss / rdf;
  F = mss / (rank - 1) / resvar;
}

    
void unpivot(vector<double> &dest, double *source, int *pivot, int rank, int columns, vector<bool> &selected)
{
  dest.clear();
  selected.clear();
  selected.insert(selected.begin(), columns, false);

  for(; rank--; pivot++) {
    dest.push_back(source[*pivot]);
    selected[*pivot] = true;
  }
}

void compute_se(vector<double> &best_se, const int &columns)
{
  best_se.insert(best_se.begin(), columns, 0);
}

void TLinRegLearner::Fselection(double *X, double*y, double *w, const int &rows, const int &columns,
                                bool forward, bool backward,
                                int *&pivot, int &rank, double *&coeffs, double *&coeffs_se, double *&cov,
                                double &SSres, double &SStot, double &N)
{
  #define SWAP3(x,y,t) { t = x; x = y; y = t; }

  if (!rows)
    raiseError("no examples");

  if (forward && backward && (Fout <= Fin))
    raiseError("Fout should be higher than Fin");


  double Sy = 0.0, SSy = 0.0;
  double avg, F;
  N = 0.0;
  if (w) {
    for(double *yi = y, *ye = y+rows, *wi = w; yi!=ye; yi++, wi++) {
      Sy += *wi * *yi;
      SSy += *wi * *yi * *yi;
      N += *wi;
    }
    if (N == 0.0)
      raiseError("total weight of examples is zero");
  }
  else {
    N = rows;
    for(double *yi = y, *ye = y+rows; yi!=ye; yi++) {
      Sy += *yi;
      SSy += *yi * *yi;
    }
  }

  avg = Sy/N;
  SStot = SSy - Sy*Sy/N;
  SSres = SStot;

  cov = (double *)malloc(columns * rows * sizeof(double));
  coeffs = (double *)malloc(columns * sizeof(double));
  coeffs_se = (double *)malloc(columns * sizeof(double));
  pivot = (int *)malloc(columns * sizeof(int));
  int usedColumns, *pi;
  for(rank = 0, pi = pivot; rank < columns; *pi++ = rank++);

  if (!forward && !backward) {
    linreg(X, y, Sy/N, rows, columns, coeffs, rank, pivot, SSres, F);
    return;
  }

  double *tryCoeffs = NULL;
  double *tryCov = NULL;
  double *tryCoeffs_se = NULL;
  int *tryPivot = NULL;

  try {
    int iterations;
    double SSminimal;
    double F;
    double bestSS = 1e30;

    if (forward) {
      usedColumns = rank = 1;
      *coeffs = avg;
      *coeffs_se = sqrt(SSres/(rows-2)/rows);

      if (backward) {
        iterations = (columns < maxIterations) ? columns : maxIterations;
        float columns3 = float(columns)*columns*columns;
        if (iterations > columns3)
          iterations = columns3;
      }
      else
        iterations = columns;

      SSminimal = avg*avg * 1e-20; // SSres below 1e-20th of average is low enough
      if (SSres < SSminimal)
        return;
    }

    else { // !forward
      iterations = usedColumns = columns;
      memcpy(cov, X, usedColumns * rows);
      linreg(cov, y, Sy/N, rows, columns, coeffs, rank, pivot, SSres, F);
      SSminimal = 1e-20;
    }

    tryCoeffs = (double *)malloc(columns * sizeof(double));
    tryCov = (double *)malloc(rows * columns * sizeof(double));
    tryCoeffs_se = (double *)malloc(columns * sizeof(double));
    tryPivot = (int *)malloc(columns * sizeof(int));

    int tint;
    int *tpint;
    double *tpdouble;

    int changed = 1;
    while(iterations-- && changed-- && (SSres > SSminimal)) {
      for(int direction = forward ? 1 : -1; direction >= -1; direction -= (!backward || (forward && changed)) ? 4 : 2) {
        if ((direction == 1) ? (usedColumns==columns) : (usedColumns <= 2))
          continue;

        int tryRank;
        double trySSres;
        int tryColumns = usedColumns + direction;
        double SSinit = SSres;

        for(int *place = pivot+tryColumns, *sc = place, *se = pivot+columns; sc != se; sc++) {
          SWAP3(*place, *sc, tint);

          double *tryCi = tryCov;
          for(int *pivoti = pivot, *pivote = pivot+tryColumns; pivoti != pivote; pivoti++, tryCi += rows)
            memcpy(tryCi, X + *pivoti * rows, rows*sizeof(double));

          linreg(tryCov, y, Sy/N, rows, columns, tryCoeffs, tryRank, tryPivot, trySSres, F);

          double F = (SSinit - trySSres)*direction / (trySSres/(rows-rank));
          double Fprob = (F>1e-10) ? fprob(1.0, double(rows-rank), F) : 0.0;

          if (((direction==1) ? (Fprob < Fin) : (Fprob > Fout)) && (trySSres < SSres)) {
            SSres = trySSres;
            rank = tryRank;
            SWAP3(cov, tryCov, tpdouble);
            SWAP3(pivot, tryPivot, tpint);
            SWAP3(coeffs, tryCoeffs, tpdouble);
            SWAP3(coeffs_se, tryCoeffs_se, tpdouble);
            changed = 1;
          }
        }

        if (changed) {
          usedColumns = tryColumns;
          memcpy(tryPivot, pivot, usedColumns * sizeof(int));
        }
      }
    }
  }
  catch (...) {
    if (tryCov)       free(tryCov);
    if (tryPivot)     free(tryPivot);
    if (tryCoeffs)    free(tryCoeffs);
    if (tryCoeffs_se) free(tryCoeffs_se);
    free(cov);
    free(pivot);
    free(coeffs);
    free(coeffs_se);
    cov = coeffs = coeffs_se = NULL;
    pivot = NULL;
    throw;
  }

  free(tryCov);
  free(tryPivot);
  free(tryCoeffs);
  free(tryCoeffs_se);

  #undef SWAP3
}


double *TLinRegLearner::unmix(double *mixed, vector<int> columnOrder, int N)
{
  return NULL;
}



class tinred {
public:
  const int *pivots;
  tinred(const int *p) : pivots(p) {}
  bool operator()(const int &a, const int &b) const { return pivots[a] < pivots[b]; }
};


PClassifier TLinRegLearner::operator()(PExampleGenerator origen, const int &weightID)
{
  double *X = NULL;
  double *y = NULL, *w = NULL;
  int rows, columns;

  if ((iterativeSelection < 0) || (iterativeSelection > 3))
    raiseError("invalid 'iterativeSelection'");

  double *cov = NULL;
  double *dcoeffs = NULL, *dcoeffs_se = NULL;
  int *pivot = NULL;

  try {
    PImputer imputer = imputerConstructor ? imputerConstructor->call(origen, weightID) : PImputer();
    PExampleGenerator gen = imputer ? imputer->call(origen, weightID) : origen;

    if (gen->domain->hasDiscreteAttributes(true)) {
      PDomain regDomain = TDomainContinuizer()(gen, weightID);
      gen = mlnew TExampleTable(regDomain, gen);
    }

    exampleGenerator2r(gen, weightID, "1A/Cw", multinomialTreatment, X, y, w, rows, columns);

    if (columns==1)
      raiseError("no useful attributes");

    double SSres, SStot, N;
    int *pivot, rank;
    double *dcoeffs, *dcoeffs_se, *cov;
    Fselection(X, y, w, rows, columns, (iterativeSelection & 1) == 1, (iterativeSelection & 2) == 2, pivot, rank, dcoeffs, dcoeffs_se, cov, SSres, SStot, N);

    const TVarList &origattr = gen->domain->attributes.getReference();
    PVarList dom_attributes = mlnew TVarList(rank);
    PVarList attributes = mlnew TVarList(rank);

    int i, e = rank;
    for(i = 0; pivot[i] && (i != e); i++);
    int interci = i == e ? -1 : i;
    int subint = i == e ? 0 : 1;

    int *spivots = (int *)malloc(rank*sizeof(int));
    for(i = 0; i != e; spivots[i] = i, i++);
    sort(spivots, spivots+rank, tinred(spivots));

    for(i = 0; i != e; i++)
      if (i != interci) {
        const int &pi = pivot[i]-1;
        const int &spi = spivots[i]-subint;
        dom_attributes->at(spi) = origattr[pi];
        attributes->at(spi) = origattr[pi];
    }
    if (interci >= 0)
      attributes->at(rank-1) = dom_attributes->at(rank-1) = gen->domain->classVar;

    TAttributedFloatList *coeffs = mlnew TAttributedFloatList(attributes, rank);
    TAttributedFloatList *coeffs_se = mlnew TAttributedFloatList(attributes, rank);
    PAttributedFloatList wcoeffs = coeffs;
    PAttributedFloatList wcoeffs_se = coeffs_se;

    for(i = 0; i != e; i++)
      if (i != interci) {
        const int &spi = spivots[i]-subint;
        (*coeffs)[spi] = dcoeffs[i];
        (*coeffs_se)[spi] = dcoeffs_se[i];
      }

    if (interci >= 0) {
      (*coeffs)[rank-1] = dcoeffs[interci];
      (*coeffs_se)[rank-1] = dcoeffs[interci];
    }

    TLinRegClassifier *classifier = mlnew TLinRegClassifier(mlnew TDomain(origen->domain->classVar, dom_attributes.getReference()), wcoeffs, wcoeffs_se, SSres, SStot, rows);
    PClassifier wclassifier(classifier);
    classifier->imputer = imputer;

    if (origen->domain->classVar->varType == TValue::INTVAR)
      classifier->threshold = 0.5; //XXX compute the threshold!!!

    return wclassifier;
  }
  catch (...) {
    if (X)          free(X);
    if (y)          free(y);
    if (w)          free(w);
    if (pivot)      free(pivot);
    if (cov)        free(cov);
    if (dcoeffs)    free(dcoeffs);
    if (dcoeffs_se) free(dcoeffs_se);
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
    TAttributedFloatList::const_iterator ci(coefficients->begin()), ce(coefficients->end()-1);
    TExample::const_iterator ei(example->begin());
    TVarList::const_iterator vi(domain->attributes->begin());

    float prediction = 0.0;
    for(; ci!=ce; ci++, ei++, vi++) {
      if ((*ei).isSpecial())
        raiseError("attribute '%s' has unknown value", (*vi)->name.c_str());
      prediction += *ci * ( (*vi)->varType==TValue::INTVAR ? (*ei).intV : (*ei).floatV);
    }
    prediction += *ci;

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

