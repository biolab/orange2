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

#include "examplegen.hpp"
#include "domain.hpp"
#include "basstat.hpp"

#include "pnn.ppp"

TPNN::TPNN(PDomain domain, const int &alaw, const bool normalize)
: TClassifierFD(domain, true),
  dimensions(0),
  offsets(),
  normalizers(),
  normalizeExamples(normalize),
  bases(NULL),
  nExamples(0),
  projections(NULL),
  law(alaw)
{}


TPNN::TPNN(PDomain domain, PExampleGenerator egen, double *bases, const int &alaw, const bool)
{ raiseError("not implemented yet"); }


TPNN::TPNN(PDomain domain, double *examples, const int &nEx, double *ba, const int &dim, PFloatList off, PFloatList norm, const int &alaw, const bool normalize)
: TClassifierFD(domain),
  dimensions(dim),
  offsets(off),
  normalizers(norm),
  normalizeExamples(normalize),
  bases((double *)memcpy(new double[domain->attributes->size()*dim], ba, domain->attributes->size()*dim*sizeof(double))),
  radii(new double[domain->attributes->size()]),
  nExamples(nEx),
  projections(new double[dim*nEx]),
  minClass(0),
  maxClass(0),
  law(alaw)
{
  const int nAttrs = domain->attributes->size();
  TFloatList::const_iterator offi, offb = offsets->begin(), offe = offsets->end();
  TFloatList::const_iterator nori, norb = normalizers->begin(), nore = normalizers->end();

  for(double *base = bases, *basee = base + nAttrs * dim, *radius = radii; base != basee; radius++) {
    for(int d = dim; d--; *radius += sqr(*base++));
    *radius = sqrt(*radius);
  }
  
  double *pi, *pe;
  for(pi = projections, pe = projections + (dim+1)*nEx; pi != pe; *(pi++) = 0.0);

  const bool contClass = domain->classVar->varType == TValue::FLOATVAR;

  for(double *example = examples, *examplee = examples + nEx*dimensions, *projection = projections; example != examplee; projection = pe) {
    offi = offb;
    nori = norb;
    pe = projection + dimensions;
    double *base = bases, *radius = radii;
    double asum = 0.0;
    for(double *ee = example + nAttrs; example != ee; example) {
      double aval = (*(example++) - *(offi++)) / *(nori++);
      for(pi = projection; pi != pe; *(pi++) += aval * *(base++));
      if (normalizeExamples)
        asum += aval * *radius++;
    }
    if (normalizeExamples && (asum > 0.0))
      for(pi = projection; pi != pe; *(pi++) /= asum);

    if (contClass) {
      if (example == examples+dimensions-1)
        minClass = maxClass = *example;
      else {
        if (*example < minClass)
          minClass = *example;
        else if (*example > maxClass)
          maxClass = *example;
      }
    }

    *pe++ = *example++; // copy the class
  }
}


TPNN::TPNN(PDomain domain, double *examples, const int &nEx, double *ba, const int &dim, PFloatList off, PFloatList norm, const int &alaw, const vector<int> &attrIndices, int &nOrigRow, const bool normalize)
: TClassifierFD(domain),
  dimensions(dim),
  offsets(off),
  normalizers(norm),
  normalizeExamples(normalize),
  bases((double *)memcpy(new double[domain->attributes->size()*dim], ba, domain->attributes->size()*dim*sizeof(double))),
  radii(new double[domain->attributes->size()]),
  nExamples(nEx),
  projections(new double[dim*nEx]),
  law(alaw)
{
  const int nAttrs = domain->attributes->size();
  TFloatList::const_iterator offi, offb = offsets->begin(), offe = offsets->end();
  TFloatList::const_iterator nori, norb = normalizers->begin(), nore = normalizers->end();

  for(double *base = bases, *basee = base + nAttrs * dim, *radiii = radii; base != basee; radii++) {
    for(int d = dim; d--; *radii += sqr(*base++));
    *radii = sqrt(*radii);
  }
  
  double *pi, *pe;
  for(pi = projections, pe = projections + (dim+1)*nEx; pi != pe; *(pi++) = 0.0);

  const bool contClass = domain->classVar->varType == TValue::FLOATVAR;

  for(double *example = examples, *examplee = examples + nEx*dimensions, *projection = projections; example != examplee; projection = pe, example += nOrigRow) {
    offi = offb;
    nori = norb;
    pe = projection + dimensions;
    double *base = bases, *radius = radii;
    double asum = 0.0;
    const_ITERATE(vector<int>, ai, attrIndices) {
      double aval = (example[*ai] - *(offi++)) / *(nori++);
      for(pi = projection; pi != pe; *(pi++) += aval * *(base++));
      if (normalizeExamples)
        asum += aval * *radius++;
    }
    if (normalizeExamples && (asum > 0.0))
      for(pi = projection; pi != pe; *(pi++) /= asum);

    const double cls = example[nOrigRow-1];

    if (contClass) {
      if (example == examples+dimensions-1)
        minClass = maxClass = cls;
      else {
        if (cls < minClass)
          minClass = cls;
        else if (cls > maxClass)
          maxClass = cls;
      }
    }

    *pe++ = cls; // copy the class
  }
}


TPNN::TPNN(const int &nDim, const int &nAtt, const int &nEx)
: dimensions(nDim),
  bases(new double[2*nAtt]),
  radii(new double[2*nAtt]),
  nExamples(nEx),
  projections(new double[3*nExamples])
{}


TPNN::TPNN(const TPNN &old)
: TClassifierFD(old),
  dimensions(0),
  bases(NULL),
  radii(NULL),
  nExamples(0),
  projections(NULL)
{ *this = old; }


TPNN &TPNN::operator =(const TPNN &old)
{
  if (bases)
    delete bases;

  const int nAttrs = domain->attributes->size();

  if (bases)
    delete bases;
  bases = old.bases ? (double *)memcpy(new double[nAttrs*dimensions], old.bases, nAttrs*dimensions*sizeof(double)) : NULL;
  
  if (radii)
    delete radii;
  radii = old.radii ? (double *)memcpy(new double[nAttrs], old.radii, nAttrs*sizeof(double)) : NULL;

  if (projections)
    delete projections;
  projections = old.projections ? (double *)memcpy(new double[nExamples*(dimensions+1)], old.projections, nExamples*(dimensions+1)*sizeof(double)) : NULL;

  if (old.offsets)
    offsets = new TFloatList(old.offsets.getReference());
  else
    offsets = PFloatList();

  if (old.normalizers)
    normalizers = new TFloatList(old.normalizers.getReference());
  else
    normalizers = PFloatList();

  nExamples = old.nExamples;
  law = old.law;
  normalizeExamples = old.normalizeExamples;
  minClass = old.minClass;
  maxClass = old.maxClass;

  return *this;
}


TPNN::~TPNN()
{
  if (bases)
    delete bases;

  if (projections)
    delete projections;

  if (radii)
    delete radii;
}


void TPNN::project(const TExample &example, double *projection)
{
  TFloatList::const_iterator offi = offsets->begin(), nori = normalizers->begin();

  double *pi, *pe = projection + dimensions;
  for(pi = projection; pi != pe; *(pi++) = 0.0);

  double *base = bases;
  double *radius = radii;
  double asum = 0.0;

  for(TExample::const_iterator ei = example.begin(), ee = example.end(); ei != ee; ) {
    if ((*ei).isSpecial())
      raiseError("cannot handle missing values");

    double aval = ((*(ei++)).floatV - *(offi++)) / *(nori++);
    for(pi = projection; pi != pe; *(pi++) += aval * *(base++));
    if (normalizeExamples)
      asum +=aval * *radius++;
  }
  if (normalizeExamples)
    for(pi = projection; pi != pe; *(pi++) /= asum);
}


PDistribution TPNN::classDistribution(const TExample &example)
{
  double *projection = mlnew double[dimensions];
  double *pe = projection + dimensions;

  const int nClasses = domain->classVar->noOfValues();
  float *cprob = mlnew float[nClasses];
  for(float *ci = cprob, *ce = cprob + nClasses; ci != ce; *(ci++) = 0.0)
  
  try {
    if (example.domain == domain)
      project(example, projection);
    else {
      TExample nex(domain, example);
      project(example, projection);
    }

    for(double *proj = projections, *proje = projections+ nExamples*(dimensions+1); proj != proje; ) {
      double dist = 0.0;
      double *pi = projection;
      while(pi!=pe)
        dist += sqr(*pi - *(proj++));
      if (dist < 1e-5)
        dist = 1e-5;
      switch(law) {
        case InverseLinear: cprob[int(*(proj++))] += 1/sqrt(dist); break;
        case InverseSquare: cprob[int(*(proj++))] += 1/dist; break;
        case InverseExponential: 
        case KNN: cprob[int(*(proj++))] += exp(-sqrt(dist)); break;
      }
    }

    TDiscDistribution *dist = mlnew TDiscDistribution(cprob, nClasses);
    PDistribution wdist = dist;
    dist->normalize();
    return wdist;
  }
  catch (...) {
    delete projection;
    delete cprob;
    throw;
  }

  delete projection;
  delete cprob;

  return PDistribution();
}




TP2NN::TP2NN(PDomain domain, PExampleGenerator egen, PFloatList basesX, PFloatList basesY, const int &alaw, const bool normalize)
: TPNN(domain, alaw, normalize)
{ 
  dimensions = 2;
  nExamples = egen->numberOfExamples();

  const int nAttrs = domain->attributes->size();

  if ((basesX->size() != nAttrs) || (basesY->size() != nAttrs))
    raiseError("the number of used attributes, x- and y-anchors coordinates mismatch");

  bases = new double[2*domain->attributes->size()];
  radii = new double[domain->attributes->size()];

  double *bi, *radiii;
  TFloatList::const_iterator bxi(basesX->begin()), bxe(basesX->end());
  TFloatList::const_iterator byi(basesY->begin());
  for(radiii = radii, bi = bases; bxi != bxe; *radiii++ = sqrt(sqr(*bxi) + sqr(*byi)), *bi++ = *bxi++, *bi++ = *byi++);

  const TDomain &gendomain = egen->domain.getReference();
  vector<int> attrIdx;
  attrIdx.reserve(nAttrs);

  offsets = new TFloatList();
  normalizers = new TFloatList();
  averages = new TFloatList();

  TDomainDistributions ddist(egen, 0, false, true); // skip continuous

  if (domain->hasContinuousAttributes()) {
    TDomainBasicAttrStat basstat(egen);
    
    const_PITERATE(TVarList, ai, domain->attributes) {
      const int aidx = gendomain.getVarNum(*ai);
      attrIdx.push_back(aidx);
      if ((*ai)->varType == TValue::INTVAR) {
        offsets->push_back(0);
        normalizers->push_back((*ai)->noOfValues() - 1);
        averages->push_back(float(ddist[aidx]->highestProbIntIndex()));
      }
      else if ((*ai)->varType == TValue::FLOATVAR) {
        if (aidx < 0)
          raiseError("P2NN does not accept continuous meta attributes");

        offsets->push_back(basstat[aidx]->min);
        normalizers->push_back(basstat[aidx]->max - basstat[aidx]->min);
        averages->push_back(basstat[aidx]->avg);;
      }
      else
        raiseError("P2NN can only handle discrete and continuous attributes");
    }
  }
  else {
    const_PITERATE(TVarList, ai, domain->attributes) 
      if ((*ai)->varType != TValue::INTVAR)
        raiseError("P2NN can only handle discrete and continuous attributes");
      else {
        const int aidx = gendomain.getVarNum(*ai);
        attrIdx.push_back(aidx);
        offsets->push_back(0.0);
        normalizers->push_back((*ai)->noOfValues()-1);
        averages->push_back(float(ddist[aidx]->highestProbIntIndex()));
      }
  }

  const int &classIdx = gendomain.getVarNum(domain->classVar);
  
  projections = new double[3*egen->numberOfExamples()];
  double *pi, *pe;
  for(pi = projections, pe = projections + 3*nExamples; pi != pe; *(pi++) = 0.0);

  const bool contClass = domain->classVar->varType == TValue::FLOATVAR;

  pi = projections;
  PEITERATE(ei,egen) {
    TValue &cval = (*ei)[classIdx];
    if (cval.isSpecial())
      continue;

    TFloatList::const_iterator offi(offsets->begin());
    TFloatList::const_iterator nori(normalizers->begin());
    TFloatList::const_iterator avgi(averages->begin());
    vector<int>::const_iterator ai(attrIdx.begin()), ae(attrIdx.end());
    double *base = bases;
    radiii = radii;
    double sumex = 0.0;
    for(; ai!=ae; ai++, offi++, nori++, avgi++) {
      const TValue &val = (*ei)[*ai];
      double av;
      if (val.isSpecial())
        av = *avgi;
      else
        av = val.varType == TValue::INTVAR ? float(val.intV) : val.floatV;

      const double aval = (av - *offi) / *nori;
      pi[0] += aval * *base++;
      pi[1] += aval * *base++;
      sumex += aval * *radiii++;
    }
    if (normalizeExamples && (sumex > 0.0)) {
      pi[0] /= sumex;
      pi[1] /= sumex;
    }

    pi[2] = cval.varType == TValue::INTVAR ? float(cval.intV) : cval.floatV;
    pi += 3;

    if (contClass) {
      if (pi == projections + 3)
        minClass = maxClass = cval.floatV;
      else {
        if (cval.floatV < minClass)
          minClass = cval.floatV;
        else if (cval.floatV > maxClass)
          maxClass = cval.floatV;
      }
    }
  }
}


/* This one projects the examples; removed in favour of the one that gets the computed projections 
TP2NN::TP2NN(PDomain, double *examples, const int &nEx, double *ba, PFloatList off, PFloatList norm, PFloatList avgs, const int &alaw, const bool normalize)
: TPNN(domain, alaw, normalize)
{
  dimensions = 2;
  offsets = off;
  normalizers = norm;
  averages = avgs;

  nExamples = nEx;

  const int nAttrs = domain->attributes->size();
  TFloatList::const_iterator offi, offb = offsets->begin(), offe = offsets->end();
  TFloatList::const_iterator nori, norb = normalizers->begin(), nore = normalizers->end();

  bases = (double *)memcpy(new double[domain->attributes->size()*2], ba, domain->attributes->size()*2*sizeof(double));

  double *radiii, *radiie, *bi;
  for(radiii = radii, radiie = radii + nAttrs, bi = bases; radiii != radiie; *radiii++ = sqrt(sqr(*bi++) + sqr(*bi++)));

  projections = new double[2*nEx];
  double *pi, *pe;
  for(pi = projections, pe = projections + 3*nEx; pi != pe; *(pi++) = 0.0);

  double *example, *examplee;
  for(example = examples, examplee = examples + nExamples*(nAttrs+1), pi = projections; example != examplee; pi += 3) {
    offi = offb;
    nori = norb;
    double *base = bases;
    radiii = radii;
    double sumex = 0.0;
    for(double *ee = example + nAttrs; example != ee; example) {
      double aval = (*(example++) - *(offi++)) / *(nori++);
      pi[0] += aval * *(base++);
      pi[1] += aval * *(base++);
      if (normalizeExamples)
        sumex += aval * *radiii++;
    }
    if (normalizeExamples && (sumex > 0.0)) {
      pi[0] /= sumex;
      pi[1] /= sumex;
    }
    pi[2] = *example++;
  }
}
*/


TP2NN::TP2NN(PDomain dom, double *aprojections, const int &nEx, double *ba, PFloatList off, PFloatList norm, PFloatList avgs, const int &alaw, const bool normalize)
: TPNN(dom, alaw, normalize)
{
  dimensions = 2;
  offsets = off;
  normalizers = norm;
  averages = avgs;
  bases = ba;
  projections = aprojections;
  nExamples = nEx;

  if (bases) {
    radii = mlnew double[2*domain->attributes->size()];
    for(double *radiii = radii, *radiie = radii + domain->attributes->size(), *bi = bases;
        radiii != radiie;
        *radiii++ = sqrt(sqr(*bi++) + sqr(*bi++)));
  }
  else
    radii = NULL;

  if (dom->classVar->varType == TValue::FLOATVAR) {
    double *proj = projections+2, *proje = projections + 3*nEx + 2;
    minClass = maxClass = *proj;
    while( (proj+=3) != proje ) {
      if (*proj < minClass)
        minClass = *proj;
      else if (*proj > maxClass)
        maxClass = *proj;
    }
  }
}


TP2NN::TP2NN(const int &nAtt, const int &nEx)
: TPNN(2, nAtt, nEx)
{}

void TP2NN::project(const TExample &example, double &x, double &y)
{
  TFloatList::const_iterator offi = offsets->begin(), nori = normalizers->begin(), avgi = averages->begin();
  x = y = 0.0;
  double *base = bases, *radius = radii;
  double sumex = 0.0;

  TExample::const_iterator ei = example.begin();
  for(int attrs = example.domain->attributes->size(); attrs--; ei++, avgi++, offi++, nori++) {
    double av;
    if ((*ei).isSpecial())
      av = *avgi;
    else
      av = (*ei).varType == TValue::INTVAR ? float((*ei).intV) : (*ei).floatV;

    const double aval = (av - *offi) / *nori;
    x += aval * *(base++);
    y += aval * *(base++);
    if (normalizeExamples)
      sumex += aval * *radius++;
  }
  if (normalizeExamples) {
    x /= sumex;
    y /= sumex;
  }
}


TValue TP2NN::operator ()(const TExample &example)
{
  checkProperty(offsets);
  checkProperty(normalizers);
  checkProperty(averages);
  checkProperty(bases);
  if (normalizeExamples)
    checkProperty(radii);

  if (classVar->varType == TValue::INTVAR)
    return TClassifier::call(example);

  double x, y;
  getProjectionForClassification(example, x, y);
  return TValue(float(averageClass(x, y)));
}
    


PDistribution TP2NN::classDistribution(const TExample &example)
{
  checkProperty(offsets);
  checkProperty(normalizers);
  checkProperty(averages);
  checkProperty(bases);
  if (normalizeExamples)
    checkProperty(radii);


  double x, y;
  getProjectionForClassification(example, x, y);
  
  if (classVar->varType == TValue::FLOATVAR) {
    PContDistribution cont = mlnew TContDistribution(classVar);
    cont->addfloat(float(averageClass(x, y)));
    return cont;
  }

  else {
    const int nClasses = domain->classVar->noOfValues();
    float *cprob = mlnew float[nClasses];

    try {
      classDistribution(x, y, cprob, nClasses);
      PDiscDistribution wdist = mlnew TDiscDistribution(cprob, nClasses);
      wdist->normalize();
      return wdist;
    }
    catch (...) {
      delete cprob;
      throw;
    }
  }

  return PDistribution();
}


void TP2NN::classDistribution(const double &x, const double &y, float *distribution, const int &nClasses) const
{
  for(float *ci = distribution, *ce = distribution + nClasses; ci != ce; *ci++ = 0.0);
  double *proj = projections, *proje = projections + 3*nExamples;

  switch(law) {
    case InverseLinear:
    case Linear:
      for(; proj != proje; proj += 3) {
        const double dist = sqr(proj[0] - x) + sqr(proj[1] - y);
        distribution[int(proj[2])] += dist<1e-8 ? 1e4 : 1.0/sqrt(dist);
      }
      return;

    case InverseSquare:
      for(; proj != proje; proj += 3) {
        const double dist = sqr(proj[0] - x) + sqr(proj[1] - y);
        distribution[int(proj[2])] += dist<1e-8 ? 1e8 : 1.0/dist;
      }
      return;

    case InverseExponential:
    case KNN:
      for(; proj != proje; proj += 3) {
        const double dist = sqr(proj[0] - x) + sqr(proj[1] - y);
        distribution[int(proj[2])] += exp(-sqrt(dist));
      }
      return;
  }
}


double TP2NN::averageClass(const double &x, const double &y) const
{
  double sum = 0.0;
  double N = 0.0;
  double *proj = projections, *proje = projections + 3*nExamples;

  switch(law) {
    case InverseLinear:
    case Linear:
      for(; proj != proje; proj += 3) {
        const double dist = sqr(proj[0] - x) + sqr(proj[1] - y);
        const double w = dist<1e-8 ? 1e4 : 1.0/sqrt(dist);
        sum += w * proj[2]; 
        N += w;
      }
      break;

    case InverseSquare:
      for(; proj != proje; proj += 3) {
        const double dist = sqr(proj[0] - x) + sqr(proj[1] - y);
        const double w = dist<1e-8 ? 1e4 : 1.0/dist;
        sum += w * proj[2]; 
        N += w;
      }
      break;

    case InverseExponential:
    case KNN:
      for(; proj != proje; proj += 3) {
        const double dist = sqr(proj[0] - x) + sqr(proj[1] - y);
        const double w = dist<1e-8 ? 1e4 : exp(-sqrt(dist));
        sum += w * proj[2]; 
        N += w;
      }
      break;
  }

  return N > 1e-4 ? sum/N : 0.0;
}
