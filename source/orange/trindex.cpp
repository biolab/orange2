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

// to include Python.h before STL defines a template set (doesn't work with VC 6.0)
#include "garbage.hpp" 

#include <math.h>
#include <algorithm>
#include <set>

#include "stladdon.hpp"
#include "random.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"

#include "trindex.ppp"


class rsrgen {
public:
  PRandomGenerator randomGenerator;

  rsrgen(const int &seed)
  : randomGenerator(PRandomGenerator(mlnew TRandomGenerator((unsigned long)(seed>=0 ? seed : 0))))
  {}

  rsrgen(PRandomGenerator rgen)
  : randomGenerator(rgen ? rgen : PRandomGenerator(mlnew TRandomGenerator()))
  {}

  rsrgen(PRandomGenerator rgen, const int &seed)
  : randomGenerator(rgen ? rgen : PRandomGenerator(mlnew TRandomGenerator((unsigned long)(seed>=0 ? seed : 0))))
  {}

  int operator()(int n)
  { return randomGenerator->randint(n); }
};


TMakeRandomIndices::TMakeRandomIndices(const int &astratified, const int &arandseed)
: stratified(astratified),
  randseed(arandseed),
  randomGenerator()
{}


TMakeRandomIndices::TMakeRandomIndices(const int &astratified, PRandomGenerator randgen)
: stratified(astratified),
  randseed(-1),
  randomGenerator(randgen)
{}



TMakeRandomIndices2::TMakeRandomIndices2(const float &ap0, const int &astratified, const int &arandseed)
: TMakeRandomIndices(astratified, arandseed),
  p0(ap0)
{}


TMakeRandomIndices2::TMakeRandomIndices2(const float &ap0, const int &astratified, PRandomGenerator randgen)
: TMakeRandomIndices(astratified, randgen),
  p0(ap0)
{}


PRandomIndices TMakeRandomIndices2::operator()(const int &n)
{ return operator()(n, p0); }


PRandomIndices TMakeRandomIndices2::operator()(const int &n, const float &p0)
 { if (stratified==TMakeRandomIndices::STRATIFIED)
     raiseError("cannot prepare stratified indices (no class values)");

   if (!randomGenerator && (randseed<0))
     raiseCompatibilityWarning("object always returns the same indices unless either 'randomGenerator' or 'randseed' is set");

   PRandomIndices indices(mlnew TFoldIndices(n, 1));
   TFoldIndices::iterator ii(indices->begin());

   int no= (p0<=1.0) ? int(p0*n+0.5) : int(p0+0.5);
   if (no>n) no=n;
   while(no--)
     *(ii++)=0;

   rsrgen rg(randomGenerator, randseed);
   or_random_shuffle(indices->begin(), indices->end(), rg);
   return indices;
 }


PRandomIndices TMakeRandomIndices2::operator()(PExampleGenerator gen)
 { return operator()(gen, p0); }


PRandomIndices TMakeRandomIndices2::operator()(PExampleGenerator gen, const float &ap0)
{ 
  if (!gen)
    raiseError("invalid example generator");

   if (!randomGenerator && (randseed<0))
     raiseCompatibilityWarning("object always returns the same indices unless either 'randomGenerator' or 'randseed' is set");

  if (stratified==TMakeRandomIndices::NOT_STRATIFIED)
    return operator()(gen->numberOfExamples(), ap0);

  if (!gen->domain->classVar)
    if (stratified==TMakeRandomIndices::STRATIFIED_IF_POSSIBLE)
      return operator()(gen->numberOfExamples(), ap0);
    else
      raiseError("invalid example generator or class-less domain");

  if (gen->domain->classVar->varType!=TValue::INTVAR)
    if (stratified==TMakeRandomIndices::STRATIFIED_IF_POSSIBLE)
      return operator()(gen->numberOfExamples(), ap0);
    else
      raiseError("cannot prepare stratified indices (non-discrete class values)");
  
  TExampleIterator ri=gen->begin();
  if (!ri)
    return PRandomIndices(mlnew TFoldIndices());
  
  typedef pair<int, int> pii; // index of example, class value
  vector<pii> ricv;

  for(int in=0; ri; ++ri)
    if ((*ri).getClass().isSpecial()) {
      if (stratified==TMakeRandomIndices::STRATIFIED_IF_POSSIBLE)
        return operator()(gen->numberOfExamples(), ap0);
      else
        raiseError("cannot prepare stratified indices (undefined class value(s))");
    }
    else
      ricv.push_back(pii(in++, (*ri).getClass()));

  random_sort(ricv.begin(), ricv.end(),
              predOn2nd<pair<int, int>, less<int> >(), predOn2nd<pair<int, int>, equal_to<int> >(),
              rsrgen(randomGenerator, randseed));

  float p0;
  if (ap0>1.0) {
    if (ap0>ricv.size())
      raiseError("p0 is greater than the number of examples");
    else
      p0 = ap0/float(ricv.size());
  }
  else
    p0 = ap0;

  float p1 = 1-p0;
  float rem = 0;

  PRandomIndices indices(mlnew TFoldIndices());
  indices->resize(ricv.size());
  ITERATE(vector<pii>, ai, ricv)
    if (rem<=0) { 
      indices->at((*ai).first) = 1;
      rem += p0;
    }
    else {
      indices->at((*ai).first) = 0;
      rem -= p1;
    }
  // E.g., if p0 is two times p1, two 0's will cancel one 1.

  return indices;
}



TMakeRandomIndicesN::TMakeRandomIndicesN(const int &astrat, const int &randseed)
: TMakeRandomIndices(astrat, randseed)
{}


TMakeRandomIndicesN::TMakeRandomIndicesN(const int &astrat, PRandomGenerator randgen)
: TMakeRandomIndices(astrat, randgen)
{}


TMakeRandomIndicesN::TMakeRandomIndicesN(PFloatList ap, const int &astrat, const int &randseed)
: TMakeRandomIndices(astrat, randseed),
  p(ap)
{}


TMakeRandomIndicesN::TMakeRandomIndicesN(PFloatList ap, const int &astrat, PRandomGenerator randgen)
: TMakeRandomIndices(astrat, randgen),
  p(ap)
{}


/*  Prepares a vector of given size and with given distribution of elements. Distribution is given as a vector of
    floats and the constructor prepares a vector with elements from 0 to p.size() (therefore p.size()+1 elements
    with the last one having probability 1-sum(p)). */
PRandomIndices TMakeRandomIndicesN::operator()(const int &n)
{ checkProperty(p); // although it is checked later, a better diagnostics can be given here
  return operator()(n, p); }


PRandomIndices TMakeRandomIndicesN::operator()(PExampleGenerator gen)
{ checkProperty(p); // although it is checked later, a better diagnostics can be given here
  return operator()(gen->numberOfExamples(), p); }


PRandomIndices TMakeRandomIndicesN::operator()(PExampleGenerator gen, PFloatList ap)
{ return operator()(gen->numberOfExamples(), ap); }


PRandomIndices TMakeRandomIndicesN::operator()(const int &n, PFloatList ap)
{ 
  if (!ap || !ap->size())
    raiseError("'p' not defined or empty");

  if (!randomGenerator && (randseed<0))
    raiseCompatibilityWarning("object always returns the same indices unless either 'randomGenerator' or 'randseed' is set");
  
  float sum = 0;
  bool props = true;
  for(TFloatList::const_iterator pis(ap->begin()), pie(ap->end()); pis!=pie; pis++) {
    sum += *pis;
    if (*pis > 1.0)
      props = false;
  }

  if (props) {
    if (sum>=1.0)
      raiseError("elements of 'p' sum to 1 or more");
  }
  else {
    if (sum>n)
      raiseError("elements of 'p' sum to more than number of examples");
  } 

  if (stratified==TMakeRandomIndices::STRATIFIED)
    raiseError("stratification not implemented");

  PRandomIndices indices(mlnew TFoldIndices(n, ap->size()));
  TFoldIndices::iterator ii(indices->begin()), ie(indices->end());
  int no, ss=-1;
  PITERATE(TFloatList, pi, ap)
    for(ss++, no = props ? int(*pi*n+0.5) : int(*pi+0.5); no-- && (ii!=ie); *(ii++)=ss);

  rsrgen rg(randomGenerator, randseed);
  or_random_shuffle(indices->begin(), indices->end(), rg);

  return indices;
}


// Prepares a vector of indices for f-fold cross validation with n examples
TMakeRandomIndicesCV::TMakeRandomIndicesCV(const int &afolds, const int &astratified, const int &arandseed)
: TMakeRandomIndices(astratified, arandseed),
  folds(afolds)
{}


TMakeRandomIndicesCV::TMakeRandomIndicesCV(const int &afolds, const int &astratified, PRandomGenerator randgen)
: TMakeRandomIndices(astratified, randgen),
  folds(afolds)
{}


PRandomIndices TMakeRandomIndicesCV::operator()(const int &n)
{ return operator()(n, folds); }


PRandomIndices TMakeRandomIndicesCV::operator()(const int &n, const int &afolds)
{ 
  if (stratified==TMakeRandomIndices::STRATIFIED)
    raiseError("cannot prepare stratified indices (no class values)");

  if (!randomGenerator && (randseed<0))
    raiseCompatibilityWarning("object always returns the same indices unless either 'randomGenerator' or 'randseed' is set");

  if (n<=0)
    raiseError("unknown number of examples");

  if (afolds<=0)
    raiseError("invalid number of folds");

  PRandomIndices indices(mlnew TFoldIndices(n, afolds-1));

  TFoldIndices::iterator ii=indices->begin();
  for(int ss=0; ss<afolds; ss++)
    for(int no=n/afolds+(ss<n%afolds ? 1 : 0); no--; *(ii++)=ss);

  rsrgen rg(randomGenerator, randseed);
  or_random_shuffle(indices->begin(), indices->end(), rg);

  return indices;
}


PRandomIndices TMakeRandomIndicesCV::operator()(PExampleGenerator gen)
{ return operator()(gen, folds); }


PRandomIndices TMakeRandomIndicesCV::operator()(PExampleGenerator gen, const int &afolds)
{
  if (!gen)
    raiseError("invalid example generator");

  if (afolds<=0)
    raiseError("invalid number of folds");


  if (stratified==TMakeRandomIndices::NOT_STRATIFIED)
    return operator()(gen->numberOfExamples(), afolds);

  if (!gen->domain->classVar)
    if (stratified==TMakeRandomIndices::STRATIFIED_IF_POSSIBLE)
      return operator()(gen->numberOfExamples(), afolds);
    else
      raiseError("invalid example generator or class-less domain");

  if (gen->domain->classVar->varType!=TValue::INTVAR)
    if (stratified==TMakeRandomIndices::STRATIFIED_IF_POSSIBLE)
      return operator()(gen->numberOfExamples(), afolds);
    else
      raiseError("cannot prepare stratified indices (non-discrete class values)");
    
  if (!randomGenerator && (randseed<0))
    raiseCompatibilityWarning("object always returns the same indices unless either 'randomGenerator' or 'randseed' is set");

  TExampleIterator ri=gen->begin();
  if (!ri)
    return PRandomIndices(mlnew TFoldIndices());

  typedef pair<int, int> pii; // index of example, class value
  vector<pii> ricv;
  for(int in=0; ri; ++ri) 
    if ((*ri).getClass().isSpecial()) {
      if (stratified==TMakeRandomIndices::STRATIFIED_IF_POSSIBLE)
        return operator()(gen->numberOfExamples(), afolds);
      else
        raiseError("cannot prepare stratified indices (undefined class value(s))");
    }
    else
      ricv.push_back(pii(in++, (*ri).getClass()));

  random_sort(ricv.begin(), ricv.end(),
              predOn2nd<pair<int, int>, less<int> >(), predOn2nd<pair<int, int>, equal_to<int> >(),
              rsrgen(randomGenerator, randseed));

  PRandomIndices indices(mlnew TFoldIndices());
  indices->resize(ricv.size());
  int gr=0;
  ITERATE(vector<pii>, ai, ricv) {
    indices->at((*ai).first)=gr++;
    gr=gr%afolds;
  }

  return indices;
};


class TRndIndCls 
{ public: 
   int rnd, ind, cls;
   TRndIndCls(const int &ar, const int &ai, const int &ac)
     : rnd(ar), ind(ai), cls(ac)
     {}
};

bool compareRnd(const TRndIndCls &fr, const TRndIndCls &sc)
{ return fr.rnd<sc.rnd; }

bool compareCls(const TRndIndCls &fr, const TRndIndCls &sc)
{ return fr.cls<sc.cls; }


void sortedRndIndCls(PExampleGenerator gen, vector<long> rands, vector<TRndIndCls> ricv)
{
  TExampleIterator ri=gen->begin();
  if (!ri)
    raiseError("no examples");
  
  char vt=(*ri).getClass().varType;
  if (vt!=TValue::INTVAR)
    raiseError("cannot perform stratified cross-validation for non-discrete classes");

  ricv.clear();
  vector<long>::const_iterator rndi(rands.begin()), endi(rands.end());

  for(int in=0; ri; ++ri) {
    if ((*ri).getClass().isSpecial())
      raiseError("cannot perform stratified cross-validation when examples have undefined class values");

    ricv.push_back(TRndIndCls(*(rndi++),  in++, (*ri).getClass()));
    if (rndi==endi) rndi=rands.begin();
  }

  sort(ricv.begin(), ricv.end(), compareRnd);
  stable_sort(ricv.begin(), ricv.end(), compareCls);
}



TMakeRandomIndicesMultiple::TMakeRandomIndicesMultiple(const float &ap0, const int &astratified, const int &arandseed)
: TMakeRandomIndices(astratified, arandseed),
  p0(ap0)
{}


TMakeRandomIndicesMultiple::TMakeRandomIndicesMultiple(const float &ap0, const int &astratified, PRandomGenerator randgen)
: TMakeRandomIndices(astratified, randgen),
  p0(ap0)
{}


PRandomIndices TMakeRandomIndicesMultiple::operator()(const int &n)
{ return operator()(n, p0); }


PRandomIndices TMakeRandomIndicesMultiple::operator()(const int &n, const float &p0)
 {
   if (stratified==TMakeRandomIndices::STRATIFIED)
     raiseError("cannot prepare stratified indices (no class values)");

   if (!randomGenerator && (randseed<0))
     raiseCompatibilityWarning("object always returns the same indices unless either 'randomGenerator' or 'randseed' is set");

   int no= (p0<=1.0) ? int(p0*n+0.5) : int(p0+0.5);
   rsrgen rg(randomGenerator, randseed);
   PRandomIndices indices(mlnew TFoldIndices(no, 1));
   PITERATE(TFoldIndices, ii, indices)
     *ii=rg(n);
   return indices;
 }


PRandomIndices TMakeRandomIndicesMultiple::operator()(PExampleGenerator gen)
{ return operator()(gen, p0); }


PRandomIndices TMakeRandomIndicesMultiple::operator()(PExampleGenerator gen, const float &ap0)
{ 
  if (stratified==TMakeRandomIndices::NOT_STRATIFIED)
     return operator()(gen->numberOfExamples(), ap0);

  if (!randomGenerator && (randseed<0))
    raiseCompatibilityWarning("object always returns the same indices unless either 'randomGenerator' or 'randseed' is set");

  TExampleIterator ri=gen->begin();
  if (!ri)
    raiseError("no examples");
  
  if (gen->domain->classVar->varType!=TValue::INTVAR)
     raiseError("cannot prepare stratified indices (non-discrete class values)");

  vector<vector<int> > byclasses=vector<vector<int> >(gen->domain->classVar->noOfValues(), vector<int>());
  long nexamples=0;
  PEITERATE(ei, gen)
    if ((*ei).getClass().isSpecial())
      raiseError("cannot prepare stratified indices (undefined class value(s))");
    else
      byclasses[(*ei).getClass().intV].push_back(nexamples++);

  int no= (p0<=1.0) ? int(p0*nexamples+0.5) : int(p0+0.5);
  rsrgen rg(randomGenerator, randseed);

  PRandomIndices indices(mlnew TFoldIndices());
  
  ITERATE(vector<vector<int> >, clsi, byclasses) {
    int texamples=(*clsi).size();
    for(int i=0, ie=int(0.5 + no * (float(texamples)/nexamples)); i<ie; i++)
      indices->push_back((*clsi)[rg(texamples)]);
  }

  if (int(indices->size())>no)
    indices->erase(indices->begin()+no);
  else 
    while (int(indices->size())<no)
      indices->push_back(rg(nexamples));

  or_random_shuffle(indices->begin(), indices->end(), rg);

  return indices;
}

