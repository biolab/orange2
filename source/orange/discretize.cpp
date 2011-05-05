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


#include <math.h>

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "getarg.hpp"

#include "classify.hpp"
#include "random.hpp"
#include "distvars.hpp"
#include "basstat.hpp"
#include "contingency.hpp"
#include "transval.hpp"
#include "classfromvar.hpp"

#include "discretize.ppp"


TEquiDistDiscretizer::TEquiDistDiscretizer(const int noi, const float fv, const float st)
: numberOfIntervals(noi),
  firstCut(fv),
  step(st)
{}


// Transforms the value; results is 1+floor((val.floatV-firstCut)/step); 0 if below firstCut, numberOfIntervals if above range 
void TEquiDistDiscretizer::transform(TValue &val)
{ if (val.varType!=TValue::FLOATVAR)
    raiseError("discrete value expected");
  
  if (!val.isSpecial()) {
    if (step<0)
      raiseError("'step' not set");
    if (numberOfIntervals<1)
      raiseError("invalid number of intervals (%i)", numberOfIntervals);

    if ((step==0) || (numberOfIntervals==1))
      val.intV = 0;

    else {
      val.intV = (val.floatV<firstCut) ? 0 : 1+int(floor((val.floatV-firstCut)/step));
      if (val.intV>=numberOfIntervals)
        val.intV = numberOfIntervals-1;
    }
  }
  
  val.varType = TValue::INTVAR;
}


inline int numDecs(const float &diff, float &factor)
{ if (diff>= 1.0) {
    factor = 100.0;
    return 2;
  }
  else {
    int decs = (int)ceil(-log10(diff));
    if (decs<2)
      decs = 2;
    factor = exp(decs*log(10.0));
    return decs;
  }
}


inline float roundFromDecs(const int &decs)
{ 
  return decs <= 0 ? 100.0 : exp(decs*log(10.0));
}

inline void roundToFactor(float &f, const float &factor)
{ f = floor(f*factor+0.5)/factor; }


string mcvt(double f, int decs)
{ 
  char buf[64];
  sprintf(buf, "%.*f", decs, f);
  return buf;
}

/*  Constructs a new TEnumVariable. Its values represent the intervals for values of passed variable var;
    getValueFrom points to a classifier which gets a value of the original variable (var) and transforms it using
    'this' transformer. */
PVariable TEquiDistDiscretizer::constructVar(PVariable var, float mindiff)
{ 
  mindiff = 1.0; // Ignores the given mindiff; see http://www.ailab.si/orange/trac/ticket/576
  TFloatVariable *fvar = var.AS(TFloatVariable);
  if (!fvar)
    raiseError("invalid attribute type (continuous attribute expected)");

  TEnumVariable *evar=mlnew TEnumVariable("D_"+var->get_name());
  PVariable revar(evar);

  evar->ordered = true;

  if (numberOfIntervals<2)
    evar->addValue("C");

  else {
    float roundfactor;
    int decs = numDecs(step<mindiff ? step : mindiff, roundfactor);

    if ((fvar->adjustDecimals != 2) && (decs < fvar->numberOfDecimals)) {
      decs = fvar->numberOfDecimals;
      roundfactor = roundFromDecs(fvar->numberOfDecimals);
    }

    roundToFactor(firstCut, roundfactor);
    roundToFactor(step, roundfactor);

    float f = firstCut;
    string pval;

    pval = mcvt(f, decs);
    evar->addValue(string("<") + pval);

    int steps = numberOfIntervals-2;
    while (steps--) {
      string s("[");
      s += pval;
      f += step;
      s += ", ";
      pval = mcvt(f, decs);
      s += pval;
      s += ")";
      evar->addValue(s);
    }

    evar->addValue(string(">") + pval);
  }
  
  TClassifierFromVar *tcfv = mlnew TClassifierFromVar(revar, var);
  tcfv->transformUnknowns = true;
  tcfv->transformer = this; // rewrapping
  revar->getValueFrom = tcfv;
  return revar;
}


void TEquiDistDiscretizer::getCutoffs(vector<float> &cutoffs) const
{
  cutoffs.clear();
  for(int i = 0; i < numberOfIntervals-1; i++)
    cutoffs.push_back(firstCut+step*i);
}


TThresholdDiscretizer::TThresholdDiscretizer(const float &athreshold)
: threshold(athreshold)
{}


void TThresholdDiscretizer::transform(TValue &val)
{ if (!val.isSpecial())
    val.intV = (val.floatV<=threshold) ? 0 : 1;
  val.varType = TValue::INTVAR;
}


PVariable TThresholdDiscretizer::constructVar(PVariable var, float mindiff)
{ 
  mindiff = 1.0; // Ignores the given mindiff; see http://www.ailab.si/orange/trac/ticket/576
  TEnumVariable *evar = mlnew TEnumVariable("D_"+var->get_name());
  PVariable revar(evar);

  evar->ordered = true;

  char s[10];
  sprintf(s, "<= %5.3f", threshold);
  evar->values->push_back(s);
  sprintf(s, "> %5.3f", threshold);
  evar->values->push_back(s);

  TClassifierFromVar *tcfv = mlnew TClassifierFromVar(revar, var);
  tcfv->transformUnknowns = true;
  tcfv->transformer = this; // rewrapping
  revar->getValueFrom = tcfv;
  return revar;
}


void TThresholdDiscretizer::getCutoffs(vector<float> &cutoffs) const
{
  cutoffs.clear();
  cutoffs.push_back(threshold);
}


TBiModalDiscretizer::TBiModalDiscretizer(const float &al, const float &ah)
: low(al),
  high(ah)
{}


void TBiModalDiscretizer::transform(TValue &val)
{ 
  if (val.varType != TValue::FLOATVAR)
    raiseError("continuous value expected");

  if (!val.isSpecial())
    val.intV = ((val.intV > low) && (val.intV > high)) ? 1 : 0;

  val.varType = TValue::INTVAR;
}


PVariable TBiModalDiscretizer::constructVar(PVariable var, float mindiff)
{ 
  mindiff = 1.0; // Ignores the given mindiff; see http://www.ailab.si/orange/trac/ticket/576
  TFloatVariable *fvar = var.AS(TFloatVariable);
  if (!fvar)
    raiseError("invalid attribute type (continuous attribute expected)");

  TEnumVariable *evar = mlnew TEnumVariable("D_"+var->get_name());
  PVariable revar(evar);

  evar->ordered = true;

  if (high<=low)
    raiseError("invalid interval: (%5.3f, %5.3f]", low, high);

  float roundfactor;
  if (high-low < mindiff) {
    mindiff = high-low;
  }
  int decs = numDecs(mindiff, roundfactor);

  if ((fvar->adjustDecimals != 2) && (decs < fvar->numberOfDecimals)) {
    decs = fvar->numberOfDecimals;
    roundfactor = roundFromDecs(fvar->numberOfDecimals);
  }

  roundToFactor(low, roundfactor);
  roundToFactor(high, roundfactor);
  string lstr = mcvt(low, decs);
  string hstr = mcvt(high, decs);

  evar->values->push_back("<=" + lstr + " or >" + hstr);
  evar->values->push_back("between "+lstr+" and "+hstr);

  TClassifierFromVar *tcfv = mlnew TClassifierFromVar(revar, var);
  tcfv->transformUnknowns = true;
  tcfv->transformer = this; // rewrapping
  revar->getValueFrom = tcfv;
  return revar;
}


void TBiModalDiscretizer::getCutoffs(vector<float> &cutoffs) const
{
  cutoffs.clear();
  cutoffs.push_back(low);
  cutoffs.push_back(high);
}


TIntervalDiscretizer::TIntervalDiscretizer()
: points(mlnew TFloatList())
{}


TIntervalDiscretizer::TIntervalDiscretizer(PFloatList apoints)
: points(apoints)
{};



void TIntervalDiscretizer::transform(TValue &val)
{ checkProperty(points);
  if (val.varType!=TValue::FLOATVAR)
    raiseError("continuous value expected");

  if (!val.isSpecial()) {
    val.intV = 0;
    for(TFloatList::iterator ri(points->begin()), re(points->end()); (ri!=re) && (*ri<val.floatV); ri++, val.intV++);
  }

  val.varType = TValue::INTVAR;
}


/*  Constructs a new TEnumVariable. Its values represent the intervals for
    values of passed variable var; getValueFrom points to a classifier which
    gets a value of the original variable (var) and transforms it using
    'this' transformer. */
PVariable TIntervalDiscretizer::constructVar(PVariable var, float mindiff )
{
  mindiff = 1.0; // Ignores the given mindiff; see http://www.ailab.si/orange/trac/ticket/576
  TFloatVariable *fvar = var.AS(TFloatVariable);
  if (!fvar)
    raiseError("invalid attribute type (continuous attribute expected)");

  TEnumVariable *evar=mlnew TEnumVariable("D_"+var->get_name());
  PVariable revar(evar);

  TEnumVariable *cl_evar=mlnew TEnumVariable("D_"+var->get_name());
  PVariable cl_revar(cl_evar);

  evar->ordered = true;

  if (!points->size())
    evar->addValue("C");

  else {
    TFloatList::iterator vb(points->begin()), ve(points->end()), vi;
    for(vi=vb+1; vi!=ve; vi++) {
      float ndiff = *vi - *(vi-1);
      if (ndiff<mindiff)
        mindiff = ndiff;
    }

    float roundfactor;
    int decs = numDecs(mindiff, roundfactor);

    if ((fvar->adjustDecimals != 2) && (decs < fvar->numberOfDecimals)) {
      decs = fvar->numberOfDecimals;
      roundfactor = roundFromDecs(fvar->numberOfDecimals);
    }

    vi=points->begin();
    string ostr;

    roundToFactor(*vi, roundfactor);    
    ostr = mcvt(*vi, decs);
    evar->addValue(string("<=") + ostr);

    while(++vi!=ve) {
      string s = "(";
      s += ostr;
      s += ", ";
      roundToFactor(*vi, roundfactor);
      ostr = mcvt(*vi, decs);
      s += ostr;
      s += "]";
      evar->addValue(s);
    }

    evar->addValue(string(">")+ostr);
  } 

  TClassifierFromVar *tcfv = mlnew TClassifierFromVar(cl_revar, var);
  tcfv->transformUnknowns = true;
  tcfv->transformer = this; // rewrapping
  revar->getValueFrom = tcfv; 
  return revar;
}



void TIntervalDiscretizer::getCutoffs(vector<float> &cutoffs) const
{
  cutoffs = points.getReference();
}


// Sets the number of intervals (default is 4)
TEquiDistDiscretization::TEquiDistDiscretization(const int anumber)
: TDiscretization(),
  numberOfIntervals(anumber)
{}


// Sets the firstCut and step according to the min and max fields of valStat.
PVariable TEquiDistDiscretization::operator()(PBasicAttrStat valStat, PVariable var) const
{ float step = (valStat->max-valStat->min)/numberOfIntervals;
  PEquiDistDiscretizer discretizer = mlnew TEquiDistDiscretizer(numberOfIntervals, valStat->min+step, step);
  return discretizer->constructVar(var);
}


// Sets the firstCut and step according to the range of values that occur in gen for variable var.
PVariable TEquiDistDiscretization::operator()(PExampleGenerator gen, PVariable var, const long &)
{ if (var->varType!=TValue::FLOATVAR)
    raiseError("attribute '%s' is not continuous", var->get_name().c_str());

  if (numberOfIntervals<=0)
    raiseError("invalid number of intervals (%i)", numberOfIntervals);

  int varPos=gen->domain->getVarNum(var);

  TExampleIterator first(gen->begin());
  while( first && (*first)[varPos].isSpecial() )
    ++first;
  if (!first)
    raiseError("attribute '%s' has no known values", var->get_name().c_str());

  float max, min;
  max = min = (*first)[varPos].floatV;
  while (++first)
    if (!(*first)[varPos].isSpecial()) {
      float val = (*first)[varPos].floatV;
      if (val>max)
        max = val;
      if (val<min)
        min = val;
    };

  float step = (max-min)/numberOfIntervals;
  PEquiDistDiscretizer discretizer = mlnew TEquiDistDiscretizer(numberOfIntervals, min+step, step);
  return discretizer->constructVar(var);
}



TFixedDiscretization::TFixedDiscretization(TFloatList &pts)
: points(mlnew TFloatList(pts))
{}


TFixedDiscretization::TFixedDiscretization(const string &boundaries)
: points()
{ vector<string> atoms;
  string2atoms(boundaries, atoms);
  points = mlnew TFloatList(atoms.size());
  TFloatList::iterator pi(points->begin());
  ITERATE(vector<string>, ai, atoms) {
    sscanf((*ai).c_str(), "%f", &*pi);
    if ((pi!=points->begin()) && (*pi<=pi[-1]))
      raiseError("mismatch in cut-off points");
    pi++;
  }
}


PVariable TFixedDiscretization::operator ()(PExampleGenerator, PVariable var, const long &)
{ PIntervalDiscretizer discretizer = mlnew TIntervalDiscretizer (mlnew TFloatList(points));
  return discretizer->constructVar(var);
}



TEquiNDiscretization::TEquiNDiscretization(int anumber)
: numberOfIntervals(anumber),
  recursiveDivision(true)
{}


PVariable TEquiNDiscretization::operator()(const TContDistribution &distr, PVariable var) const
{ 
  PIntervalDiscretizer discretizer=mlnew TIntervalDiscretizer;
  float mindiff;
  
  if (distr.size() <= numberOfIntervals) {
    cutoffsByMidpoints(discretizer, distr, mindiff);
  }
  else if (recursiveDivision && false) { // XXX remove when the routine is finished
    cutoffsByDivision(discretizer, distr, mindiff);
  }
  else {
    cutoffsByCounting(discretizer, distr, mindiff);
  }

  return discretizer->constructVar(var, mindiff);
}

void TEquiNDiscretization::cutoffsByMidpoints(PIntervalDiscretizer discretizer, const TContDistribution &distr, float &mindiff) const
{
  mindiff = 1.0;
  TContDistribution::const_iterator cdi(distr.begin()), cde(distr.end());
  if (cdi!=cde) {
    float prev = (*cdi).first;
    while (++cdi != cde) {
      discretizer->points->push_back((prev+(*cdi).first)/2.0);
      if (((*cdi).first - prev) < mindiff) {
          mindiff = (*cdi).first - prev;
      }
    }
  }
}

void TEquiNDiscretization::cutoffsByCounting(PIntervalDiscretizer discretizer, const TContDistribution &distr, float &mindiff) const
{
  if (numberOfIntervals<=0)
    raiseError("invalid number of intervals (%i)", numberOfIntervals);

  mindiff = 1.0;
  float N = distr.abs;
  int toGo = numberOfIntervals;
  float inthis = 0, prevel = -1; // initialized to avoid warnings
  float inone = N/toGo;

  for(map<float, float>::const_iterator db(distr.begin()), di(db), de(distr.end()), ni; (toGo>1) && (di!=de); di++) {
    inthis += (*di).second;
    if ((inthis<inone) || (di==db))
      prevel = (*di).first;
    else {
      ni = di; ni++;
      if ((ni!=de) && (inthis - inone < (*di).second / 2)) {
        discretizer->points->push_back( ((*ni).first + (*di).first) /2);
        if ((*ni).first - (*di).first < mindiff) {
          mindiff = (*ni).first - (*di).first;
        }
        N -= inthis;
        inthis = 0;
        prevel = (*ni).first;
      }
      else {
        discretizer->points->push_back( (prevel + (*di).first) / 2);
        if ((*di).first - prevel < mindiff) {
          mindiff = (*di).first - prevel;
        }
        N -= (inthis - ((*di).second));
        inthis = (*di).second;
        prevel = (*di).first;
      }
      if (--toGo) 
        inone = N/toGo;
    }
  }
}


void TEquiNDiscretization::cutoffsByDivision(PIntervalDiscretizer discretizer, const TContDistribution &distr, float &mindiff) const
{ cutoffsByDivision(numberOfIntervals, discretizer->points.getReference(), distr.begin(), distr.end(), distr.abs, mindiff); }


void TEquiNDiscretization::cutoffsByDivision(const int &, TFloatList &, 
                                            map<float, float>::const_iterator, map<float, float>::const_iterator,
                                            const float &, float &) const
{ /*XXX to be finished

  if (noInt & 1) {
    if (noInt & 2) {
      noIntLeft = (noInt-1)/2;
      noIntRight = (noInt+1)/2;
    }
    else {
      noIntLeft = (noInt+1)/2;
      noIntRight = (noInt+1)/2;
    }

    float Nleft = N * noIntLeft / (noIntLeft + noIntRight);
    float Nright = N - Nleft;

    if ((Nleft<1) || (Nright<1))
      return; // should set a cut-off, but couldn't -- N=1...

    map<float, float>::const_iterator fii = fbeg;
    while ((Nn<Nleft) && (fii!=fend))
      Nn += (*fii).second;
    Nn -= (*fii).second;

    if (fii==fend) {
    }

  }
  else {
    float N2 = N/2, Nn = 0.0;
    if (N2<1)
      return; // should set a cut-off, but couldn't -- N=1...

    map<float, float>::const_iterator fii = fbeg;
    while ((Nn<N2) && (fii!=fend))
      Nn += (*fii).second;
    Nn -= (*fii).second;

    if (fii==fend) {
      fii--;
      if (fii==fbeg)
        return; // should set a cut-off, but there's only one value
      else {
        map<float, float>::const_iterator fjj = fii;
        fjj--;
        points.push_back(((*fjj).first + (*fii).first) / 2.0);
        return;
      }
    }

    if (noInt>2) {
      cutoffsByDivision(noInt/2, points, fbeg, fii, Nn);

      map<float, float>::const_iterator fjj = fii;
      fjj--;
      points.push_back(((*fjj).first + (*fii).first) / 2.0);
      
      cutoffsByDivision(noInt/2, points, fii, fend, N-Nn);
    }
  }*/
}

PVariable TEquiNDiscretization::operator()(PExampleGenerator gen, PVariable var, const long &weightID)
{ if (var->varType!=TValue::FLOATVAR)
    raiseError("attribute '%s' is not continuous", var->get_name().c_str());

  int varPos=gen->domain->getVarNum(var);

  TExampleIterator first(gen->begin());
  while(first && (*first)[varPos].isSpecial() )
    ++first;

  if (!first)
    raiseError("attribute '%s' has no known values.", var->get_name().c_str());

  TContDistribution distr(var);
  do {
    TValue &val=(*first)[varPos];
    if (!val.isSpecial())
      distr.addfloat(float(val), WEIGHT(*first));
  } while (++first);

  return operator()(distr, var);
}



// Defined in measures.cpp
float getEntropy(const vector<float> &);


TEntropyDiscretization::TEntropyDiscretization()
: maxNumberOfIntervals(0),
  forceAttribute(false)
{}


PVariable TEntropyDiscretization::operator()(PExampleGenerator gen, PVariable var, const long &weightID)
{ if (!gen->domain->classVar)
    raiseError("class-less domain");

  if (gen->domain->classVar!=TValue::INTVAR)
    raiseError("class '%s' is not discrete", gen->domain->classVar->get_name().c_str());

  if (var->varType!=TValue::FLOATVAR)
    raiseError("attribute '%s' is not continuous", var->get_name().c_str());

  int varPos=gen->domain->getVarNum(var);

  TS S;
  TDiscDistribution all;

  PEITERATE(ei, gen) {
    TValue &val = (*ei)[varPos];
    if (!val.isSpecial()) {
	    const TValue &eclass = (*ei).getClass();
      if (!eclass.isSpecial()) {
  	    float weight = WEIGHT(*ei);
        S[float(val)].addint(int(eclass), weight);
	      all.addint(int(eclass), weight);
      }
    }
  }

  /* No need to initialize seed by number of examples.
     Different number will obviously result in different decisions. */
  TSimpleRandomGenerator rgen;
  return operator()(S, all, var, weightID, rgen);
}


PVariable TEntropyDiscretization::operator()(const TS &S, const TDiscDistribution &all, PVariable var, const long &, TSimpleRandomGenerator &rgen) const
{
  int k=0;
  const_ITERATE(TDiscDistribution, ci, all)
    if (*ci>0)
      k++;

  if (!k)
    raiseError("no examples or all values of attribute '%s' are unknown", var->get_name().c_str());

  float mindiff = 1.0;

  vector<pair<float, float> > points;
  divide(S.begin(), S.end(), all, float(getEntropy(all)), k, points, rgen, mindiff);

  /* This is not correct: if, for instance, we have two cut-off points we should always remove
     the one that was added later... */
  if ((maxNumberOfIntervals>0) && (int(points.size())+1>maxNumberOfIntervals)) {
    random_sort(points.begin(), points.end(), predOn2nd<pair<float, float>, less<float> >(), predOn2nd<pair<float, float>, equal_to<float> >(), rgen);
    points.erase(points.begin()+maxNumberOfIntervals-1, points.end());
    sort(points.begin(), points.end(), predOn1st<pair<float, float>, less<float> >());
  }
    
  PIntervalDiscretizer discretizer = mlnew TIntervalDiscretizer();
  TFloatList &dpoints = dynamic_cast<TFloatList &>(discretizer->points.getReference());
  if (points.size()) {
    vector<pair<float, float> >::const_iterator fi(points.begin()), fe(points.end());
    discretizer->points->push_back((*(fi++)).first);
    for(; fi!=fe; fi++)
      if ((*fi).first != dpoints.back())
        discretizer->points->push_back((*fi).first);
  }

  return discretizer->constructVar(var, mindiff);
}


void TEntropyDiscretization::divide(
  const TS::const_iterator &first, const TS::const_iterator &last,
	const TDiscDistribution &distr, float entropy, int k,
  vector<pair<float, float> > &points,
  TSimpleRandomGenerator &rgen,
  float &mindiff) const
{
  TDiscDistribution S1dist, S2dist = distr, bestS1, bestS2;
  float bestE = -1.0;
  float N = distr.abs;
  int wins = 0;
  TS::const_iterator Ti = first, bestT;
  for(; Ti!=last; Ti++) {
    S1dist += (*Ti).second;
    S2dist -= (*Ti).second;
    if (S2dist.abs==0)
      break;

	  float entro1 = S1dist.abs*float(getEntropy(S1dist))/N;
	  float entro2 = S2dist.abs*float(getEntropy(S2dist))/N;
	  float E = entro1+entro2;
    if (   (!wins || (E<bestE)) && ((wins=1)==1)
        || (E==bestE) && rgen.randbool(++wins)) {
      bestS1 = S1dist;
      bestS2 = S2dist;
      bestE = E;
      bestT = Ti;
    }
  }

  if (!wins)
    return;

  int k1 = 0, k2 = 0;
  ITERATE(TDiscDistribution, ci1, bestS1)
    if (*ci1>0)
      k1++;
  ITERATE(TDiscDistribution, ci2, bestS2)
    if (*ci2>0)
      k2++;

  float entropy1 = float(getEntropy(bestS1));
  float entropy2 = float(getEntropy(bestS2));

  float MDL =  log(float(N-1))/log(2.0)/N
             + (log(exp(k*log(3.0))-2)/log(2.0) - (k*entropy - k1*entropy1 - k2*entropy2))/N;
  float gain = entropy-bestE;

  float cutoff = (*bestT).first;
  bestT++;

  if ((*bestT).first - cutoff < mindiff) {
     mindiff = (*bestT).first - cutoff;
  }

//  cout << cutoff << ", info gain=" << gain << ", MDL=" << MDL << endl;
  if (gain>MDL) {
    if ((k1>1) && (first!=bestT))
      divide(first, bestT, bestS1, entropy1, k1, points, rgen, mindiff);

    points.push_back(pair<float, float>(cutoff, gain-MDL));

    if ((k2>1) && (bestT!=last))
      divide(bestT, last, bestS2, entropy2, k2, points, rgen, mindiff);
  }
  else if (forceAttribute && !points.size())
    points.push_back(pair<float, float>(cutoff, gain-MDL));
}


template<class T> inline T sqr(const T &t)
{ return t*t; }


TBiModalDiscretization::TBiModalDiscretization(const bool sit)
: splitInTwo(sit)
{}


PVariable TBiModalDiscretization::operator()(PExampleGenerator gen, PVariable var, const long &weightID)
{ if (var->varType!=TValue::FLOATVAR)
    raiseError("attribute '%s' is not continuous", var->get_name().c_str());
  if (gen->domain->classVar!=TValue::INTVAR)
    raiseError("class '%s' is not discrete", gen->domain->classVar->get_name().c_str());
  
  TContingencyAttrClass ccont(gen, var, weightID);
  int nClasses = gen->domain->classVar->noOfValues();
  float best1, best2;
  float bestEval = -99999;

  PDistribution classDist = getClassDistribution(gen, weightID);
  TDiscDistribution &totDist = dynamic_cast<TDiscDistribution &>(classDist.getReference());
  totDist.normalize();

  // middle will contain sum of distributions from cut1 (exclusive) to cut2 (inclusive)
  for(TDistributionMap::iterator cut1(ccont.continuous->begin()), cute(ccont.continuous->end()); cut1!=cute; cut1++) {
    TDiscDistribution middle(nClasses);

    TDistributionMap::iterator cut2 = cut1;
    for(cut2++; cut2!=cute; cut2++) {
      middle += (*cut2).second;

      float chisq = 0.0;
      float tabs = middle.abs;
      int N = nClasses;
      for(TDiscDistribution::const_iterator toti = totDist.begin(), midi = middle.begin();  N--; toti++, midi++) {
        const float E = tabs**toti;
        const float &n = *midi;
        chisq += sqr( fabs(E - n) - 0.5 ) / E;
      }

      if (chisq > bestEval) {
        bestEval = chisq;
        best1 = (*cut1).first;
        best2 = (*cut2).first;
      }
    }
  }

  PDiscretizer discretizer;

  if (splitInTwo)
    discretizer = mlnew TBiModalDiscretizer(best1, best2);

  else {
    TIntervalDiscretizer *idisc = mlnew TIntervalDiscretizer;
    discretizer = idisc;
    idisc->points->push_back(best1);
    idisc->points->push_back(best2);
  }

  return discretizer->constructVar(var);
}
  
 

TDomainDiscretization::TDomainDiscretization(PDiscretization adisc)
: discretization(adisc)
{}


PDomain TDomainDiscretization::equiDistDomain(PExampleGenerator gen)
{
  PDomain newDomain = mlnew TDomain();
  newDomain->metas = gen->domain->metas;

  TDomainBasicAttrStat valStats(gen);
  const TEquiDistDiscretization &discs = dynamic_cast<TEquiDistDiscretization &>(discretization.getReference());

  TVarList::iterator vi=gen->domain->variables->begin();
  ITERATE(TDomainBasicAttrStat, si, valStats)
    if (*si) {
      PVariable evar=discs(*si, *vi);

      newDomain->variables->push_back(evar);
      newDomain->attributes->push_back(evar);
      vi++;
    }
    else {
      newDomain->variables->push_back(*vi);
      newDomain->attributes->push_back(*vi);
      vi++;
    }

  if (gen->domain->classVar) {
    newDomain->classVar=newDomain->variables->back();
    newDomain->attributes->erase(newDomain->attributes->end()-1);
  }

  return newDomain;
}


PDomain TDomainDiscretization::equiNDomain(PExampleGenerator gen, const long &weightID)
{
  PDomain newDomain = mlnew TDomain();
  newDomain->metas = gen->domain->metas;
  TDomainDistributions valDs(gen, weightID);

  const TEquiNDiscretization &discs = dynamic_cast<TEquiNDiscretization &>(discretization.getReference());

  TVarList::iterator vi=gen->domain->variables->begin();
  ITERATE(TDomainDistributions, si, valDs)
    if ((*si)->variable->varType==TValue::FLOATVAR) {
      PVariable evar = discs(CAST_TO_CONTDISTRIBUTION(*si), *vi);

      newDomain->variables->push_back(evar);
      newDomain->attributes->push_back(evar);
      vi++;
    }
    else {
      newDomain->variables->push_back(*vi);
      newDomain->attributes->push_back(*vi);
      vi++;
    }

  if (gen->domain->classVar) {
    newDomain->classVar = newDomain->variables->back();
    newDomain->attributes->erase(newDomain->attributes->end()-1);
  }

  return newDomain;
}


PDomain TDomainDiscretization::otherDomain(PExampleGenerator gen, const long &weightID)
{
  PDomain newDomain = mlnew TDomain();
  newDomain->metas = gen->domain->metas;

  PITERATE(TVarList, vi, gen->domain->variables)
    if ((*vi)->varType==TValue::FLOATVAR) {
      PVariable evar=discretization->operator()(gen, *vi, weightID);

      newDomain->variables->push_back(evar);
      newDomain->attributes->push_back(evar);
    }
    else {
      newDomain->variables->push_back(*vi);
      newDomain->attributes->push_back(*vi);
    }

  if (gen->domain->classVar) {
    newDomain->classVar=newDomain->variables->back();
    newDomain->attributes->erase(newDomain->attributes->end()-1);
  }

  return newDomain;
}


PDomain TDomainDiscretization::operator()(PExampleGenerator gen, const long &weightID)
{ checkProperty(discretization);

  if (discretization.is_derived_from(TEquiDistDiscretization))
    return equiDistDomain(gen);
  if (discretization.is_derived_from(TEquiNDiscretization))
    return equiNDomain(gen, weightID);

  return otherDomain(gen, weightID);
}

