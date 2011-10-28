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


#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"


#include "examples.hpp"
#include "examplegen.hpp"

#include <vector>
#include <functional>

#include "basstat.ppp"

DEFINE_TOrangeVector_classDescription(PBasicAttrStat, "TBasicAttrStatList", true, ORANGE_API)

// Initializes min to max_float, max to min_float, avg, dev and n to 0.
TBasicAttrStat::TBasicAttrStat(PVariable var, const bool &ahold)
: variable(var),
  holdRecomputation(ahold)
{ reset(); }


TBasicAttrStat::TBasicAttrStat(PExampleGenerator gen, PVariable var, const long &weightID)
: variable(var),
  holdRecomputation(true)
{
  reset();

  if (var->varType != TValue::FLOATVAR)
    raiseError("cannot compute statistics of non-continuous attribute");

  int attrNo = gen->domain->getVarNum(var, false);

  if (attrNo != ILLEGAL_INT) {
    if (!weightID)
      PEITERATE(ei, gen) {
        const TValue &val = (*ei).getValue(attrNo);
        if (!val.isSpecial())
          add(val.floatV);
      }
    else
      PEITERATE(ei, gen) {
        const TValue &val = (*ei).getValue(attrNo);
        if (!val.isSpecial())
          add(val.floatV, WEIGHT(*ei));
      }
  }

  else {
    TVariable &varr = var.getReference();
    if (var->getValueFrom)
      PEITERATE(ei, gen) {
        const TValue &val = varr.computeValue(*ei);
        if (!val.isSpecial())
          add(val.floatV, WEIGHT(*ei));
      }
  }

  holdRecomputation = false;
  recompute();
}

// This is numerically unstable.
// Here is a better algorithm, which also handles higher moments:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Higher-order_statistics

// Adds an example with value f and weight p; n is increased by p, avg by p*f and dev by p*sqr(f)
void TBasicAttrStat::add(float f, float p)
{ sum += p*f;
  sum2 += p*f*f;
  n += p;
  if (!holdRecomputation && (n>0)) {
    avg = sum/n;
    dev = sqrt(std::max(sum2/n - avg*avg, 0.0f));
  }

  if (f<min) 
    min = f;
  if (f>max)
    max = f;
}

void TBasicAttrStat::recompute()
{ if (n>0) {
    avg = sum/n;
    dev = sqrt(std::max(sum2/n - avg*avg, 0.0f));
  }
  else
    avg = dev = -1;
}


void TBasicAttrStat::reset()
{ sum = sum2 = n = avg = dev = 0.0;
  min = numeric_limits<float>::max();
  max = -numeric_limits<float>::max();
}




TDomainBasicAttrStat::TDomainBasicAttrStat()
: hasClassVar(true)
{}


TDomainBasicAttrStat::TDomainBasicAttrStat(PExampleGenerator gen, const long &weightID)
: hasClassVar(gen->domain->classVar)
{ PITERATE(TVarList, vi, gen->domain->variables)
    push_back(((*vi)->varType==TValue::FLOATVAR) ? PBasicAttrStat(mlnew TBasicAttrStat(*vi, true)) : PBasicAttrStat());

  PEITERATE(fi, gen) {
    TExample::iterator ei = (*fi).begin();
    float wei = WEIGHT(*fi);
    for(iterator di(begin()); di!=end(); di++, ei++)
      if (*di && !ei->isSpecial())
        (*di)->add(*ei, wei);
  }

  this_ITERATE(di)
    if (*di) {
      (*di)->holdRecomputation = false;
      (*di)->recompute();
    }
}


/* Removes empty BasicAttrStat (i.e. those at places corresponding to non-continuous attributes */
void TDomainBasicAttrStat::purge()
{ erase(remove_if(begin(), end(), logical_not<PBasicAttrStat>()), end()); }


#include "stat.hpp"

TPearsonCorrelation::TPearsonCorrelation(PExampleGenerator gen, PVariable v1, PVariable v2, const long &weightID)
{
  const bool d1 = v1->varType == TValue::INTVAR;
  const bool d2 = v2->varType == TValue::INTVAR;

  if (   !d1 && (v1->varType != TValue::FLOATVAR)
      || !d2 && (v2->varType != TValue::FLOATVAR))
    raiseError("correlation can only be computed for discrete and continuous attributes");
    
  const int i1 = gen->domain->getVarNum(v1, false);
  const int i2 = gen->domain->getVarNum(v2, false);

  float Sx=0, Sy=0, Sxx=0, Syy=0, Sxy=0, N=0;
  PEITERATE(ei, gen) {
    const TValue &vl1 = i1==ILLEGAL_INT ? v1->computeValue(*ei) : (*ei)[i1];
    const TValue &vl2 = i2==ILLEGAL_INT ? v2->computeValue(*ei) : (*ei)[i2];
    if (vl1.isSpecial() || vl2.isSpecial())
      continue;
      
    const float w = WEIGHT(*ei);
    const float f1 = d1 ? vl1.intV : vl1.floatV;
    const float f2 = d2 ? vl2.intV : vl2.floatV;
    N += w;
    Sx += w*f1;
    Sxx += w*f1*f1;
    Sy += w*f2;
    Syy += w*f2*f2;
    Sxy += w*f1*f2;
  }

  float div = N<1e-10 ? 0 : sqrt((Sxx-Sx*Sx/N)*(Syy-Sy*Sy/N));
  if (div < 1e-10) {
    r = t = 0;
    p = 1;
    df = -1;
    return;
  }
  
  r = (Sxy - Sx*Sy/N) / div;
  if (r == 1) {
    t = 999999;
    p = 0;
    df = -1;
    return;
  }
  
  t = r*sqrt((N-2)/(1-r*r));
  df = int(floor(N));
  
  p=betai(double(df*0.5), double(0.5), double(df/(df+t*t)));
}
