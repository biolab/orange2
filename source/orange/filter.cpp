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


#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "stladdon.hpp"
#include "random.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "distvars.hpp"
#include "examplegen.hpp"

#include "filter.ppp"


DEFINE_TOrangeVector_classDescription(PValueFilter, "TValueFilterList")

// Sets the negate field (default is false)
TFilter::TFilter(bool anegate, PDomain dom) 
  : negate(anegate), domain(dom)
  {}

void TFilter::reset()
{}

// Sets the maxrand field to RAND_MAX*ap
TFilter_random::TFilter_random(const float ap, bool aneg, PDomain dom)
  : TFilter(aneg, dom), prob(ap) 
  {};

// Chooses an example (returns true) if rand()<maxrand; example is ignored
bool TFilter_random::operator()(const TExample &)
  { return (LOCAL_OR_GLOBAL_RANDOM.randfloat()<prob)!=negate; }



TFilter_hasSpecial::TFilter_hasSpecial(bool aneg, PDomain dom)
  : TFilter(aneg, dom)
  {}


// Chooses an example if it has (no) special values.
bool TFilter_hasSpecial::operator()(const TExample &exam)
{ int i=0, Nv;
  if (domain) {
    TExample example(domain, exam);
    for(Nv=domain->variables->size(); (i<Nv) && !example[i].isSpecial(); i++);
  }
  else for(Nv=exam.domain->variables->size(); (i<Nv) && !exam[i].isSpecial(); i++);

  return ((i==Nv)==negate);
}


TFilter_hasClassValue::TFilter_hasClassValue(bool aneg, PDomain dom)
  : TFilter(aneg, dom)
  {}

// Chooses an example if it has (no) special values.
bool TFilter_hasClassValue::operator()(const TExample &exam)
{ return (domain ? TExample(domain, exam).getClass().isSpecial() : exam.getClass().isSpecial()) ==negate; }


// Constructor; sets the value and position
TFilter_sameValue::TFilter_sameValue(const TValue &aval, int apos, bool aneg, PDomain dom)
: TFilter(aneg, dom), 
  position(apos),
  value(aval)
{}


// Chooses an example if position-th attribute's value equals (or not) the specified value
bool TFilter_sameValue::operator()(const TExample &example)
{ if (position<0) return negate;
  signed char equ=(domain ? TExample(domain, example)[position] : example[position])==value;
  return (equ!=-1) && (negate!=true);
}


TValueFilter::TValueFilter(const int &accs)
: acceptSpecial(accs)
{}


TValueFilter_continuous::TValueFilter_continuous(const float &amin, const float &amax, const bool &outs, const int &accs)
: TValueFilter(accs),
  min(amin),
  max(amax),
  outside(outs)
{}


int TValueFilter_continuous::operator()(const TValue &val) const
{ if (val.isSpecial())
    return acceptSpecial;

  return ((val.floatV>=min) && (val.floatV<=max)) != outside ? 1 : 0;
}


TValueFilter_discrete::TValueFilter_discrete(PBoolList bl, const int &accs)
: TValueFilter(accs),
  acceptableValues(bl)
{}


int TValueFilter_discrete::operator()(const TValue &val) const
{ if (val.isSpecial())
    return acceptSpecial;

  return ((val.intV<acceptableValues->size()) && acceptableValues->operator[](val.intV)) ? 1 : 0;
}


TFilter_Values::TFilter_Values(bool anAnd, bool aneg, PDomain dom)
: TFilter(aneg, dom),
  values(dom ? PValueFilterList(mlnew TValueFilterList(dom->variables->size(), PValueFilter())) : PValueFilterList()),
  doAnd(anAnd)
{}


TFilter_Values::TFilter_Values(PValueFilterList v, bool anAnd, bool aneg, PDomain dom)
: TFilter(aneg, dom),
  values(v),
  doAnd(anAnd)
{}


bool TFilter_Values::operator()(const TExample &exam)
{ checkProperty(domain);
  checkProperty(values);
  if (values->size() > domain->variables->size())
    raiseError("invalid size of 'values'");

  TExample example(domain, exam);
  TExample::const_iterator ei(example.begin());

  for(vector<PValueFilter>::const_iterator fi(values->begin()), fe(values->end()); fi!=fe; fi++, ei++)
    if (*fi) {
      const int r = (*fi)->call(*ei);
      if ((r==0) && doAnd)
        return negate;
      if ((r==1) && !doAnd)
        return !negate; // if this one is OK, we should return true if negate=false and vice versa
    }

  // If we've come this far; if doAnd==true, all were OK; doAnd==false, none were OK
  return doAnd!=negate;
}



/// Constructor; sets the example
TFilter_sameExample::TFilter_sameExample(PExample anexample, bool aneg)
  : TFilter(aneg, anexample->domain), example(anexample)
  {}


/// Chooses an examples (not) equal to the 'example'
bool TFilter_sameExample::operator()(const TExample &other)
{ return (example->compare(TExample(domain, other))==0)!=negate; }



/// Constructor; sets the example
TFilter_compatibleExample::TFilter_compatibleExample(PExample anexample, bool aneg)
: TFilter(aneg, anexample->domain),
  example(anexample)
{}


/// Chooses an examples (not) compatible with the 'example'
bool TFilter_compatibleExample::operator()(const TExample &other)
{ return example->compatible(TExample(domain, other))!=negate; }




TFilter_index::TFilter_index()
: value(-1),
  position((long *)NULL)
{}


/// Constructor; sets the vector of indices and value
TFilter_index::TFilter_index(TFoldIndices &anind, int aval, bool aneg, PDomain dom)
: TFilter(aneg, dom),
  indices(anind),
  value(aval)
{ reset(); }


/// Returns (*position==value) and increases position.
bool TFilter_index::operator()(const TExample &)
{ if (!indices->size())
    return negate;

  bool res=(*(position++)==value)!=negate;
  if (position==indices->end())
    reset();
  return res; 
};


/// Resets position to the beginning of the filter
void TFilter_index::reset()
{ position=indices->begin(); }

