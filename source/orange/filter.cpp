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
: negate(anegate),
  domain(dom)
{}

void TFilter::reset()
{}

// Sets the maxrand field to RAND_MAX*ap
TFilter_random::TFilter_random(const float ap, bool aneg, PDomain dom)
: TFilter(aneg, dom),
  prob(ap) 
{};

// Chooses an example (returns true) if rand()<maxrand; example is ignored
bool TFilter_random::operator()(const TExample &)
{
  return (LOCAL_OR_GLOBAL_RANDOM.randfloat()<prob)!=negate;
}



TFilter_hasSpecial::TFilter_hasSpecial(bool aneg, PDomain dom)
  : TFilter(aneg, dom)
  {}


// Chooses an example if it has (no) special values.
bool TFilter_hasSpecial::operator()(const TExample &exam)
{ int i=0, Nv;
  if (domain) {
    TExample example(domain, exam);
    for(Nv = domain->variables->size(); (i<Nv) && !example[i].isSpecial(); i++);
  }
  else
    for(Nv = exam.domain->variables->size(); (i<Nv) && !exam[i].isSpecial(); i++);

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
{ signed char equ = (domain ? TExample(domain, example)[position] : example[position]) == value;
  return equ==-1 ? negate : ((equ!=0) != negate);
}


TValueFilter::TValueFilter(const int &pos, const int &accs)
: position(pos),
  acceptSpecial(accs)
{}


TValueFilter_continuous::TValueFilter_continuous(const int &pos, const float &amin, const float &amax, const bool &outs, const int &accs)
: TValueFilter(pos, accs),
  min(amin),
  max(amax),
  outside(outs)
{}


int TValueFilter_continuous::operator()(const TExample &example) const
{ const TValue &val = example[position];
  if (val.isSpecial())
    return acceptSpecial;

  return ((val.floatV>=min) && (val.floatV<=max)) != outside ? 1 : 0;
}


TValueFilter_discrete::TValueFilter_discrete(const int &pos, PValueList bl, const int &accs)
: TValueFilter(pos, accs),
  values(bl)
{}


TValueFilter_discrete::TValueFilter_discrete(const int &pos, PVariable var, const int &accs)
: TValueFilter(pos, accs),
  values(mlnew TValueList(var))
{}


int TValueFilter_discrete::operator()(const TExample &example) const
{ const TValue &val = example[position];
  if (val.isSpecial())
    return acceptSpecial;

  const_PITERATE(TValueList, vi, values)
    if ((*vi).intV == val.intV)
      return 1;

  return 0;
}


TFilter_values::TFilter_values(bool anAnd, bool aneg, PDomain dom)
: TFilter(aneg, dom),
  conditions(mlnew TValueFilterList()),
  conjunction(anAnd)
{}


TFilter_values::TFilter_values(PValueFilterList v, bool anAnd, bool aneg, PDomain dom)
: TFilter(aneg, dom),
  conditions(v),
  conjunction(anAnd)
{}


bool TFilter_values::operator()(const TExample &exam)
{ checkProperty(domain);
  checkProperty(conditions);

  TExample *example;
  PExample wex;
  if (domain) {
    example = mlnew TExample(domain, exam);
    wex = example;
  }
  else
    example = const_cast<TExample *>(&exam);

  PITERATE(TValueFilterList, fi, conditions) {
    const int r = (*fi)->call(*example);
    if ((r==0) && conjunction)
      return negate;
    if ((r==1) && !conjunction)
      return !negate; // if this one is OK, we should return true if negate=false and vice versa
  }

  // If we've come this far; if conjunction==true, all were OK; conjunction==false, none were OK
  return conjunction!=negate;
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
