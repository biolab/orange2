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
  { return (randfloat()<prob)!=negate; }



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


TValueRange::TValueRange(PDiscDistribution dist, signed char spec)
: probs(dist),
  special(spec)
{}


TValueRange::TValueRange(float ai, float aa, signed char spec)
: min(ai),
  max(aa),
  special(spec)
{}


TFilter_sameValues::TFilter_sameValues(bool anAnd, bool aneg, PDomain dom)
: TFilter(aneg, dom),
  values(dom ? dom->variables->size() : 0, PValueRange()),
  doAnd(anAnd)
{}


TFilter_sameValues::TFilter_sameValues(const vector<PValueRange> &ap, bool anAnd, bool aneg, PDomain dom)
: TFilter(aneg, dom),
  values(ap),
  doAnd(anAnd)
{}


bool TFilter_sameValues::operator()(const TExample &exam)
{ TExample example(domain, exam);
  TExample::const_iterator ei(example.begin());
  for(iterator fi(begin()), fe(end());
      fi!=fe;
      ei++, fi++)
    if (*fi) {
      if ((*ei).isSpecial())
        switch ((*fi)->special) {
          case  0: if (doAnd)
                     return negate;
                   break;
          case  1: if (!doAnd)
                     return !negate;
                   break;
          default: continue;
        }
      else 
  
        if ((*ei).varType==TValue::INTVAR) {
          if ((*ei).intV<int((*fi)->probs->size())) {
            float p = (*fi)->probs->at(int(*ei));
            if ((p>=1.0) || (randfloat()<p)) {
              if (!doAnd) 
                return !negate; // if this one is OK and one already suffices, return OK
            } 
            else {
              if (doAnd)        // if this one is not OK, but everyone must be, return not Ok
                return negate; 
            } 
          }
          else 
            if (doAnd)          // this one is out of scope, has prob of zero and fails
              return negate;
        }
        else if ((*ei).varType==TValue::FLOATVAR) {
          // if min<max is true and value is in interval OR not true and not in interval ...
          if (   ( ((*fi)->min<=(*fi)->max) && ((*fi)->min<=(*ei).floatV) && ((*ei).floatV<=(*fi)->max) ) 
              || ( ((*fi)->min >(*fi)->max) && (((*ei).floatV>(*fi)->min) || ((*ei).floatV< (*fi)->max)) ) ) {
            if (!doAnd)
              return !negate;
          }
          else {
            if (doAnd)
              return negate; 
          }
        }
    }

  // If we've come this far; if doAnd==true, all were OK; doAnd==false, none were OK
  return doAnd!=negate;
}


int TFilter_sameValues::traverse(visitproc visit, void *arg)
{ TRAVERSE(TFilter::traverse);
  this_ITERATE(vri)
    PVISIT(*vri);
  return 0;
}

int TFilter_sameValues::dropReferences()
{ DROPREFERENCES(TFilter::dropReferences);
  clear();
  return 0;
}


TValueFilter::TValueFilter(bool anAnd, bool aneg, PDomain dom)
 : TFilter_sameValues(anAnd, aneg, dom)
 {}


TValueFilter::TValueFilter(const TMultiStringParameters &pars, string keyword, PDomain dom, bool anAnd, bool aneg)
 : TFilter_sameValues(anAnd, aneg, dom)
{ decode(pars, keyword); }


TValueFilter::TValueFilter(istream &istr, PDomain dom, bool anAnd, bool aneg)
 : TFilter_sameValues(anAnd, aneg, dom)
{ decode(istr); }


void TValueFilter::decode(const TMultiStringParameters &pars, string keyword)
{ vector<string> drops;
  const_ITERATE(TMultiStringParameters, pi, pars)
    if ((*pi).first==keyword) drops.push_back((*pi).second);
    else
      if ((*pi).first==keyword+'f') {
        ifstream outstr ((*pi).second.c_str());
        decode(outstr);
      }
  decode(drops);
}


#define MAX_LINE_LENGTH 1024

void TValueFilter::decode(istream &istr)
{ vector<string> drops;
  while (!istr.eof()) {
    char line[MAX_LINE_LENGTH];
    istr.getline(line, MAX_LINE_LENGTH);
    if (istr.gcount()==MAX_LINE_LENGTH-1)
      raiseError("parameter line too long");
    if (istr.gcount())
      drops.push_back(line);
  }
  decode(drops);
}
#undef MAX_LINE_LENGTH


void TValueFilter::decode(const vector<string> &drops)
{ const_ITERATE(vector<string>, pi, drops) {
    // decode name=value string
    string::const_iterator si=(*pi).begin(), eos=(*pi).end();
    while ((si!=eos) && (*si!='=')) si++;
    string name((*pi).begin(), si);
    while ((si!=eos) && (*(++si)==' '));

    // get the attribute number
    int varnum = domain->getVarNum(name);

    PVariable var=domain->variables->at(varnum);
    PValueRange range = values[varnum];
    if (!range) 
      range = values[varnum] = mlnew TValueRange();

    // if no value is given, unknown will be dropped
    if (si==eos) {
      range->special=false;
      continue;
    }
    
    if (var->varType==TValue::INTVAR)
      while (si!=eos) {
        string::const_iterator bov=si;
        while ((si!=eos) && (*si!=',')) si++;
        string value(bov, si);
        while ((si!=eos) && (*(++si)==' '));

        TValue wv;
        var->str2val(value, wv);
        if (wv.isSpecial())
          range->special=false;
        else 
          range->probs->add(wv);
      }
    else if (var->varType==TValue::FLOATVAR) {
      string::const_iterator bov=(*pi).begin();
      while ((si!=eos) && (*si!='-')) si++;
      range->min=atof(string(bov, si).c_str());
      range->max=atof(string(si, eos).c_str());
    }
    else 
      raiseError("attribute '%s' is of invalid type for this filter", name.c_str());
  }
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

