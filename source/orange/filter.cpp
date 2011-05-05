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


#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "stladdon.hpp"
#include "random.hpp"

#include "vars.hpp"
#include "stringvars.hpp"
#include "domain.hpp"
#include "distvars.hpp"
#include "examplegen.hpp"

#include "filter.ppp"


DEFINE_TOrangeVector_classDescription(PValueFilter, "TValueFilterList", true, ORANGE_API)
DEFINE_TOrangeVector_classDescription(PFilter, "TFilterList", true, ORANGE_API)

// Sets the negate field (default is false)
TFilter::TFilter(bool anegate, PDomain dom) 
: negate(anegate),
  domain(dom)
{}

void TFilter::reset()
{}

PFilter TFilter::deepCopy() const
{
  raiseWarning("Deep copy not implemented.");
  return PFilter();
}

// Sets the maxrand field to RAND_MAX*ap
TFilter_random::TFilter_random(const float ap, bool aneg, PRandomGenerator rgen)
: TFilter(aneg, PDomain()),
  prob(ap),
  randomGenerator(rgen ? rgen : PRandomGenerator(mlnew TRandomGenerator()))
{};

// Chooses an example (returns true) if rand()<maxrand; example is ignored
bool TFilter_random::operator()(const TExample &)
{
  if (!randomGenerator)
    randomGenerator = mlnew TRandomGenerator;

  return (randomGenerator->randfloat()<prob)!=negate;
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


TFilter_isDefined::TFilter_isDefined(bool aneg, PDomain dom)
: TFilter(aneg, dom),
  check(mlnew TAttributedBoolList(dom ? dom->variables : PVarList(), dom ? dom->variables->size(): 0, true))
{}


bool TFilter_isDefined::operator()(const TExample &exam)
{
  TExample *example;
  PExample wex;
  if (domain && (domain != exam.domain)) {
    example = mlnew TExample(domain, exam);
    wex = example;
  }
  else
    example = const_cast<TExample *>(&exam);

  if (!check || !check->size()) {
    const_PITERATE(TExample, ei, example)
      if ((*ei).isSpecial())
        return negate;
    return !negate;
  }

  else {
    TBoolList::const_iterator ci(check->begin()), ce(check->end());
    TExample::const_iterator ei(example->begin()), ee(example->end());
    for(; (ci!=ce) && (ei!=ee); ci++, ei++)
      if (*ci && (*ei).isSpecial())
        return negate;
    return !negate;
  }
}


void TFilter_isDefined::afterSet(const char *name)
{
  if (!strcmp(name, "domain") && domain && (!check || !check->size()) && (domain->variables != check->attributes))
    check = mlnew TAttributedBoolList(domain->variables, domain->variables->size(), true);

  TFilter::afterSet(name);
}


TFilter_hasMeta::TFilter_hasMeta(const int &anid, bool aneg, PDomain dom)
: TFilter(aneg, dom),
  id(anid)
{}


bool TFilter_hasMeta::operator()(const TExample &exam)
{
  return exam.hasMeta(id) != negate;
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
{ 
  if (domain && (domain != example.domain)) {
    // this is slow & inefficient, but it's the only legal way of doing it
    TExample ex(domain, example);
    return ((position != -1 ? ex[position] : example.getClass()) == value) != negate;
  }
  else
    return ((position != -1 ? example[position]  : example.getClass()) == value) != negate;
}


TValueFilter::TValueFilter(const int &pos, const int &accs)
: position(pos),
  acceptSpecial(accs)
{}

PValueFilter TValueFilter::deepCopy() const
{
  raiseWarning("Deep copy not implemented.");
  return PValueFilter();
}


TValueFilter_continuous::TValueFilter_continuous()
: TValueFilter(ILLEGAL_INT, -1),
  min(0.0),
  max(0.0),
  outside(false),
  oper(None)
{}

TValueFilter_continuous::TValueFilter_continuous(const int &pos, const float &amin, const float &amax, const bool &outs, const int &accs)
: TValueFilter(pos, accs),
  min(amin),
  max(amax),
  outside(outs),
  oper(None)
{}


TValueFilter_continuous::TValueFilter_continuous(const int &pos, const int &op, const float &amin, const float &amax, const int &accs)
: TValueFilter(pos, accs),
  min(amin),
  max(amax),
  oper(op)
{}


#define EQUAL(x,y)  (fabs(x-y) <= y*1e-10) ? 1 : 0
#define LESS_EQUAL(x,y) (x-y <= y*1e-10) ? 1 : 0
#define TO_BOOL(x) (x) ? 1 : 0;

int TValueFilter_continuous::operator()(const TExample &example) const
{ const TValue &val = example[position];
  if (val.isSpecial())
    return acceptSpecial;

  switch (oper) {
    case None:         return TO_BOOL(((val.floatV>=min) && (val.floatV<=max)) != outside);

    case Equal:        return EQUAL(val.floatV, min);
    case NotEqual:     return 1 - EQUAL(val.floatV, min);
    case Less:         return TO_BOOL(val.floatV < min);
    case LessEqual:    return LESS_EQUAL(val.floatV, min);
    case Greater:      return TO_BOOL(min < val.floatV);
    case GreaterEqual: return LESS_EQUAL(min, val.floatV);
    case Between:      return (LESS_EQUAL(min, val.floatV)) * (LESS_EQUAL(val.floatV, max));
    case Outside:      return TO_BOOL((val.floatV < min) || (val.floatV > max));

    default:  return -1;
  }
}

PValueFilter TValueFilter_continuous::deepCopy() const
{
  TValueFilter *filter = mlnew TValueFilter_continuous(position,oper,min,max,acceptSpecial);
  PValueFilter wfilter = filter;
  return wfilter;
}

TValueFilter_discrete::TValueFilter_discrete(const int &pos, PValueList bl, const int &accs, bool neg)
: TValueFilter(pos, accs),
  values(bl ? bl : mlnew TValueList()),
  negate(neg)
{}

TValueFilter_discrete::TValueFilter_discrete(const int &pos, PVariable var, const int &accs, bool neg)
: TValueFilter(pos, accs),
  values(mlnew TValueList(var)),
  negate(neg)
{}

int TValueFilter_discrete::operator()(const TExample &example) const
{ const TValue &val = example[position];
  if (val.isSpecial())
    return negate ? 1-acceptSpecial : acceptSpecial;

  const_PITERATE(TValueList, vi, values)
    if ((*vi).intV == val.intV)
      return negate ? 0 : 1;

  return negate ? 1 : 0;
}

PValueFilter TValueFilter_discrete::deepCopy() const
{
  if (values->size())
  {
    TValueList *newValues = mlnew TValueList();
    PValueList wnewValues = newValues;
    const_PITERATE(TValueList, vi, values) 
      wnewValues->push_back(TValue(*vi));
    TValueFilter *filter = mlnew TValueFilter_discrete(position,wnewValues,acceptSpecial, negate);
    PValueFilter wfilter = filter;
    return wfilter;
  }
  TValueFilter *filter = mlnew TValueFilter_discrete(position,PValueList(),acceptSpecial, negate);
  PValueFilter wfilter = filter;
  return wfilter;
}

TValueFilter_string::TValueFilter_string()
: TValueFilter(ILLEGAL_INT, -1),
  min(),
  max(),
  oper(None),
  caseSensitive(true)
{}



TValueFilter_string::TValueFilter_string(const int &pos, const int &op, const string &amin, const string &amax, const int &accs, const bool csens)
: TValueFilter(pos, accs),
  min(amin),
  max(amax),
  oper(op),
  caseSensitive(csens)
{}


char *strToLower(string nm)
{ 
  char *s = strcpy(new char[nm.size()+1], nm.c_str());
  for(char *i = s; *i; i++)
    *i = tolower(*i);
  return s;
}

int TValueFilter_string::operator()(const TExample &example) const
{ 
  const TValue &val = example[position];
  if (val.isSpecial())
    return acceptSpecial;

  char *value = caseSensitive ? const_cast<char *>(val.svalV.AS(TStringValue)->value.c_str())
                              : strToLower(val.svalV.AS(TStringValue)->value.c_str());

  char *ref = caseSensitive ? const_cast<char *>(min.c_str()) : strToLower(min);
  
  switch(oper) {
    case Equal:        return TO_BOOL(!strcmp(value, ref));
    case NotEqual:     return TO_BOOL(strcmp(value, ref));
    case Less:         return TO_BOOL(strcmp(value, ref) < 0);
    case LessEqual:    return TO_BOOL(strcmp(value, ref) <= 0);
    case Greater:      return TO_BOOL(strcmp(value, ref) > 0);
    case GreaterEqual: return TO_BOOL(strcmp(value, ref) >= 0);
    case Between:      return TO_BOOL((strcmp(value, ref) >= 0) && (strcmp(value, max.c_str()) <= 0));
    case Outside:      return TO_BOOL((strcmp(value, ref) < 0) && (strcmp(value, max.c_str()) >= 0));
    case Contains:     return TO_BOOL(string(value).find(ref) != string::npos);
    case NotContains:  return TO_BOOL(string(value).find(ref) == string::npos);
    case BeginsWith:   return TO_BOOL(!strncmp(value, ref, strlen(ref)));

    case EndsWith:
      { const int vsize = strlen(value), rsize = strlen(ref);
        return TO_BOOL((vsize >= rsize) && !strcmp(value + (vsize-rsize), ref));
      }

    default:
      return -1;
  }

  if (!caseSensitive) {
    delete value;
    delete ref;
  }
}


TValueFilter_stringList::TValueFilter_stringList()
: TValueFilter(ILLEGAL_INT, -1),
  values(mlnew TStringList()),
  caseSensitive(true)
{}


TValueFilter_stringList::TValueFilter_stringList(const int &pos, PStringList bl, const int &accs, const int &op, const bool csens)
: TValueFilter(pos, accs),
  values(bl),
  caseSensitive(csens)
{}

int TValueFilter_stringList::operator()(const TExample &example) const
{ 
  const TValue &val = example[position];
  if (val.isSpecial())
    return acceptSpecial;

  char *value = caseSensitive ? const_cast<char *>(val.svalV.AS(TStringValue)->value.c_str())
                              : strToLower(val.svalV.AS(TStringValue)->value.c_str());
  char *ref = caseSensitive ? NULL : new char[1024];

  TStringList::const_iterator vi(values->begin()), ve(values->end());
  for(; vi!=ve; vi++)
    if (caseSensitive) {
      if (!strcmp((*vi).c_str(), value))
        break;
    }
    else {
      if ((*vi).size() >= 1024)
        raiseError("reference string too long (1023 characters is the limit)");
      strcpy(ref, (*vi).c_str());
      for(char *i = ref; *i; i++)
        *i = tolower(*i);
      if (!strcmp(ref, value))
        break;
    }

  if (!caseSensitive) {
    delete ref;
    delete value;
  }

  return vi==ve ? 0 : 1;
}


#undef DIFFERENT
#undef LESS_EQUAL
#undef TO_BOOL

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


TValueFilterList::iterator TFilter_values::findCondition(PVariable var, const int &varType, int &position)
{
  if (varType && (var->varType != varType))
    raiseError("invalid variable type");

  checkProperty(domain);

  position = domain->getVarNum(var);
  TValueFilterList::iterator condi(conditions->begin()), conde(conditions->end());
  while((condi!=conde) && ((*condi)->position != position))
    condi++;

  return condi;
}

void TFilter_values::updateCondition(PVariable var, const int &varType, PValueFilter filter)
{
  TValueFilterList::iterator condi = findCondition(var, varType, filter->position);
  if (condi==conditions->end())
    conditions->push_back(filter);
  else
    *condi = filter;
}


void TFilter_values::addCondition(PVariable var, const TValue &val, bool negate)
{
  int position;
  TValueFilterList::iterator condi = findCondition(var, TValue::INTVAR, position);

  TValueFilter_discrete *valueFilter;

  if (condi==conditions->end()) {
    valueFilter = mlnew TValueFilter_discrete(position); // it gets wrapped in the next line
    conditions->push_back(valueFilter);
  }
  else {
    valueFilter = (*condi).AS(TValueFilter_discrete);
    if (!valueFilter)
      raiseError("addCondition(Value) con only be used for setting ValueFilter_discrete");
  }

  if (val.isSpecial())
    valueFilter->acceptSpecial = 1;
  else {
    valueFilter->values->clear();
    valueFilter->values->push_back(val);
  }

  valueFilter->negate = negate;
}


void TFilter_values::addCondition(PVariable var, PValueList vallist, bool negate)
{
  int position;
  TValueFilterList::iterator condi = findCondition(var, TValue::INTVAR, position);

  if (condi==conditions->end())
    conditions->push_back(mlnew TValueFilter_discrete(position, vallist));

  else {
    TValueFilter_discrete *valueFilter = (*condi).AS(TValueFilter_discrete);
    if (!valueFilter)
      raiseError("addCondition(Value) can only be used for setting ValueFilter_discrete");
    else
      valueFilter->values = vallist;
    valueFilter->negate = negate;
  }
}


void TFilter_values::addCondition(PVariable var, const int &oper, const float &min, const float &max)
{
  updateCondition(var, TValue::FLOATVAR, mlnew TValueFilter_continuous(ILLEGAL_INT, oper, min, max));
}


void TFilter_values::addCondition(PVariable var, const int &oper, const string &min, const string &max)
{
  updateCondition(var, STRINGVAR, mlnew TValueFilter_string(ILLEGAL_INT, oper, min, max));
}


void TFilter_values::addCondition(PVariable var, PStringList slist)
{
  updateCondition(var, STRINGVAR, mlnew TValueFilter_stringList(ILLEGAL_INT, slist));
}


void TFilter_values::removeCondition(PVariable var)
{
  int position;
  TValueFilterList::iterator condi = findCondition(var, 0, position);

  if (condi==conditions->end())
    raiseError("there is no condition on value of '%s' in the filter", var->get_name().c_str());

  conditions->erase(condi);
}
  

bool TFilter_values::operator()(const TExample &exam)
{ checkProperty(domain);
  checkProperty(conditions);

  TExample *example;
  PExample wex;
  if (domain && (domain != exam.domain)) {
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

PFilter TFilter_values::deepCopy() const
{
  TValueFilterList *newValueFilters = mlnew TValueFilterList();
  PValueFilterList wnewValueFilters = newValueFilters;

  const_PITERATE(TValueFilterList, vi, conditions) 
    wnewValueFilters->push_back((*vi)->deepCopy());

  TFilter *filter = mlnew TFilter_values(wnewValueFilters,conjunction,negate,domain);
  PFilter wfilter = filter;
  return wfilter;
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




TFilter_conjunction::TFilter_conjunction()
: filters(mlnew TFilterList())
{}


TFilter_conjunction::TFilter_conjunction(PFilterList af)
: filters(af)
{}

bool TFilter_conjunction::operator()(const TExample &ex)
{
  if (filters)
    PITERATE(TFilterList, fi, filters)
      if (!(*fi)->call(ex))
        return negate;

  return !negate;
}


TFilter_disjunction::TFilter_disjunction()
: filters(mlnew TFilterList())
{}


TFilter_disjunction::TFilter_disjunction(PFilterList af)
: filters(af)
{}


bool TFilter_disjunction::operator()(const TExample &ex)
{
  if (filters)
    PITERATE(TFilterList, fi, filters)
      if ((*fi)->call(ex))
        return !negate;

  return negate;
}

