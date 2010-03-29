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

#include <set>
#include <stack>
#include <map>
#include <algorithm>
#include <queue>
#include <list>
#include <float.h>
#include <locale>

#ifdef DARWIN
#include <strings.h>
#endif

#include "stladdon.hpp"
#include "errors.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "classify.hpp"
#include "domain.hpp"
#include "random.hpp"
#include "orvector.hpp"
#include "stringvars.hpp"


#include "vars.ppp"


DEFINE_TOrangeVector_classDescription(PVariable, "TVarList", true, ORANGE_API)
DEFINE_TOrangeVector_classDescription(PVarList, "TVarListList", true, ORANGE_API)

list<TVariable *> TVariable::allVariables;


const char *sortedDaNe[] = {"da", "ne", 0 };
const char *resortedDaNe[] = {"ne", "da", 0};

const char **specialSortCases[] = { sortedDaNe, 0};
const char **specialCasesResorted[] = { resortedDaNe, 0};

const char *putAtBeginning[] = {"no", "none", "absent", "normal", 0};

TVariable *TVariable::getExisting(const string &name, const int &varType, TStringList *fixedOrderValues, set<string> *values,
                                  const int failOn, int *status)
{
  if ((fixedOrderValues && fixedOrderValues->size() ) && (varType != TValue::INTVAR))
    ::raiseErrorWho("Variable", "cannot specify the value list for non-discrete attributes");
    
  if (failOn == TVariable::OK) {
    if (status)
      *status = TVariable::OK;
    return NULL;
  }
  
  vector<pair<TVariable *, int> > candidates;
  TStringList::const_iterator vvi, vve;
  
  ITERATE(list<TVariable *>, vi, TVariable::allVariables) {
    if (((*vi)->varType == varType) && ((*vi)->name == name)) {
      int tempStat = TVariable::OK;

      // non-discrete attributes are always ok,
      // discrete ones need further checking if they have any defined values
      TEnumVariable *evar = dynamic_cast<TEnumVariable *>(*vi);
      if (evar && evar->values->size()) {
      
        if (fixedOrderValues && !evar->checkValuesOrder(*fixedOrderValues))
          tempStat = TVariable::Incompatible;
          
        if ((tempStat == TVariable::OK) 
            && (values && values->size() || fixedOrderValues && fixedOrderValues->size())) {
          for(vvi = evar->values->begin(), vve = evar->values->end();
              (vvi != vve)
               && (!values || (values->find(*vvi) == values->end()))
               && (!fixedOrderValues || (find(fixedOrderValues->begin(), fixedOrderValues->end(), *vvi) == fixedOrderValues->end()));
              vvi++);
          if (vvi == vve)
            tempStat = TVariable::NoRecognizedValues;
         }
         
         if ((tempStat == TVariable::OK) && fixedOrderValues) {
           for(vvi = fixedOrderValues->begin(), vve = fixedOrderValues->end();
               (vvi != vve) && evar->hasValue(*vvi);
               vvi++);
           if (vvi != vve)
             tempStat = TVariable::MissingValues;
         }
          
         if ((tempStat == TVariable::OK) && values) {
           set<string>::const_iterator vsi(values->begin()), vse(values->end());
           for(; (vsi != vse) && evar->hasValue(*vsi); vsi++);
           if (vsi != vse)
             tempStat = TVariable::MissingValues;
         }
       }
    
      candidates.push_back(make_pair(*vi, tempStat));
      if (tempStat == TVariable::OK)
        break;
    }
  }

  TVariable *var = NULL;

  int intStatus;
  if (!status)
    status = &intStatus;
  *status = TVariable::NotFound;
  
  const int actFailOn = failOn > TVariable::Incompatible ? TVariable::Incompatible : failOn;
  for(vector<pair<TVariable *, int> >::const_iterator ci(candidates.begin()), ce(candidates.end());
      ci != ce; ci++)
    if (ci->second < *status) {
      *status = ci->second;
      if (*status < actFailOn)
        var = ci->first;
    }

  var = mlnew TEnumVariable(name);
  TEnumVariable *evar = dynamic_cast<TEnumVariable *>(var);
  if (evar) { 
    if (fixedOrderValues)
      const_PITERATE(TStringList, si, fixedOrderValues)
        evar->addValue(*si);
  
    if (values) {
      vector<string> sorted;
      TEnumVariable::presortValues(*values, sorted);
      const_ITERATE(vector<string>, ssi, sorted)
        evar->addValue(*ssi);
    }
  }

  return var;
}


TVariable *TVariable::make(const string &name, const int &varType, TStringList *fixedOrderValues, set<string> *values,
                           const int createNewOn, int *status)
{
  int intStatus;
  if (!status)
    status = &intStatus;

  TVariable *var;
  if (createNewOn == TVariable::OK) {
    var = NULL;
    *status = TVariable::OK;
  }
  else {
    var = getExisting(name, varType, fixedOrderValues, values, createNewOn, status);
  }
    
  if (!var) {
      switch (varType) {
        case TValue::INTVAR: {
          var = mlnew TEnumVariable(name);
          TEnumVariable *evar = dynamic_cast<TEnumVariable *>(var);
          if (evar) { 
            if (fixedOrderValues)
              const_PITERATE(TStringList, si, fixedOrderValues)
                evar->addValue(*si);
          
            if (values) {
              vector<string> sorted;
              TEnumVariable::presortValues(*values, sorted);
              const_ITERATE(vector<string>, ssi, sorted)
                evar->addValue(*ssi);
            }
          }
          break;
        }

        case TValue::FLOATVAR:
          var = mlnew TFloatVariable(name);
          break;

        case STRINGVAR:
          var = mlnew TStringVariable(name);
          break;
      }
  }
   
  return var;
}


bool TVariable::isEquivalentTo(const TVariable &old) const {
  return    (varType == old.varType) && (ordered == old.ordered) && (distributed == old.distributed)
         && (!sourceVariable || !old.sourceVariable || (sourceVariable == old.sourceVariable))
         && (!getValueFrom || !old.getValueFrom || (getValueFrom == old.getValueFrom));
}


TVariable::TVariable(const int &avarType, const bool &ord)
: varType(avarType),
  ordered(ord),
  distributed(false),
  getValueFromLocked(false),
  DC_value(varType, valueDC),
  DK_value(varType, valueDK),
  defaultMetaId(0)
{}


TVariable::TVariable(const string &aname, const int &avarType, const bool &ord)
: varType(avarType),
  ordered(ord),
  distributed(false),
  getValueFromLocked(false),
  DC_value(varType, valueDC),
  DK_value(varType, valueDK),
  defaultMetaId(0)
{ name = aname; };


const TValue &TVariable::DC() const
{ return DC_value; }


const TValue &TVariable::DK() const
{ return DK_value; }


TValue  TVariable::specialValue(int spec) const
{ return TValue(varType, spec); }


/*  Converts a human-readable string, representing the value (as read from the file, for example) to TValue.
    TVariable::str2val_add interprets ? as DK and ~ as DC; otherwise it sets val.varType to varType, other fields
    are left intact.*/
bool TVariable::str2special(const string &valname, TValue &valu) const
{ if ((valname=="?") || !valname.length()){
    valu = TValue(DK());
    return true;
  }
  else if (valname=="~") {
    valu = TValue(DC());
    return true;
  }
  return false;
}


bool TVariable::special2str(const TValue &val, string &str) const
{ switch (val.valueType) {
    case 0: return false;
    case valueDC : str="~"; break;
    case valueDK : str="?"; break;
    default: str = ".";
  }
  return true;
}



bool TVariable::str2val_try(const string &valname, TValue &valu)
{ try {
    str2val(valname, valu);
    return true;
  } catch (exception) {
    return false;
  }
}


void TVariable::str2val_add(const string &valname, TValue &valu)
{ str2val(valname, valu); }


void TVariable::filestr2val(const string &valname, TValue &valu, TExample &)
{ str2val_add(valname, valu); }


void TVariable::val2filestr(const TValue &val, string &str, const TExample &) const
{ val2str(val, str); }


// Calls classifier, specified in getValueFrom, if one is available
TValue TVariable::computeValue(const TExample &ex)
{ if (getValueFrom && !getValueFromLocked)
    try {
      if (distributed) {
        getValueFromLocked = true;
        const PSomeValue val = PSomeValue(getValueFrom->classDistribution(ex));
        getValueFromLocked = false;
        return TValue(PSomeValue(val));
      }
      else {
        getValueFromLocked = true;
        const TValue val = getValueFrom->operator()(ex);
        getValueFromLocked = false;
        return val;
      }
    }
    catch (...) {
      getValueFromLocked = false;
      throw;
    }
  else
    return DK();
}


bool TVariable::firstValue(TValue &val) const
{ 
  raiseError("attribute '%s' does not support 'firstValue' method", name.c_str());
  return false;
}

bool TVariable::nextValue(TValue &val) const
{ 
  raiseError("attribute '%s' does not support 'nextValue' method", name.c_str());
  return false;
}

TValue TVariable::randomValue(const int &rand)
{ 
  raiseError("attribute '%s' does not support 'randomValue' method", name.c_str());
  return TValue();
}


// Sets autoValues to true, maxLen to 2 and varType to TValue::INVAR
TEnumVariable::TEnumVariable()
: TVariable(TValue::INTVAR, false),
  values(mlnew TStringList()),
  baseValue(-1)
{}



TEnumVariable::TEnumVariable(const string &aname)
: TVariable(aname, TValue::INTVAR, false),
  values(mlnew TStringList()),
  baseValue(-1)
{}


TEnumVariable::TEnumVariable(const string &aname, PStringList val)
: TVariable(aname, TValue::INTVAR, false),
  values(val),
  baseValue(-1)
{}


TEnumVariable::TEnumVariable(const TEnumVariable &var)
: TVariable(var), 
  values(mlnew TStringList(var.values.getReference())),
  baseValue(var.baseValue)
{}


bool TEnumVariable::isEquivalentTo(const TVariable &old) const
{
  TEnumVariable const *eold = dynamic_cast<TEnumVariable const *>(&old);
  
  if (!eold || !TVariable::isEquivalentTo(old) ||
         ((baseValue != -1) && (eold->baseValue != -1) && (baseValue != eold->baseValue)))
     return false;
     
  TStringList::const_iterator vi1(values->begin()), ve1(values->end());
  TStringList::const_iterator vi2(eold->values->begin()), ve2(eold->values->end());
  for(; (vi1 != ve1) && (vi2 != ve2) && (*vi1 == *vi2); vi1++, vi2++);
  return (vi1 == ve1) || (vi2 == ve2);
}


int  TEnumVariable::noOfValues() const
{ return values->size(); };


bool TEnumVariable::firstValue(TValue &val) const
{ if (values->size()) {
    val = TValue(0);
    return true; 
  }
  else {
    val = TValue(DK());
    return false;
  }
}



bool TEnumVariable::nextValue(TValue &val) const
{ return (++val.intV<int(values->size())); }



TValue TEnumVariable::randomValue(const int &rand)
{
  if (!randomGenerator)
    randomGenerator = mlnew TRandomGenerator;

  if (!values->size())
    raiseErrorWho("randomValue", "no values");

  return TValue(int(rand<=0 ? randomGenerator->randint(values->size()) : rand%values->size()));
}


void TEnumVariable::addValue(const string &val)
{ 
  if (values->size() > 50) {
    if (valuesTree.empty())
      createValuesTree();

    map<string, int>::iterator lb = valuesTree.lower_bound(val);
    if ((lb != valuesTree.end()) && (lb->first != val)) {
      // watch the order!
      valuesTree.insert(lb, make_pair(val, values->size()));
      values->push_back(val);
    }
  }

  else {
    if (!exists(values->begin(), values->end(), val))
      values->push_back(val);

    if ((values->size() == 5) && ((values->front() == "f") || (values->front() == "float"))) {
      TStringList::const_iterator vi(values->begin()), ve(values->end());
      char *eptr;
      char numtest[32];
      while(++vi != ve) { // skip the first (f/float)
        if ((*vi).length() > 31)
          break;

        strcpy(numtest, (*vi).c_str());
        for(eptr = numtest; *eptr; eptr++)
          if (*eptr == ',')
            *eptr = '.';

        strtod(numtest, &eptr);
        while (*eptr==32)
          eptr++;

        if (*eptr)
          break;
      }

      if (vi==ve)
        raiseWarning("is '%s' a continuous attribute unintentionally defined by '%s'?", name.c_str(), values->front().c_str());
    }
  }
}


bool TEnumVariable::hasValue(const string &s)
{
  if (!valuesTree.empty())
    return valuesTree.lower_bound(s) != valuesTree.end();
    
  PITERATE(TStringList, vli, values)
    if (*vli == s)
      return true;
      
  return false;
}


/*  Converts a value from string representation to TValue by searching for it in value list.
    If value is not found, it is added to the list if 'autoValues'==true, else exception is raised. */
void TEnumVariable::str2val_add(const string &valname, TValue &valu)
{
  const int noValues = values->size();

  if (noValues > 50) {
    if (valuesTree.empty())
      createValuesTree();

    map<string, int>::iterator lb = valuesTree.lower_bound(valname);
    if ((lb != valuesTree.end()) && (lb->first == valname))
      valu = TValue(lb->second);
    else if (!str2special(valname, valu)) {
      valuesTree.insert(lb, make_pair(valname, noValues));
      values->push_back(valname);
      valu = TValue(noValues);
    }
  }

  else {
    TStringList::iterator vi = find(values->begin(), values->end(), valname);
    if (vi!=values->end())
      valu = TValue(int(vi - values->begin())); 
    else if (!str2special(valname, valu)) {
      addValue(valname);
      valu = TValue(noValues);
    }
  }
}


void TEnumVariable::str2val(const string &valname, TValue &valu)
{
  if (values->size() > 50) {
    if (valuesTree.empty())
      createValuesTree();

    map<string, int>::const_iterator vi = valuesTree.find(valname);
    if (vi != valuesTree.end())
      valu = TValue(vi->second);
    else if (!str2special(valname, valu))
      raiseError("attribute '%s' does not have value '%s'", name.c_str(), valname.c_str());
  }

  else {
    TStringList::const_iterator vi = find(values->begin(), values->end(), valname);
    if (vi!=values->end())
      valu = TValue(int(vi - values->begin())); 
    else if (!str2special(valname, valu))
      raiseError("attribute '%s' does not have value '%s'", name.c_str(), valname.c_str());
  }
}



bool TEnumVariable::str2val_try(const string &valname, TValue &valu)
{
  if (values->size() > 50) {
    if (valuesTree.empty())
      createValuesTree();

    map<string, int>::const_iterator vi = valuesTree.find(valname);
    if (vi != valuesTree.end()) {
      valu = TValue(vi->second);
      return true;
    }
    return str2special(valname, valu);
  }
    
  else {
    TStringList::const_iterator vi = find(values->begin(), values->end(), valname);
    if (vi!=values->end()) {
      valu = TValue(int(vi - values->begin())); 
      return true;
    }
    return str2special(valname, valu);
  }
}



// Converts TValue into a string representation of value. If invalid, string 'ERR' is returned.
void TEnumVariable::val2str(const TValue &val, string &str) const
{ if (val.isSpecial()) {
    special2str(val, str);
    return;
  }

  if (val.svalV) {
    const TDiscDistribution *dval = dynamic_cast<const TDiscDistribution *>(val.svalV.getUnwrappedPtr());
    if (!dval) 
      raiseError("invalid value type");

    str = "(";
    char buf[12];
    const_PITERATE(TDiscDistribution, di, dval) {
      if (di != dval->begin())
        str+=", ";
      sprintf(buf, "%1.3f", *di);
      str += buf;
    }
    str += ")";
  }

  str = val.intV<int(values->size()) ? values->operator[](val.intV) : "#RNGE";
}


void TEnumVariable::createValuesTree()
{
  int i = 0;
  const_PITERATE(TStringList, vi, values)
    valuesTree[*vi] = i++;
}


bool TEnumVariable::checkValuesOrder(const TStringList &refValues)
{
  for(TStringList::const_iterator ni(refValues.begin()), ne(refValues.end()), ei(values->begin()), ee(values->end()); 
      (ei != ee) && (ni != ne); ei++, ni++)
    if (*ei != *ni)
      return false;
  return true;
}


void TEnumVariable::presortValues(const set<string> &unsorted, vector<string> &sorted)
{
  sorted.clear();
  sorted.insert(sorted.begin(), unsorted.begin(), unsorted.end());
  
  vector<string>::iterator si, se(sorted.end());
  const char ***ssi, **ssii, ***rssi;
  for(ssi = specialSortCases, rssi = specialCasesResorted; *ssi; ssi++, rssi++) {
    for(si = sorted.begin(), ssii = *ssi; *ssii && (si != se) && !stricmp(*ssii, si->c_str()); *ssii++);
    if (!*ssii && (si==se)) {
      sorted.clear();
      sorted.insert(sorted.begin(), *rssi, *rssi + (ssii - *ssi));
      return;
    }
  }
  
  se = sorted.end();
  for(ssii = putAtBeginning; *ssii; ssii++) {
    for(si = sorted.begin(); (si != se) && stricmp(*ssii, si->c_str()); si++);
    if (si != se) {
      const string toMove = *si;
      sorted.erase(si);
      sorted.insert(sorted.begin(), toMove);
      break;
    }
  }
}


TFloatVariable::TFloatVariable()
: TVariable(TValue::FLOATVAR, true),
  startValue(-1.0),
  endValue(0.0),
  stepValue(-1.0),
  numberOfDecimals(3),
  scientificFormat(false),
  adjustDecimals(2)
{}


TFloatVariable::TFloatVariable(const string &aname)
: TVariable(aname, TValue::FLOATVAR, true),
  startValue(-1.0),
  endValue(0.0),
  stepValue(-1.0),
  numberOfDecimals(3),
  scientificFormat(false),
  adjustDecimals(2)
{}

bool TFloatVariable::isEquivalentTo(const TVariable &old) const
{
  TFloatVariable const *eold = dynamic_cast<TFloatVariable const *>(&old);
  return eold && TVariable::isEquivalentTo(old) && 
         (startValue == eold->startValue) && (endValue == eold->endValue) && (stepValue == eold->stepValue);
}


bool TFloatVariable::firstValue(TValue &val) const
{ if ((stepValue<=0) || (startValue<endValue))
    return false;
  val = TValue(startValue);
  return true;
}


bool TFloatVariable::nextValue(TValue &val) const
{ if (stepValue<=0)
    return false;
  return ((val.floatV+=stepValue)<=endValue);
}


TValue TFloatVariable::randomValue(const int &rand)
{ if ((stepValue<=0) || (startValue>=endValue))
    raiseError("randomValue: interval not given");

  if (!randomGenerator)
    randomGenerator = mlnew TRandomGenerator();

  if (rand<0)
    return TValue(randomGenerator->randfloat(startValue, endValue));
  else
    return TValue(float(double(rand)/double(4294967295.0)*(endValue-startValue)+startValue));
}


int  TFloatVariable::noOfValues() const
{ return stepValue>0 ? int((endValue-startValue)/stepValue) : -1; }


inline int getNumberOfDecimals(const char *vals, bool &hasE)
{
  const char *valsi;
  for(valsi = vals; *valsi && ((*valsi<'0') || (*valsi>'9')) && (*valsi != '.'); valsi++);
  if (!*valsi)
    return -1;

  if ((*valsi=='e') || (*valsi=='E')) {
    hasE = true;
    return 0;
  }

  for(; *valsi && (*valsi!='.'); valsi++);
  if (!*valsi)
    return 0;

  int decimals = 0;
  for(valsi++; *valsi && (*valsi>='0') && (*valsi<='9'); valsi++, decimals++);

  hasE = hasE || (*valsi == 'e') || (*valsi == 'E');
  return decimals;
}


int TFloatVariable::str2val_low(const string &valname, TValue &valu)
{
  if (str2special(valname, valu))
    return 1;

  const char *vals;
  char *tmp = NULL;

  const char radix = *localeconv()->decimal_point;
  const char notGood = radix=='.' ? ',' : '.';
  int cp = valname.find(notGood);
  if (cp!=string::npos) {
    vals = tmp = strcpy(new char[valname.size()+1], valname.c_str());
    tmp[cp] = radix;
  }
  else
    vals = valname.c_str();

  float f;
  int ssr = sscanf(vals, "%f", &f);

  int res;

  if (!ssr || (ssr==(char)EOF)) {
    res = -1;
  }
  else {
    valu = TValue(f);

    if (((startValue<=endValue) && (stepValue>0) && ((f<startValue) || (f>endValue)))) {
      res = -2;
    }

    else {
      res = 1;

      valu = TValue(f);

      int decimals;
      switch (adjustDecimals) {
        case 2:
          numberOfDecimals = getNumberOfDecimals(vals, scientificFormat);
          adjustDecimals = 1;
          break;
        case 1:
          decimals = getNumberOfDecimals(vals, scientificFormat);
          if (decimals > numberOfDecimals)
            numberOfDecimals = decimals;
      }
    }
  }

  if (tmp)
    delete tmp;

  return res;
}


void TFloatVariable::str2val(const string &valname, TValue &valu)
{ 
  switch (str2val_low(valname, valu)) {
    case -1: raiseError("'%s' is not a legal value for continuous attribute '%s'", valname.c_str(), name.c_str());
    case -2: raiseError("value %5.3f out of range %5.3f-%5.3f", valu.floatV, startValue, endValue);
  }
}

bool TFloatVariable::str2val_try(const string &valname, TValue &valu)
{ 
  return str2val_low(valname, valu) == 1;
}


void TFloatVariable::val2str(const TValue &valu, string &vname) const
{ if (valu.isSpecial())
    special2str(valu, vname);
  else {
    char buf[64];
    const float f = fabs(valu.floatV);
    if (scientificFormat)
      sprintf(buf, "%g", valu.floatV);
    else
      sprintf(buf, "%.*f", numberOfDecimals, valu.floatV);
    vname = buf;
  }
}
