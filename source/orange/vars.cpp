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


#include <set>
#include <stack>
#include <map>
#include <algorithm>
#include <queue>
#include <float.h>

#include "stladdon.hpp"
#include "errors.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "classify.hpp"
#include "domain.hpp"
#include "random.hpp"
#include "orvector.hpp"


#include "vars.ppp"


DEFINE_TOrangeVector_classDescription(PVariable, "TVarList")
DEFINE_TOrangeVector_classDescription(PVarList, "TVarListList")


TPropertyDescription TAttributedFloatList_properties[] = {
  {"attributes", "The list of attributes that corresponds to elements of the list", &typeid(POrange), &TVarList::st_classDescription, offsetof(TAttributedFloatList, attributes), false, false},
  {NULL}
};

size_t const TAttributedFloatList_components[] = { 0};
TClassDescription TAttributedFloatList::st_classDescription = { "TAttributedFloatList", &typeid(TAttributedFloatList), &TOrange::st_classDescription, TAttributedFloatList_properties, TAttributedFloatList_components };
TClassDescription const *TAttributedFloatList::classDescription() const { return &TAttributedFloatList::st_classDescription; }
TOrange *TAttributedFloatList::clone() const { return mlnew TAttributedFloatList(*this); }



TPropertyDescription TValueList_properties[] = {
  {"variable", "The attribute to which the list applies", &typeid(POrange), &TVariable::st_classDescription, offsetof(TValueList, variable), false, false},
  {NULL}
};

size_t const TValueList_components[] = { 0};
TClassDescription TValueList::st_classDescription = { "TValueList", &typeid(TValueList), &TOrange::st_classDescription, TValueList_properties, TValueList_components };
TClassDescription const *TValueList::classDescription() const { return &TValueList::st_classDescription; }
TOrange *TValueList::clone() const { return mlnew TValueList(*this); }


TVariable::TVariable(const int &avarType, const bool &ord)
: varType(avarType),
  ordered(ord),
  distributed(false),
  getValueFromLocked(false)
{}


TVariable::TVariable(const string &aname, const int &avarType, const bool &ord)
: varType(avarType),
  ordered(ord),
  distributed(false),
  getValueFromLocked(false)
{ name = aname; };


TValue  TVariable::DC() const
{ return TValue(varType, valueDC); }


TValue  TVariable::DK() const
{ return TValue(varType, valueDK); }


TValue  TVariable::specialValue(int spec) const
{ return TValue(varType, spec); }


/*  Converts a human-readable string, representing the value (as read from the file, for example) to TValue.
    TVariable::str2val_add interprets ? as DK and ~ as DC; otherwise it sets val.varType to varType, other fields
    are left intact.*/
bool TVariable::str2special(const string &valname, TValue &valu) const
{ if ((valname=="?") || !valname.length()){
    valu = DK();
    return true;
  }
  else if (valname=="~") {
    valu = DC();
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
  return false;
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



int  TEnumVariable::noOfValues() const
{ return values->size(); };


bool TEnumVariable::firstValue(TValue &val) const
{ if (values->size()) {
    val = TValue(0);
    return true; 
  }
  else {
    val=DK();
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
{ if (!exists(values->begin(), values->end(), val))
    values->push_back(val);
}



/*  Converts a value from string representation to TValue by searching for it in value list.
    If value is not found, it is added to the list if 'autoValues'==true, else exception is raised. */
void TEnumVariable::str2val_add(const string &valname, TValue &valu)
{
  TStringList::iterator vi = find(values->begin(), values->end(), valname);
  if (vi!=values->end())
    valu = TValue(int(vi - values->begin())); 
  else if (!str2special(valname, valu)) {
    addValue(valname); 
    valu = TValue(int(values->size()-1));
  }
}


void TEnumVariable::str2val(const string &valname, TValue &valu)
{
  TStringList::const_iterator vi = find(values->begin(), values->end(), valname);
  if (vi!=values->end())
    valu = TValue(int(vi - values->begin())); 
  else if (!str2special(valname, valu))
    raiseError("attribute '%s' does not have value '%s'", name.c_str(), valname.c_str());
}



bool TEnumVariable::str2val_try(const string &valname, TValue &valu)
{
  TStringList::const_iterator vi = find(values->begin(), values->end(), valname);
  if (vi!=values->end()) {
    valu = TValue(int(vi - values->begin())); 
    return true;
  }
  return str2special(valname, valu);
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





TIntVariable::TIntVariable()
: TVariable(TValue::INTVAR, true),
  startValue(0),
  endValue(-1)
{}


TIntVariable::TIntVariable(const string &aname)
: TVariable(aname, TValue::INTVAR, true),
  startValue(0),
  endValue(-1)
{}


#define CHECK_INTERVAL if (startValue>endValue) raiseError("interval not given");

bool TIntVariable::firstValue(TValue &val) const
{ CHECK_INTERVAL
  return ((val = TValue(startValue)).intV<endValue);
}


bool TIntVariable::nextValue(TValue &val) const
{ CHECK_INTERVAL
  return (++val.intV<=endValue);
}


TValue TIntVariable::randomValue(const int &rand)
{ CHECK_INTERVAL

  if (!randomGenerator)
    randomGenerator = mlnew TRandomGenerator();

  return TValue(rand<0 ? randomGenerator->randint(startValue, endValue) : (rand % (endValue-startValue+1) + startValue));
}


int TIntVariable::noOfValues() const
{ return startValue<=endValue ? endValue-startValue+1 : -1; }



void TIntVariable::str2val(const string &valname, TValue &valu)
{ if (str2special(valname, valu))
    return ;

  int i;
  if (!sscanf(valname.c_str(), "%i", &i))
    raiseError("invalid argument (integer expected)");
  if ((startValue<=endValue) && ((i<startValue) || (i>endValue)))
    raiseError("value %i is out of range %i-%i", i, startValue, endValue);

  valu = TValue(i);
}


bool TIntVariable::str2val_try(const string &valname, TValue &valu)
{ if (str2special(valname, valu))
    return true;

  int i;
  if (!sscanf(valname.c_str(), "%i", &i) || ((startValue<=endValue) && ((i<startValue) || (i>endValue))))
    return false;

  valu = TValue(i);
  return true;
}


void TIntVariable::val2str(const TValue &valu, string &vname) const
{ if (valu.isSpecial())
    special2str(valu, vname);
  else {
    char buf[10];
    sprintf(buf, "%i", valu.intV);
    vname=buf;
  }
}



TFloatVariable::TFloatVariable()
: TVariable(TValue::FLOATVAR, true),
  startValue(-1.0),
  endValue(0.0),
  stepValue(-1.0),
  numberOfDecimals(3),
  adjustDecimals(2)
{}


TFloatVariable::TFloatVariable(const string &aname)
: TVariable(aname, TValue::FLOATVAR, true),
  startValue(-1.0),
  endValue(0.0),
  stepValue(-1.0),
  numberOfDecimals(3),
  adjustDecimals(2)
{}


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


inline int getNumberOfDecimals(const char *vals)
{
  const char *valsi;
  for(valsi = vals; *valsi && ((*valsi<'0') || (*valsi>'9')) && (*valsi!='.'); valsi++);
  if (!*valsi)
    return -1;

  for(; *valsi && (*valsi!='.'); valsi++);
  if (!*valsi)
    return 0;

  int decimals = 0;
  for(valsi++; *valsi && (*valsi>='0') && (*valsi<='9'); valsi++, decimals++);
  return decimals;
}


void TFloatVariable::str2val(const string &valname, TValue &valu)
{ if (str2special(valname, valu))
    return;

  const char *vals = valname.c_str();
  float f;
  if (!sscanf(vals, "%f", &f))
    raiseError("invalid argument (a number expected)");
  if ((startValue<=endValue) && (stepValue>0) && ((f<startValue) || (f>endValue)))
    raiseError("value %5.3f out of range %5.3f-%5.3f", f, startValue, endValue);

  valu = TValue(f);

  int decimals;
  switch (adjustDecimals) {
    case 2:
      numberOfDecimals = getNumberOfDecimals(vals);
      adjustDecimals = 1;
      break;
    case 1:
      decimals = getNumberOfDecimals(vals);
      if (decimals > numberOfDecimals)
        numberOfDecimals = decimals;
  }
}


bool TFloatVariable::str2val_try(const string &valname, TValue &valu)
{ if (str2special(valname, valu))
    return true;

  const char *vals = valname.c_str();
  float f;
  int ssr = sscanf(vals, "%f", &f);
  if (!ssr || (ssr==EOF) || ((startValue<=endValue) && (stepValue>0) && ((f<startValue) || (f>endValue))))
    return false;

  valu = TValue(f);

  int decimals;
  switch (adjustDecimals) {
    case 2:
      numberOfDecimals = getNumberOfDecimals(vals);
      adjustDecimals = 1;
      break;
    case 1:
      decimals = getNumberOfDecimals(vals);
      if (decimals > numberOfDecimals)
        numberOfDecimals = decimals;
  }

  return true;
}


void TFloatVariable::val2str(const TValue &valu, string &vname) const
{ if (valu.isSpecial())
    special2str(valu, vname);
  else {
    char buf[64];
    const float f = fabs(valu.floatV);
    if ((f>1e-6) && (f<1e6))
      sprintf(buf, "%.*f", numberOfDecimals, valu.floatV);
    else
      sprintf(buf, "%e", valu.floatV);
    vname = buf;
  }
}
