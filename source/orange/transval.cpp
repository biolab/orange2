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


#include "values.hpp"
#include "errors.hpp"
#include "vars.hpp"
#include "examples.hpp"
#include "examplegen.hpp"

#include "classfromvar.hpp"
#include "basstat.hpp"


#include "transval.ppp"


TTransformValue::TTransformValue(TTransformValue *tr)
: subTransform(tr)
{}


TTransformValue::TTransformValue(const TTransformValue &old)
: TOrange(old),
  subTransform(CLONE(TTransformValue, old.subTransform))
{}


TValue TTransformValue::operator()(const TValue &val)
{
  TValue newval=val;
  transform(newval);
  return newval;
}


void TTransformValue::transform(TValue &val)
{ if (subTransform)
    subTransform->transform(val); 
}



TMapIntValue::TMapIntValue(PIntList al)
: mapping(al)
{}


TMapIntValue::TMapIntValue(const TIntList &al)
: mapping(mlnew TIntList(al))
{}


void TMapIntValue::transform(TValue &val)
{ checkProperty(mapping);

  if (val.isSpecial())
    return;
  if (val.varType!=TValue::INTVAR)
    raiseErrorWho("transform", "invalid value type (discrete expected)");
  if (val.intV>=int(mapping->size()))
    raiseErrorWho("transform", "value out of range");

  int res = mapping->at(val.intV);
  if (res<0)
    val.setDK();
  else
    val.intV = res;
}



TDiscrete2Continuous::TDiscrete2Continuous(const int aval, bool inv)
: value(aval),
  invert(inv)
{}

void TDiscrete2Continuous::transform(TValue &val)
{ if (val.varType!=TValue::INTVAR)
    raiseError("invalid value type (non-int)");
  if (val.isSpecial())
    raiseError("unknown value");

  val = TValue(float(((val.intV == value) != invert) ? 1.0 : 0.0));
}



TNormalizeContinuous::TNormalizeContinuous(const float av, const float sp)
 : average(av), span(sp)
 { if (span==0.0)
     span=1.0;
 }


void TNormalizeContinuous::transform(TValue &val)
{ if (val.varType!=TValue::FLOATVAR)
    raiseError("invalid value type (non-float)");

  val = TValue(float(val.isSpecial() ? 0.0 : (2*(val.floatV-average)/span)));
}


PVariable discrete2continuous(TEnumVariable *evar, PVariable wevar, const int &val)
{
  PVariable newvar = mlnew TFloatVariable(evar->name+"="+evar->values->at(val));
  TClassifierFromVar *cfv = mlnew TClassifierFromVar(newvar, wevar);
  cfv->transformer = mlnew TDiscrete2Continuous(val);
  newvar->getValueFrom = cfv;
  return newvar;
}


void discrete2continuous(PVariable var, TVarList &vars)
{ 
  TEnumVariable *evar = var.AS(TEnumVariable);
  if (evar->values->size() < 2)
    return;

  const int baseValue = evar->baseValue>=0 ? evar->baseValue : 0;
  for(int val = 0, mval = evar->values->size(); val<mval; val++)
    if (val!=baseValue)
      vars.push_back(discrete2continuous(evar, var, val));
}


PVariable normalizeContinuous(PVariable var, const float &avg, const float &span)
{ 
  PVariable newvar = mlnew TFloatVariable(var->name+"_N");
  TClassifierFromVar *cfv = mlnew TClassifierFromVar(newvar, var);
  cfv->transformer = mlnew TNormalizeContinuous(avg, span);
  newvar->getValueFrom = cfv;
  return newvar;
}


PVariable discreteClass2continous(PVariable classVar, const int &targetClass, bool invertClass)
{
  TEnumVariable *eclass = classVar.AS(TEnumVariable);
  const int classBase = targetClass >= 0 ? targetClass : eclass->baseValue;

  if (eclass->values->size() < 2)
    raiseError("class has less than two different values");
  if (classBase >= int(eclass->values->size()))
      raiseError("base class value out of range");

  if (eclass->values->size() == 2) {
    if (classBase >= 0)
      return discrete2continuous(eclass, classVar, 1-classBase);
    else
      return discrete2continuous(eclass, classVar, 1);
  }
  else { // class has more than 2 values
    if (classBase < 0)
      raiseError("cannot handle multinomial classes if baseValue is nor set nor given");

    PVariable newClassVar = mlnew TFloatVariable(eclass->name+"<>"+eclass->values->at(classBase));
    TClassifierFromVar *cfv = mlnew TClassifierFromVar(newClassVar, classVar);
    cfv->transformer = mlnew TDiscrete2Continuous(classBase, true);
    newClassVar->getValueFrom = cfv;
    return newClassVar;
  }
}


PDomain regressionDomain(PDomain dom, const int &targetClass, bool invertClass)
{ 
  PVariable newClassVar;
  if (dom->classVar->varType == TValue::INTVAR)
    newClassVar = discreteClass2continous(dom->classVar, targetClass, invertClass);
  else if (dom->classVar->varType == TValue::FLOATVAR)
    newClassVar = dom->classVar;
  else
    raiseError("class '%s' cannot be converted to continuous", dom->classVar->name.c_str());

  TVarList newvars;
  PITERATE(TVarList, vi, dom->attributes)
    if ((*vi)->varType == TValue::INTVAR)
      discrete2continuous(*vi, newvars);
    else if ((*vi)->varType == TValue::FLOATVAR)
      newvars.push_back(*vi);
    else
      raiseError("attribute '%s' cannot be converted to continuous", (*vi)->name.c_str());
    
  return mlnew TDomain(newClassVar, newvars);
}


PDomain regressionDomain(PExampleGenerator egen, const int &targetClass, bool invertClass, bool normContinuous)
{ 
  if (!normContinuous)
    return regressionDomain(egen->domain, targetClass, invertClass);

  const TDomain &domain = egen->domain.getReference();
  TDomainBasicAttrStat dombas(egen);

  PVariable newClassVar;
  if (domain.classVar->varType == TValue::INTVAR)
    newClassVar = discreteClass2continous(domain.classVar, targetClass, invertClass);
  else if (domain.classVar->varType == TValue::FLOATVAR) {
    const TBasicAttrStat &cstat = dombas.back().getReference();
    newClassVar = normalizeContinuous(domain.classVar, cstat.avg, cstat.max - cstat.min);
  }
  else
    raiseError("class '%s' cannot be converted to continuous", domain.classVar->name.c_str());

  TVarList newvars;
  TVarList::const_iterator vi(domain.attributes->begin()), ve(domain.attributes->end());
  TDomainBasicAttrStat::iterator di(dombas.begin());
  for(; vi!=ve; vi++, di++)
    if ((*vi)->varType == TValue::INTVAR)
      discrete2continuous(*vi, newvars);
    else if ((*vi)->varType == TValue::FLOATVAR)
      newvars.push_back(normalizeContinuous(*vi, (*di)->avg, (*di)->max-(*di)->min));
    else
      raiseError("attribute '%s' cannot be converted to continuous", (*vi)->name.c_str());
    
  return mlnew TDomain(newClassVar, newvars);
}


bool hasNonContinuousAttributes(PDomain dom, bool checkClass)
{
  PITERATE(TVarList, vi, checkClass ? dom->attributes : dom->variables)
    if ((*vi)->varType != TValue::FLOATVAR)
      return true;
  return false;
}