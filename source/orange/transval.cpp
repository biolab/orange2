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
{ 
  checkProperty(mapping);

  if (val.isSpecial())
    return;
  if (val.varType!=TValue::INTVAR)
    raiseError("invalid value type (discrete expected)");
  if (val.intV>=int(mapping->size()))
    raiseError("value out of range");

  int res = mapping->at(val.intV);
  if (res<0)
    val.setDK();
  else
    val.intV = res;
}



TDiscrete2Continuous::TDiscrete2Continuous(const int aval, bool inv, bool zeroB)
: value(aval),
  invert(inv),
  zeroBased(zeroB)
{}


void TDiscrete2Continuous::transform(TValue &val)
{ 
  if (val.varType!=TValue::INTVAR)
    raiseError("invalid value type (non-int)");
  if (val.isSpecial()) {
    if (zeroBased)
      raiseError("unknown value");
    else
      val = TValue(0.0);
  }
  else {
    if ((val.intV == value) != invert)
      val = TValue(1.0);
    else
      val = TValue(float(zeroBased ? 0.0 : -1.0));
  }
}


TOrdinal2Continuous::TOrdinal2Continuous(const float &f)
: factor(f)
{}


void TOrdinal2Continuous::transform(TValue &val)
{
  if (val.isSpecial())
    return;
  if (val.varType!=TValue::INTVAR)
    raiseError("invalid value type (discrete expected)");

  val = TValue(float(val.intV) * factor);
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


TDomainContinuizer::TDomainContinuizer()
: zeroBased(true),
  normalizeContinuous(false),
  baseValueSelection(FrequentIsBase),
  classTreatment(ReportError)
{}


PVariable TDomainContinuizer::discrete2continuous(TEnumVariable *evar, PVariable wevar, const int &val, bool inv) const
{
  PVariable newvar = mlnew TFloatVariable(evar->name+"="+evar->values->at(val));
  TClassifierFromVar *cfv = mlnew TClassifierFromVar(newvar, wevar);
  cfv->transformer = mlnew TDiscrete2Continuous(val, inv, zeroBased);
  newvar->getValueFrom = cfv;
  return newvar;
}


PVariable TDomainContinuizer::ordinal2continuous(TEnumVariable *evar, PVariable wevar) const
{
  PVariable newvar = mlnew TFloatVariable("C_"+evar->name);
  TClassifierFromVar *cfv = mlnew TClassifierFromVar(newvar, wevar);
  cfv->transformer = mlnew TOrdinal2Continuous(1.0/evar->values->size());
  newvar->getValueFrom = cfv;
  return newvar;
}


void TDomainContinuizer::discrete2continuous(PVariable var, TVarList &vars, const int &mostFrequent) const
{ 
  TEnumVariable *evar = var.AS(TEnumVariable);
  if (evar->values->size() < 2)
    return;

  if (evar->values->size() == 2)
    vars.push_back(discrete2continuous(evar, var, 1));

  int baseValue;
  switch (baseValueSelection) {
    case Ignore:
      return;

    case ReportError: 
      raiseError("attribute '%s' is multinomial", var->name.c_str());

    case AsOrdinal:
      vars.push_back(ordinal2continuous(evar, var));
      return;
 
    default:
      if (evar->baseValue >= 0)
        baseValue = evar->baseValue;
      else if (baseValueSelection == FrequentIsBase)
        baseValue = mostFrequent;
      else
        baseValue = 0;
    
      for(int val = 0, mval = evar->values->size(); val<mval; val++)
        if ((baseValueSelection==NValues) || (val!=baseValue))
          vars.push_back(discrete2continuous(evar, var, val));
  }
}


PVariable TDomainContinuizer::continuous2normalized(PVariable var, const float &avg, const float &span) const
{ 
  PVariable newvar = mlnew TFloatVariable("N_"+var->name);
  TClassifierFromVar *cfv = mlnew TClassifierFromVar(newvar, var);
  cfv->transformer = mlnew TNormalizeContinuous(avg, span);
  newvar->getValueFrom = cfv;
  return newvar;
}


PVariable TDomainContinuizer::discreteClass2continous(PVariable classVar, const int &targetClass) const
{
  TEnumVariable *eclass = classVar.AS(TEnumVariable);
  const int classBase = targetClass >= 0 ? targetClass : eclass->baseValue;

  if (classBase >= 0) {
    if (classBase >= int(eclass->values->size()))
        raiseError("base class value out of range");

    PVariable newClassVar = mlnew TFloatVariable(eclass->name+"<>"+eclass->values->at(classBase));
    TClassifierFromVar *cfv = mlnew TClassifierFromVar(newClassVar, classVar);
    cfv->transformer = mlnew TDiscrete2Continuous(classBase, false, zeroBased);
    newClassVar->getValueFrom = cfv;
    return newClassVar;
  }

  if (classTreatment == Ignore)
    return classVar;

  if (eclass->values->size() < 2)
    raiseError("class has less than two different values");

  if (eclass->values->size() == 2)
    return discrete2continuous(eclass, classVar, 1);

  if (classTreatment != AsOrdinal)
    raiseError("class '%s' is multinomial", eclass->name.c_str());

  return ordinal2continuous(eclass, classVar);
}


PDomain TDomainContinuizer::operator()(PDomain dom, const int &targetClass) const
{ 
  PVariable otherAttr = dom->hasOtherAttributes((targetClass>=0) || (classTreatment != Ignore));
  if (otherAttr)
    raiseError("attribute '%s' is of a type that cannot be converted to continuous", otherAttr->name.c_str());
  
  if (normalizeContinuous)
    raiseError("cannot normalize continuous attributes without seeing the data");
  if (baseValueSelection == FrequentIsBase)
    raiseError("cannot determine the most frequent values without seeing the data");

  PVariable newClassVar;
  if (((targetClass>=0) || (classTreatment != Ignore)) && (dom->classVar->varType == TValue::INTVAR))
    newClassVar = discreteClass2continous(dom->classVar, targetClass);
  else
    newClassVar = dom->classVar;

  TVarList newvars;
  PITERATE(TVarList, vi, dom->attributes)
    if ((*vi)->varType == TValue::INTVAR)
      discrete2continuous(*vi, newvars, -1);
    else
      newvars.push_back(*vi);
    
  return mlnew TDomain(newClassVar, newvars);
}


PDomain TDomainContinuizer::operator()(PExampleGenerator egen, const int &weightID, const int &targetClass) const
{ 
  bool convertClass = (targetClass>=0) || (classTreatment != Ignore);

  if (!convertClass && (targetClass>=0))
    raiseWarning("class is not being converted, 'targetClass' argument is ignored");

  if (!normalizeContinuous)
    return call(egen->domain, targetClass);

  const TDomain &domain = egen->domain.getReference();

  PVariable otherAttr = domain.hasOtherAttributes(convertClass);
  if (otherAttr)
    raiseError("attribute '%s' is of a type that cannot be converted to continuous", otherAttr->name.c_str());


  vector<float> avgs, spans;
  vector<int> mostFrequent;

  if ((baseValueSelection == FrequentIsBase) && domain.hasDiscreteAttributes(convertClass)) {
    TDomainDistributions ddist(egen, weightID, false, true);
    ITERATE(TDomainDistributions, ddi, ddist) {
      if ((*ddi)->variable->varType == TValue::INTVAR) {
        // won't call modus here, I want the lowest values if there are more values with equal frequencies
        int val = 0, highVal = 0;
        float highestF = 0.0;
        TDiscDistribution *ddva = (*ddi).AS(TDiscDistribution);
        for(TDiscDistribution::const_iterator di(ddva->begin()), de(ddva->end()); di!=de; di++, val++)
          if (*di>highestF) {
            highestF = *di;
            highVal = val;
          }
        mostFrequent.push_back(highVal);
      }
      else
        mostFrequent.push_back(-1);
    }
  }

  if (domain.hasContinuousAttributes(convertClass)) {
    TDomainBasicAttrStat dombas(egen);
    ITERATE(TDomainBasicAttrStat, di, dombas)
    if ((*di)->variable->varType == TValue::FLOATVAR) {
      avgs.push_back((*di)->avg);
      spans.push_back((*di)->max-(*di)->min);
    }
    else {
      avgs.push_back(-1);
      spans.push_back(-1);
    }
  }


  PVariable newClassVar;
  if (convertClass) {
    if (domain.classVar->varType == TValue::INTVAR)
      newClassVar = discreteClass2continous(domain.classVar, targetClass);
    else
      newClassVar = continuous2normalized(domain.classVar, avgs.back(), spans.back());
  }
  else
    newClassVar = domain.classVar;

  TVarList newvars;
  TVarList::const_iterator vi(domain.attributes->begin()), ve(domain.attributes->end());
  for(int i = 0; vi!=ve; vi++, i++)
    if ((*vi)->varType == TValue::INTVAR)
      discrete2continuous(*vi, newvars, mostFrequent[i]);
    else
      newvars.push_back(continuous2normalized(*vi, avgs[i], spans[i]));
    
  return mlnew TDomain(newClassVar, newvars);
}

