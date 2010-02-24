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


#include "values.hpp"
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

  val = TValue(float(val.isSpecial() ? 0.0 : (val.floatV-average)/span));
}


TDomainContinuizer::TDomainContinuizer()
: zeroBased(true),
  continuousTreatment(Leave),
  multinomialTreatment(FrequentIsBase),
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


PVariable TDomainContinuizer::ordinal2continuous(TEnumVariable *evar, PVariable wevar, const float &factor) const
{
  PVariable newvar = mlnew TFloatVariable("C_"+evar->name);
  TClassifierFromVar *cfv = mlnew TClassifierFromVar(newvar, wevar);
  TOrdinal2Continuous *transf = mlnew TOrdinal2Continuous(1.0/evar->values->size());
  cfv->transformer = transf;
  transf->factor = factor;
  newvar->getValueFrom = cfv;
  return newvar;
}


void TDomainContinuizer::discrete2continuous(PVariable var, TVarList &vars, const int &mostFrequent) const
{ 
  TEnumVariable *evar = var.AS(TEnumVariable);
  if (evar->values->size() < 2)
    return;

  int baseValue;
  switch (multinomialTreatment) {
    case Ignore:
      if (evar->values->size() == 2)
        vars.push_back(discrete2continuous(evar, var, 1));
      return;

    case IgnoreAllDiscrete:
      return;
      
    case ReportError:
      if (evar->values->size() == 2) {
        vars.push_back(discrete2continuous(evar, var, 1));
        return;
      }
      raiseError("attribute '%s' is multinomial", var->name.c_str());

    case AsOrdinal:
      vars.push_back(ordinal2continuous(evar, var, 1));
      return;

    case AsNormalizedOrdinal:
      vars.push_back(ordinal2continuous(evar, var, 1.0 / (evar->values->size()-1.0)));
      return;
 
    default:
      if (evar->baseValue >= 0)
        baseValue = evar->baseValue;
      else if (multinomialTreatment == FrequentIsBase)
        baseValue = mostFrequent;
      else
        baseValue = 0;
    
      if (evar->values->size() == 2)
        vars.push_back(discrete2continuous(evar, var, 1-baseValue));
      else
        for(int val = 0, mval = evar->values->size(); val<mval; val++)
          if ((multinomialTreatment==NValues) || (val!=baseValue))
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

    PVariable newClassVar = mlnew TFloatVariable(eclass->name+"="+eclass->values->at(classBase));
    TClassifierFromVar *cfv = mlnew TClassifierFromVar(newClassVar, classVar);
    cfv->transformer = mlnew TDiscrete2Continuous(classBase, false, zeroBased);
    newClassVar->getValueFrom = cfv;
    return newClassVar;
  }

  if ((classTreatment == Ignore) || (eclass->values->size() < 2))
    return classVar;

  if (eclass->values->size() == 2)
    return discrete2continuous(eclass, classVar, 1);

  if (classTreatment == AsOrdinal)
    return ordinal2continuous(eclass, classVar, 1.0);

  if (classTreatment == AsNormalizedOrdinal)
    return ordinal2continuous(eclass, classVar, 1.0 / (eclass->values->size() - 1));

  raiseError("class '%s' is multinomial", eclass->name.c_str());
  return PVariable();
}


PDomain TDomainContinuizer::operator()(PDomain dom, const int &targetClass) const
{ 
  PVariable otherAttr = dom->hasOtherAttributes((targetClass>=0) || (classTreatment != Ignore));
  if (otherAttr)
    raiseError("attribute '%s' is of a type that cannot be converted to continuous", otherAttr->name.c_str());
  
  if (continuousTreatment)
    raiseError("cannot normalize continuous attributes without seeing the data");
  if (multinomialTreatment == FrequentIsBase)
    raiseError("cannot determine the most frequent values without seeing the data");

  PVariable newClassVar;
  if (dom->classVar) {
    if (((targetClass>=0) || (classTreatment != Ignore)) && (dom->classVar->varType == TValue::INTVAR) && (dom->classVar->noOfValues() >= 1))
      newClassVar = discreteClass2continous(dom->classVar, targetClass);
    else
      newClassVar = dom->classVar;
  }

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
  bool convertClass = ((targetClass>=0) || (classTreatment != Ignore)) && egen->domain->classVar;

  if (!convertClass && (targetClass>=0))
    raiseWarning("class is not being converted, 'targetClass' argument is ignored");

  if (!continuousTreatment && (multinomialTreatment != FrequentIsBase))
    return call(egen->domain, targetClass);

  const TDomain &domain = egen->domain.getReference();

  PVariable otherAttr = domain.hasOtherAttributes(convertClass);
  if (otherAttr)
    raiseError("attribute '%s' is of a type that cannot be converted to continuous", otherAttr->name.c_str());


  vector<float> avgs, spans;
  vector<int> mostFrequent;

  bool hasMostFrequent = (multinomialTreatment == FrequentIsBase) && domain.hasDiscreteAttributes(convertClass);
  if (hasMostFrequent) {
    TDomainDistributions ddist(egen, weightID, false, true);
    ITERATE(TDomainDistributions, ddi, ddist) {
      if (*ddi) {
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

  if (continuousTreatment && domain.hasContinuousAttributes(convertClass)) {
    TDomainBasicAttrStat dombas(egen);
    ITERATE(TDomainBasicAttrStat, di, dombas)
    if (*di) {
      if (continuousTreatment == NormalizeBySpan) {
        const float &min = (*di)->min;
        const float &max = (*di)->max;
        if (zeroBased) {
          avgs.push_back(min);
          spans.push_back(max-min);
        }
        else {
          avgs.push_back((max+min) / 2.0);
          spans.push_back((max-min) / 2.0);
        }
      }
      else {
        avgs.push_back((*di)->avg);
        spans.push_back((*di)->dev);
      }
    }
    else {
      avgs.push_back(-1);
      spans.push_back(-1);
    }
  }


  PVariable newClassVar;
  if (convertClass && (domain.classVar->varType == TValue::INTVAR))
      newClassVar = discreteClass2continous(domain.classVar, targetClass);
  else
    newClassVar = domain.classVar;

  TVarList newvars;
  TVarList::const_iterator vi(domain.attributes->begin()), ve(domain.attributes->end());
  for(int i = 0; vi!=ve; vi++, i++)
    if ((*vi)->varType == TValue::INTVAR)
      discrete2continuous(*vi, newvars, hasMostFrequent ? mostFrequent[i] : 0);
    else
      if (continuousTreatment)
        newvars.push_back(continuous2normalized(*vi, avgs[i], spans[i]));
      else
        newvars.push_back(*vi);
  
  TDomain *newDomain = mlnew TDomain(newClassVar, newvars);
  PDomain wnewDomain = newDomain;
  wnewDomain->metas = egen->domain->metas;

  return wnewDomain;
}

