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


#include "random.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "meta.hpp"


#include "filter.hpp"
#include "trindex.hpp"
#include "spec_gen.hpp"
#include "stladdon.hpp"
#include "tabdelim.hpp"
#include "discretize.hpp"
#include "classfromvar.hpp"
#include "cost.hpp"
#include "survival.hpp"

#include <string>
#include "preprocessors.ppp"

DEFINE_TOrangeMap_classDescription(PVariable, PValueFilter, true, true, "VariableFilterMap")
DEFINE_TOrangeMap_classDescription(PVariable, float, true, false, "VariableFloatMap")

#ifdef _MSC_VER
  #pragma warning (disable : 4100) // unreferenced local parameter (macros name all arguments)
#endif

void atoms2varList(const vector<string> &vnames, PDomain domain, TVarList &varList, const string &error)
{ const_ITERATE(TStringList, ni, vnames) {
    int vnum = domain->getVarNum(*ni, false);
    if (vnum<0) {
      char errorout[128];
      sprintf(errorout, error.c_str(), (*ni).c_str());
      raiseError(errorout);
    }
    else 
      varList.push_back(domain->variables->at(vnum));
  }
}

void string2varList(const string &str, PDomain domain, TVarList &varList, const string &error)
{
  vector<string> vnames;
  string2atoms(str, vnames);
  atoms2varList(vnames, domain, varList, error);
}

PStringList string2atoms(const string &line)
{ PStringList atoms = mlnew TStringList();
  string2atoms(line, atoms->__orvector);
  return atoms;
}




PExampleGenerator TPreprocessor::filterExamples(PFilter filter, PExampleGenerator generator)
{ TFilteredGenerator fg(filter, generator);
  return PExampleGenerator(mlnew TExampleTable(PExampleGenerator(fg))); 
}



TPreprocessor_ignore::TPreprocessor_ignore()
: attributes(mlnew TVarList())
{}


TPreprocessor_ignore::TPreprocessor_ignore(PVarList attrs)
: attributes(attrs)
{}


PExampleGenerator TPreprocessor_ignore::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  PDomain outDomain = CLONE(TDomain, gen->domain);
  PITERATE(TVarList, vi, attributes)
    if (!outDomain->delVariable(*vi))
      if (*vi == outDomain->classVar)
        outDomain->removeClass();
      else
        raiseError("attribute '%s' not found", (*vi)->name.c_str());

  newWeight = weightID;
  return PExampleGenerator(mlnew TExampleTable(outDomain, gen));
}



TPreprocessor_select::TPreprocessor_select()
: attributes(mlnew TVarList())
{}


TPreprocessor_select::TPreprocessor_select(PVarList attrs)
: attributes(attrs)
{}


PExampleGenerator TPreprocessor_select::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  PDomain outDomain = CLONE(TDomain, gen->domain);
  TVarList::const_iterator bi(attributes->begin()), be(attributes->end());

  PITERATE(TVarList, vi, gen->domain->attributes)
    if (find(bi, be, *vi)==be)
      outDomain->delVariable(*vi);

  if (find(bi, be, outDomain->classVar) == be)
    outDomain->removeClass();

  newWeight = weightID;
  return PExampleGenerator(mlnew TExampleTable(outDomain, gen));
}




TPreprocessor_drop::TPreprocessor_drop()
: values(mlnew TVariableFilterMap())
{}


TPreprocessor_drop::TPreprocessor_drop(PVariableFilterMap avalues)
: values(avalues)
{}


PExampleGenerator TPreprocessor_drop::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ TValueFilterList *dropvalues = mlnew TValueFilterList(gen->domain->variables->size());
  PValueFilterList wdropvalues = dropvalues;
  PITERATE(TVariableFilterMap, vi, values)
    (*dropvalues)[gen->domain->getVarNum((*vi).first)] = (*vi).second;

  newWeight = weightID;
  return filterExamples(mlnew TFilter_Values(wdropvalues, true, true, gen->domain), gen);
}

  

TPreprocessor_take::TPreprocessor_take()
: values(mlnew TVariableFilterMap())
{}


TPreprocessor_take::TPreprocessor_take(PVariableFilterMap avalues)
: values(avalues)
{}


PExampleGenerator TPreprocessor_take::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  newWeight = weightID;
  return filterExamples(constructFilter(values, gen->domain), gen);
}


PFilter TPreprocessor_take::constructFilter(PVariableFilterMap values, PDomain domain)
{ TValueFilterList *dropvalues = mlnew TValueFilterList(domain->variables->size());
  PValueFilterList wdropvalues = dropvalues;
  PITERATE(TVariableFilterMap, vi, values)
    (*dropvalues)[domain->getVarNum((*vi).first)] = (*vi).second;
  return mlnew TFilter_Values(wdropvalues, true, false, domain);
}



PExampleGenerator TPreprocessor_removeDuplicates::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ PExampleGenerator table = mlnew TExampleTable(gen);

  if (weightID)
    newWeight = weightID;
  else {
    newWeight = getMetaID();
    TValue val0(float(1.0));
    table.AS(TExampleTable)->addMetaAttribute(newWeight, val0);
  }

  table.AS(TExampleTable)->removeDuplicates(newWeight);
  return table;
}



PExampleGenerator TPreprocessor_dropMissing::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ newWeight = weightID;
  return filterExamples(mlnew TFilter_hasSpecial(true), gen);
}


PExampleGenerator TPreprocessor_takeMissing::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ newWeight = weightID;
  return filterExamples(mlnew TFilter_hasSpecial(false), gen);
}



PExampleGenerator TPreprocessor_dropMissingClasses::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ newWeight = weightID;
  return filterExamples(mlnew TFilter_hasClassValue(true), gen);
}



PExampleGenerator TPreprocessor_takeMissingClasses::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ newWeight = weightID;
  return filterExamples(mlnew TFilter_hasClassValue(false), gen);
}



TPreprocessor_addNoise::TPreprocessor_addNoise()
: proportions(mlnew TVariableFloatMap()),
  defaultProportion(0.0)  
{}


TPreprocessor_addNoise::TPreprocessor_addNoise(PVariableFloatMap probs, const float &defprob)
: proportions(probs),
  defaultProportion(defprob)  
{}


PExampleGenerator TPreprocessor_addNoise::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  newWeight = weightID;

  if (!proportions && (defaultProportion<=0.0))
    return mlnew TExampleTable(gen);

  const TDomain &domain = gen->domain.getReference();
  vector<float> ps(domain.attributes->size(), defaultProportion);

  if (proportions)
    PITERATE(TVariableFloatMap, vi, proportions) {
      if ((*vi).first == domain.classVar)
        ps.push_back((*vi).second);
      ps[domain.getVarNum((*vi).first)] = (*vi).second;
    }

  TExampleTable *table = mlnew TExampleTable(gen);
  PExampleGenerator wtable = table;

  const int n = table->size();
  TMakeRandomIndices2 makerind;

  int idx = 0;
  TVarList::const_iterator vi (table->domain->variables->begin());
  vector<float>::const_iterator pi(ps.begin()), pe(ps.end());
  for(; pi!=pe; idx++, vi++, pi++)
    if (*pi>0.0) {
      PLongList rind = makerind(n, 1 - *pi);
      const TVariable &var = (*vi).getReference();

      int eind = 0;
      PITERATE(TLongList, ri, rind) {
        if (*ri)
          (*table)[eind][idx] = var.randomValue();
        eind++;
      }
    }

  return wtable;
} 



TPreprocessor_addGaussianNoise::TPreprocessor_addGaussianNoise()
: deviations(mlnew TVariableFloatMap()),
  defaultDeviation(0.0)
{}



TPreprocessor_addGaussianNoise::TPreprocessor_addGaussianNoise(PVariableFloatMap devs, const float &defdev)
: deviations(devs),
  defaultDeviation(defdev)  
{}



PExampleGenerator TPreprocessor_addGaussianNoise::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  newWeight = weightID;

  if (!deviations && (defaultDeviation<=0.0))
    return mlnew TExampleTable(gen);

  const TDomain &domain = gen->domain.getReference();
  vector<float> ps(domain.attributes->size(), defaultDeviation);

  if (deviations)
    PITERATE(TVariableFloatMap, vi, deviations) {
      if ((*vi).first == domain.classVar)
        ps.push_back((*vi).second);
      ps[domain.getVarNum((*vi).first)] = (*vi).second;
    }

  TGaussianNoiseGenerator gg = TGaussianNoiseGenerator(ps, gen);
  return PExampleGenerator(mlnew TExampleTable(PExampleGenerator(gg)));
}



TPreprocessor_addMissing::TPreprocessor_addMissing()
: proportions(mlnew TVariableFloatMap()),
  defaultProportion(0.0),
  specialType(valueDK)
{}


TPreprocessor_addMissing::TPreprocessor_addMissing(PVariableFloatMap probs, const float &defprob, const int &st)
: proportions(probs),
  defaultProportion(defprob),
  specialType(st)
{}


PExampleGenerator TPreprocessor_addMissing::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  newWeight = weightID;
  
  if (!proportions && (defaultProportion<=0.0))
    return mlnew TExampleTable(gen);

  const TDomain &domain = gen->domain.getReference();
  vector<float> ps(domain.attributes->size(), defaultProportion);

  if (proportions)
    PITERATE(TVariableFloatMap, vi, proportions) {
      if ((*vi).first == domain.classVar)
        ps.push_back((*vi).second);
      ps[domain.getVarNum((*vi).first)] = (*vi).second;
    }

  TExampleTable *table = mlnew TExampleTable(gen);
  PExampleGenerator wtable = table;

  const int n = table->size();
  TMakeRandomIndices2 makerind;

  int idx = 0;
  TVarList::const_iterator vi (table->domain->variables->begin());
  vector<float>::const_iterator pi(ps.begin()), pe(ps.end());
  for(; pi!=pe; idx++, vi++, pi++)
    if (*pi>0.0) {
      PLongList rind = makerind(n, 1 - *pi);
      const int &varType = (*vi)->varType;

      int eind = 0;
      PITERATE(TLongList, ri, rind) {
        if (*ri)
          (*table)[eind][idx] = TValue(varType, specialType);
        eind++;
      }
    }

  return wtable;
}



TPreprocessor_addClassNoise::TPreprocessor_addClassNoise(const float &cn)
: proportion(cn)
{}


PExampleGenerator TPreprocessor_addClassNoise::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  if (!gen->domain->classVar)
    raiseError("Class-less domain");

  TExampleTable *table = mlnew TExampleTable(gen);
  PExampleGenerator wtable = table;

  if (proportion>0.0) {
    TMakeRandomIndices2 mri2;
    PLongList rind(mri2(table->size(), 1-proportion));

    const TVariable &classVar = table->domain->classVar.getReference();
    int eind = 0;
    PITERATE(TLongList, ri, rind) {
      if (*ri)
        (*table)[eind].setClass(classVar.randomValue());
      eind++;
    }
  }

  newWeight = weightID;
  return wtable;
}



TPreprocessor_addGaussianClassNoise::TPreprocessor_addGaussianClassNoise(const float &dev)
: deviation(dev)
{}


PExampleGenerator TPreprocessor_addGaussianClassNoise::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  if (!gen->domain->classVar)
    raiseError("Class-less domain");

  newWeight = weightID;

  if (deviation>0.0) {
    vector<float> deviations(gen->domain->variables->size(), 0);
    deviations.back() = deviation;
    TGaussianNoiseGenerator gngen(deviations, gen);
    return PExampleGenerator(mlnew TExampleTable(PExampleGenerator(gngen)));
  }

  else
    return mlnew TExampleTable(gen);
}


TPreprocessor_addMissingClasses::TPreprocessor_addMissingClasses(const float &cm, const int &st)
: proportion(cm),
  specialType(st)
{}
  
  
PExampleGenerator TPreprocessor_addMissingClasses::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  if (!gen->domain->classVar)
    raiseError("Class-less domain");

  TExampleTable *table = mlnew TExampleTable(gen);
  PExampleGenerator wtable = table;

  if (proportion>0.0) {
    TMakeRandomIndices2 mri2;
    PLongList rind(mri2(table->size(), 1-proportion));

    const TVariable &classVar = table->domain->classVar.getReference();
    const int &varType = classVar.varType;
    int eind = 0;
    PITERATE(TLongList, ri, rind) {
      if (*ri)
        (*table)[eind].setClass(TValue(varType, specialType));
      eind++;
    }
  }

  newWeight = weightID;
  return wtable;
}



TPreprocessor_addCostWeight::TPreprocessor_addCostWeight()
: classWeights(mlnew TFloatList),
  equalize(false)
{}


TPreprocessor_addCostWeight::TPreprocessor_addCostWeight(PFloatList cw, const bool &eq)
: equalize(eq),
  classWeights(cw)
{}


PExampleGenerator TPreprocessor_addCostWeight::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  if (!gen->domain->classVar || (gen->domain->classVar->varType != TValue::INTVAR))
    raiseError("Class-less domain or non-discrete class");

  TExampleTable *table = mlnew TExampleTable(gen);
  PExampleGenerator wtable = table;

  const int nocl = gen->domain->classVar->noOfValues();

  if (!equalize && !classWeights->size() || !nocl) {
    newWeight = 0;
    return wtable;
  }

  newWeight = getMetaID();

  vector<float> weights(nocl, 1.0);
  if (classWeights) {
    int i = 0;
    PITERATE(vector<float>, ci, classWeights) {
      weights[i++] = *ci;
      if (i==nocl)
        break;
    }
  }  

  if (equalize) {
    PDistribution dist(getClassDistribution(gen, weightID));
    const TDiscDistribution &ddist = CAST_TO_DISCDISTRIBUTION(dist);
    float N = ddist.abs;
    TDiscDistribution dv;
    vector<float>::iterator wi(weights.begin());
    const_ITERATE(TDiscDistribution, di, ddist) {
      if (*di>0.0)
        (*(wi++)) *= N / nocl / *di;
      else
        *(wi++) = 1.0;
    }
  }

  PEITERATE(ei, table)
    (*ei).meta.setValue(newWeight, TValue(WEIGHT(*ei) * weights[(*ei).getClass().intV]));

  return wtable;
}



TPreprocessor_addCensorWeight::TPreprocessor_addCensorWeight()
: outcomeVar(),
  eventValue(),
  timeID(0),
  method(km),
  maxTime(0.0)
{}


TPreprocessor_addCensorWeight::TPreprocessor_addCensorWeight(PVariable ov, const TValue &ev, const int &tv, const int &me, const float &mt)
: outcomeVar(ov),
  eventValue(ev),
  timeID(tv),
  method(me),
  maxTime(0.0)
{}


PExampleGenerator TPreprocessor_addCensorWeight::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  checkProperty(outcomeVar);

  if (eventValue.isSpecial())
    raiseError("'eventValue' not set");

  if (eventValue.varType != TValue::INTVAR)
    raiseError("'eventValue' invalid (discrete value expected)");

  if (!timeID)
    raiseError("'timeVar' not set");

  if ((method<km) || (method>linear))
    raiseError("invalid method");

  const int outcomeIndex = outcomeVar ? gen->domain->getVarNum(outcomeVar) : gen->domain->attributes->size();
  const int failIndex = eventValue.intV;

  TExampleTable *table = mlnew TExampleTable(gen);
  PExampleGenerator wtable = table;

  if (method == linear) {
    float thisMaxTime = maxTime;
    if (thisMaxTime<=0.0)
      PEITERATE(ei, table) {
        const TValue &tme = (*ei).meta[timeID];
        if (!tme.isSpecial()) {
          if (tme.varType != TValue::FLOATVAR)
            raiseError("invalid time (continuous attribute expected)");
          else
            if (tme.floatV>thisMaxTime)
              thisMaxTime = tme.floatV;
        }
      }

    if (thisMaxTime<=0.0)
      raiseError("invalid time values (max<=0)");

    newWeight = getMetaID();
    PEITERATE(ei, table) {
      if (!(*ei)[outcomeIndex].isSpecial() && (*ei)[outcomeIndex].intV==failIndex)
        (*ei).meta.setValue(newWeight, TValue(WEIGHT(*ei)));
      else {
        const TValue &tme = (*ei).meta[timeID];
        if (tme.isSpecial())
          (*ei).meta.setValue(newWeight, 0.0);
        else if (tme.varType != TValue::FLOATVAR)
          raiseError("invalid time (continuous value expected)");
        else
          (*ei).meta.setValue(newWeight, TValue(WEIGHT(*ei) * tme.floatV / thisMaxTime));
      }
    }
  }

  else { // method == km or nmr
    TKaplanMeier *kaplanMeier = mlnew TKaplanMeier(gen, outcomeIndex, failIndex, timeID, weightID);;

    if (method==km)
      kaplanMeier->toFailure();
    else // method == nmr
      kaplanMeier->toLog();

    if (maxTime>0.0)
      kaplanMeier->normalizedCut(maxTime);

    newWeight = getMetaID();
    PEITERATE(ei, table) {
      if (!(*ei)[outcomeIndex].isSpecial() && (*ei)[outcomeIndex].intV==failIndex)
        (*ei).meta.setValue(newWeight, TValue(WEIGHT(*ei)));
      else {
        const TValue &tme = (*ei).meta[timeID];
        if (tme.isSpecial())
          (*ei).meta.setValue(newWeight, 0.0);
        else if (tme.varType != TValue::FLOATVAR)
          raiseError("invalid time (continuous value expected)");
        else
          (*ei).meta.setValue(newWeight, TValue(WEIGHT(*ei) * kaplanMeier->operator()(tme.floatV)));
      }
    }
  }

  return wtable;
}
  


TPreprocessor_discretize::TPreprocessor_discretize()
: attributes(),
  notClass(true),
  method()
{}


TPreprocessor_discretize::TPreprocessor_discretize(PVarList attrs, const bool &nocl, PDiscretization meth)
: attributes(attrs),
  notClass(nocl),
  method(meth)
{}


PExampleGenerator TPreprocessor_discretize::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  checkProperty(method);

  const TDomain &domain = gen->domain.getReference();

  vector<int> discretizeId;
  if (attributes && attributes->size()) {
    PITERATE(TVarList, vi, attributes)
      discretizeId.push_back(domain.getVarNum(*vi));
  }
  else {
    int idx = 0;
    const_PITERATE(TVarList, vi, domain.attributes) {
      if ((*vi)->varType==TValue::FLOATVAR)
        discretizeId.push_back(idx);
      idx++;
    }
    if (!notClass && (domain.classVar->varType == TValue::FLOATVAR))
      discretizeId.push_back(idx);
  }

  newWeight = weightID;
  return mlnew TExampleTable(PDomain(mlnew TDiscretizedDomain(gen, discretizeId, weightID, method)), gen);
}




TPreprocessor_filter::TPreprocessor_filter(PFilter filt)
: filter(filt)
{}

PExampleGenerator TPreprocessor_filter::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ checkProperty(filter);
  newWeight = weightID;
  return filterExamples(filter, gen);
}
