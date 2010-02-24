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


#include "vars.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "classify.hpp"
#include "learn.hpp"
#include "basstat.hpp"
#include "table.hpp"
#include "lookup.hpp"
#include "classfromvar.hpp"

#include "imputation.ppp"

WRAPPER(Classifier)

void TTransformValue_IsDefined::transform(TValue &val)
{
  val = TValue(val.isSpecial() ? 1 : 0);
}

PExampleGenerator TImputer::operator()(PExampleGenerator gen, const int &weightID)
{
  if (!gen)
    return PExampleGenerator();
  if (!gen->numberOfExamples())
    return mlnew TExampleTable(gen->domain);

  TExample *imputedExample = call(*gen->begin());
  TExampleTable *newtable = mlnew TExampleTable(imputedExample->domain);
  PExampleGenerator newgen = newtable;
  mldelete imputedExample;

  PEITERATE(ei, gen)
    newtable->addExample(call(*ei));

  return newgen;
}


void TImputer::imputeDefaults(TExample *example, PExample defaults)
{ 
  if (example->domain != defaults->domain)
    raiseError("invalid domain");

  try {
    TExample::const_iterator ei(defaults->begin());
    TExample::iterator oi(example->begin()), oe(example->end());
    for(; oi!=oe; oi++, ei++)
      if ((*oi).isSpecial() && !(*ei).isSpecial())
        *oi = *ei;
  }
  catch (...) {
    mldelete example;
    throw;
  }
}

TImputer_defaults::TImputer_defaults(PDomain domain)
: defaults(mlnew TExample(domain))
{}


TImputer_defaults::TImputer_defaults(PExample example)
: defaults(example)
{}


TImputer_defaults::TImputer_defaults(const TExample &valu)
: defaults(mlnew TExample(valu))
{}


TExample *TImputer_defaults::operator()(TExample &example)
{
  checkProperty(defaults);
  TExample *imputed = CLONE(TExample, &example);
  imputeDefaults(imputed, defaults);
  return imputed;
};


TExample *TImputer_asValue::operator()(TExample &example)
{ 
  checkProperty(domain);
  TExample *imputed = mlnew TExample(domain, example);
  if (defaults)
    imputeDefaults(imputed, defaults);
  return imputed;
}


TExample *TImputer_model::operator ()(TExample &example)
{
  checkProperty(models);

  if (models->size() != example.domain->variables->size())
    raiseError("wrong domain (invalid size)");

  TExample *imputed = CLONE(TExample, &example);

  try {
    TExample::iterator ei(imputed->begin()), eie(imputed->end());
    TClassifierList::iterator mi(models->begin()), me(models->end());
    TVarList::const_iterator di(example.domain->variables->begin());
    for(; (ei!=eie) && (mi!=me); ei++, mi++, di++) {
      if ((*ei).isSpecial() && *mi) {
        if ((*mi)->classVar) {
          if ((*mi)->classVar != *di)
            raiseError("wrong domain (wrong model for '%s')", (*di)->name.c_str());
          *ei = (*mi)->call(example);
        }
        else {
          TValue val = (*mi)->call(example);
          if (val.varType != (*di)->varType)
            raiseError("wrong domain (wrong model for '%s')", (*di)->name.c_str());
          *ei = val;
        }
      }
    }
  }
  catch (...) {
    mldelete imputed;
    throw;
  }
  return imputed;
}



TImputer_random::TImputer_random(const bool ic, const bool dete, PDistributionList dist)
: imputeClass(ic),
  deterministic(dete),
  distributions(dist)
{}

TExample *TImputer_random::operator()(TExample &example)
{
  TExample *imputed = CLONE(TExample, &example);

  bool initialized = !deterministic; // if deterministic, randgen is initialized with crc32 for each exapmle
  TVarList::iterator vi(imputed->domain->variables->begin()), ve(imputed->domain->variables->end());
  if (vi==ve)
    return imputed;
  if (!imputeClass && imputed->domain->classVar) {
    if (vi == --ve)
      return imputed;
  }

  if (!distributions) {
    for(TExample::iterator ei(imputed->begin()); vi!=ve; vi++, ei++)
      if ((*ei).isSpecial()) {
        if (!initialized) {
          randgen.initseed = imputed->sumValues();
          randgen.reset();
          initialized = true;
        }
        *ei = (*vi)->randomValue(randgen.randint());
      }
  }

  else {
    TDistributionList::iterator di(distributions->begin());

    for(TExample::iterator ei(imputed->begin()); vi!=ve; vi++, ei++, di++) {
      if ((*ei).isSpecial()) {
        if (!initialized) {
          randgen.initseed = imputed->sumValues();
          randgen.reset();
          initialized = true;
        }

        if ((*ei).varType == TValue::INTVAR)
          *ei = TValue((*di)->randomInt(randgen.randlong()));
        else
          *ei = TValue((*di)->randomFloat(randgen.randlong()));
      }
    }
  }

  return imputed;
}





TImputerConstructor::TImputerConstructor()
: imputeClass(true)
{}


PImputer TImputerConstructor_defaults::operator()(PExampleGenerator egen, const int &weightID)
{
  return mlnew TImputer_defaults(defaults);
}


PImputer TImputerConstructor_average::operator()(PExampleGenerator egen, const int &weightID)
{
  TImputer_defaults *imputer = mlnew TImputer_defaults(egen->domain);
  PImputer wimputer(imputer);

  TDomainDistributions ddist(egen, weightID);
  TExample::iterator vi(imputer->defaults->begin()), ve(imputer->defaults->end());
  TDomainDistributions::const_iterator di(ddist.begin());
  TVarList::const_iterator doi(egen->domain->variables->begin());
  for(; vi!=ve; vi++, di++, doi++)
    if ((*di)->supportsDiscrete)
      *vi = (*di)->highestProbValue(egen->numberOfExamples());
    else if ((*di)->supportsContinuous)
      *vi = TValue((*di)->percentile(50));
    else
      *vi = TValue((*doi)->DK());

  if (!imputeClass && egen->domain->classVar)
    imputer->defaults->setClass(egen->domain->classVar->DK());
  
  return wimputer;
}


PImputer TImputerConstructor_minimal::operator()(PExampleGenerator egen, const int &weightID)
{
  TImputer_defaults *imputer = mlnew TImputer_defaults(egen->domain);
  PImputer wimputer(imputer);

  TDomainBasicAttrStat basstat(egen, weightID);
  TExample::iterator vi(imputer->defaults->begin()), ve(imputer->defaults->end());
  TDomainBasicAttrStat::const_iterator bi(basstat.begin());
  for(; vi!=ve; vi++, bi++)
    if (*bi)
      *vi = TValue((*bi)->min);
    else
      *vi = TValue(0);

  if (!imputeClass && egen->domain->classVar)
    imputer->defaults->setClass(egen->domain->classVar->DK());
  
  return wimputer;
}


PImputer TImputerConstructor_maximal::operator()(PExampleGenerator egen, const int &weightID)
{
  TImputer_defaults *imputer = mlnew TImputer_defaults(egen->domain);
  PImputer wimputer(imputer);

  TDomainBasicAttrStat basstat(egen, weightID);
  TExample::iterator vi(imputer->defaults->begin()), ve(imputer->defaults->end());
  TDomainBasicAttrStat::const_iterator bi(basstat.begin());
  TVarList::const_iterator di(egen->domain->variables->begin());
  for(; vi!=ve; vi++, bi++, di++)
    if (*bi)
      *vi = TValue((*bi)->max);
    else
      *vi = TValue((*di)->noOfValues()-1);

  if (!imputeClass && egen->domain->classVar)
    imputer->defaults->setClass(egen->domain->classVar->DK());
  
  return wimputer;
}


TTransformValue_IsDefined staticTransform_IsDefined;

PVariable TImputerConstructor_asValue::createImputedVar(PVariable var)
{
  if (var->varType == TValue::INTVAR) {
    TEnumVariable *newvar = mlnew TEnumVariable(var->name);
    PVariable res = newvar;
    newvar->values = mlnew TStringList(var.AS(TEnumVariable)->values.getReference());
    newvar->values->push_back("NA");

    TClassifierByLookupTable1 *cblt = mlnew TClassifierByLookupTable1(newvar, var);
    newvar->getValueFrom = cblt;
    TValueList &table = cblt->lookupTable.getReference();
    for(int i = 0, e = table.size(); i!=e; i++)
      table[i] = TValue(i);

    return res;
  }

  if (var->varType == TValue::FLOATVAR) {
    TEnumVariable *newvar = mlnew TEnumVariable(var->name + "_def");
    PVariable res = newvar;
    newvar->values->push_back("def");
    newvar->values->push_back("undef");

    TClassifierFromVar *cfv = mlnew TClassifierFromVar(newvar, var);
    newvar->getValueFrom = cfv;
    cfv->transformUnknowns = true;

    cfv->transformer = PTransformValue(staticTransform_IsDefined);
    return res;
  }

  return PVariable();
}


PImputer TImputerConstructor_asValue::operator ()(PExampleGenerator egen, const int &weightID)
{
  PDomain &domain = egen->domain;
  if (imputeClass && domain->classVar && domain->classVar->varType == TValue::FLOATVAR)
    raiseError("This method cannot impute continuous classes");

  bool hasContinuous = false;
  TVarList newVariables;
  PITERATE(TVarList, vi, domain->attributes) {
    PVariable newvar = createImputedVar(*vi);
    if (newvar) {
      newVariables.push_back(newvar);
      if ((*vi)->varType == TValue::FLOATVAR) {
        newVariables.push_back(*vi);
        hasContinuous = true;
      }
    }
    else
      newVariables.push_back(*vi);
  }

  PVariable classVar;
  if (domain->classVar) {
    if (imputeClass)
      createImputedVar(domain->classVar);
    if (!classVar)
      classVar = domain->classVar;
  }

  TImputer_asValue *imputer = mlnew TImputer_asValue;
  PImputer wimputer(imputer);
  imputer->domain = mlnew TDomain(classVar, newVariables);

  if (hasContinuous) {
    imputer->defaults = mlnew TExample(imputer->domain);
    TDomainBasicAttrStat basstat(egen, weightID);
    TExample::iterator aei(imputer->defaults->begin());
    ITERATE(TDomainBasicAttrStat, bi, basstat) {
      aei++;
      if (*bi)
        *(aei++) = TValue((*bi)->avg);
    }        
  }

  return wimputer;
}

TImputerConstructor_model::TImputerConstructor_model()
: useClass(false)
{}


PImputer TImputerConstructor_model::operator()(PExampleGenerator egen, const int &weightID)
{
  TImputer_model *imputer = mlnew TImputer_model;
  PImputer wimputer(imputer);
  imputer->models = mlnew TClassifierList;

  TVarList vl = egen->domain->variables.getReference();
  if (!useClass && egen->domain->classVar)
    vl.erase(vl.end()-1);
  PVariable tmp;

  ITERATE(TVarList, vli, vl) {
    const int varType = (*vli)->varType;
    if (   (varType == TValue::INTVAR) && learnerDiscrete
        || (varType == TValue::FLOATVAR) && learnerContinuous) {
      tmp = *vli; *vli = vl.back(); vl.back() = tmp;
      PDomain newdomain = mlnew TDomain(vl);
      PExampleGenerator newgen = mlnew TExampleTable(newdomain, egen);
      imputer->models->push_back((varType == TValue::INTVAR ? learnerDiscrete : learnerContinuous)->call(newgen, weightID));
      tmp = *vli; *vli = vl.back(); vl.back() = tmp;
    }
    else
      imputer->models->push_back(PClassifier());
  }

  if (egen->domain->classVar) {
    const int varType = egen->domain->classVar->varType;
    if (imputeClass &&
       (   (varType == TValue::INTVAR) && learnerDiscrete
        || (varType == TValue::FLOATVAR) && learnerContinuous))
      imputer->models->push_back((varType == TValue::INTVAR ? learnerDiscrete : learnerContinuous)->call(egen, weightID));
    else
      imputer->models->push_back(PClassifier());
  }

  return wimputer;
}



TImputerConstructor_random::TImputerConstructor_random(const bool dete)
: deterministic(dete)
{}


PImputer TImputerConstructor_random::operator()(PExampleGenerator egen, const int &weightID)
{
  PDomainBasicAttrStat dbas;
  TDomainBasicAttrStat::const_iterator dbi;
  if (egen->domain->hasContinuousAttributes(true)) {
    dbas = new TDomainBasicAttrStat(egen, weightID);
    dbi = dbas->begin();
  }

  
  PDomainDistributions ddist;
  TDomainDistributions::const_iterator ddi;
  if (egen->domain->hasDiscreteAttributes(true)) {
    ddist = new TDomainDistributions(egen, weightID, false, true);
    ddi = ddist->begin();
  }

  PDistributionList distributions = new TDistributionList();

  PITERATE(TVarList, vi, egen->domain->variables) {
    if ((*vi)->varType == TValue::INTVAR)
      distributions->push_back(*ddi);
    else
      distributions->push_back(new TGaussianDistribution((*dbi)->avg, (*dbi)->dev));
    if (dbas)
      dbi++;
    if (ddist)
      ddi++;
  }

  return mlnew TImputer_random(imputeClass, deterministic, distributions);
}
