#include "vars.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "classify.hpp"
#include "learn.hpp"
#include "basstat.hpp"
#include "table.hpp"

#include "imputation.ppp"


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


TImputer_defaults::TImputer_defaults(PDomain domain)
: defaults(mlnew TExample(domain))
{}


TImputer_defaults::TImputer_defaults(const TExample &valu)
: defaults(mlnew TExample(valu))
{}


TExample *TImputer_defaults::operator()(TExample &example)
{
  checkProperty(defaults);

  if (example.domain != defaults->domain)
    raiseError("invalid domain");

  TExample *imputed = CLONE(TExample, &example);
  try {
    TExample::const_iterator ei(defaults->begin());
    TExample::iterator oi(imputed->begin()), oe(imputed->end());
    for(; oi!=oe; oi++, ei++)
      if ((*oi).isSpecial() && !(*ei).isSpecial())
        *oi = *ei;
  }
  catch (...) {
    mldelete imputed;
    throw;
  }

  return imputed;
};


TExample *TImputer_asValue::operator()(TExample &example)
{ 
  checkProperty(domain);
  return mlnew TExample(domain, example);
}


TExample *TImputer_model::operator ()(TExample &example)
{
  if (models->size() != example.domain->variables->size())
    raiseError("wrong domain (invalid size)");

  TExample *imputed = CLONE(TExample, &example);

  try {
    TExample::iterator ei(imputed->begin()), eie(imputed->end());
    TClassifierList::iterator mi(models->begin());
    TVarList::const_iterator di(example.domain->variables->begin());
    for(; ei!=eie; ei++, mi++, di++) {
      if ((*ei).isSpecial() && *mi) {
        if ((*mi)->classVar != *di)
          raiseError("wrong domain (wrong model for '%s')", (*di)->name.c_str());
        *ei = (*mi)->call(example);
      }
    }
  }
  catch (...) {
    mldelete imputed;
    throw;
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
      *vi = (*doi)->DK();

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
