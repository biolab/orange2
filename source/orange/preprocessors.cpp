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
#include "measures.hpp"
#include "classify.hpp"
#include "learn.hpp"


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



CONSTRUCTOR(drop)
: vnames(string2atoms(parameters))
{}


OPERATOR(drop)
{ TValueFilter *dropfilter = mlnew TValueFilter(true, true, generators.back()->domain);
  dropfilter->decode(vnames.getReference());
  addFilterAdapter(dropfilter, generators);
}


DIRECT_OPERATOR(drop)
{ TValueFilter *dropfilter=mlnew TValueFilter(true, true, generator->domain);
  dropfilter->decode(vnames.getReference());
  return filterExamples(dropfilter, generator);
}
  

CONSTRUCTOR(take)
: vnames(string2atoms(parameters))
{}


OPERATOR(take)
{ TValueFilter *takefilter = mlnew TValueFilter(true, false, generators.back()->domain);
  takefilter->decode(vnames.getReference());
  addFilterAdapter(takefilter, generators);
}


DIRECT_OPERATOR(take) {
  TValueFilter *takefilter = mlnew TValueFilter(true, true, generator->domain);
  takefilter->decode(vnames.getReference());
  return filterExamples(takefilter, generator);
}



CONSTRUCTOR(ignore)
: vnames(string2atoms(parameters))
{}


OPERATOR(ignore)
{ PDomain outDomain = CLONE(TDomain, generators.back()->domain);
  PITERATE(vector<string>, ni, vnames) {
    int vn = outDomain->getVarNum(*ni, false);
    if (vn<0)
      raiseError("attribute '%s' not found", (*ni).c_str());
    else
      outDomain->delVariable(outDomain->operator[](*ni));
  }
  storeToTable(generators, outDomain);
}


DIRECT_OPERATOR(ignore) {
  PDomain outDomain = CLONE(TDomain, generator->domain);
  PITERATE(vector<string>, ni, vnames) {
    int vn = outDomain->getVarNum(*ni, false);
    if (vn<0)
      raiseError("attribute '%s' not found", (*ni).c_str());
    else
      outDomain->delVariable(outDomain->operator[](*ni));
  }
  return PExampleGenerator(mlnew TExampleTable(outDomain, generator));
}



CONSTRUCTOR(select)
: vnames(string2atoms(parameters))
{}


OPERATOR(select)
{ PDomain outDomain = CLONE(TDomain, generators.back()->domain);
  PITERATE(TVarList, vi, generators.back()->domain->attributes)
    if (find(vnames->begin(), vnames->end(), (*vi)->name)==vnames->end())
      outDomain->delVariable(*vi);   
  storeToTable(generators, outDomain);
}


DIRECT_OPERATOR(select)
{ PDomain outDomain = CLONE(TDomain, generator->domain);
  PITERATE(TVarList, vi, generator->domain->attributes)
    if (find(vnames->begin(), vnames->end(), (*vi)->name)==vnames->end())
      outDomain->delVariable(*vi);   
  return PExampleGenerator(mlnew TExampleTable(outDomain, generator));
}



CONSTRUCTOR(remove_duplicates)
{}


OPERATOR(remove_duplicates)
{ storeToTable(generators);
  generators.back().AS(TExampleTable)->removeDuplicates();
}


DIRECT_OPERATOR(remove_duplicates)
{ PExampleGenerator table = mlnew TExampleTable(generator);
  table.AS(TExampleTable)->removeDuplicates();
  return table;
}



CONSTRUCTOR(skip_missing)         
{}


OPERATOR(skip_missing)
{ addFilterAdapter(mlnew TFilter_hasSpecial(true), generators); }


DIRECT_OPERATOR(skip_missing)
{ TFilter_hasSpecial fhs(true);
  return filterExamples(PFilter(fhs), generator);
}



CONSTRUCTOR(only_missing)
{}


OPERATOR(only_missing)
{ addFilterAdapter(mlnew TFilter_hasSpecial(false), generators); }


DIRECT_OPERATOR(only_missing)
{ TFilter_hasSpecial fhs(false);
  return filterExamples(PFilter(fhs), generator); 
}



CONSTRUCTOR(skip_missing_classes)
{}


OPERATOR(skip_missing_classes)
{ addFilterAdapter(mlnew TFilter_hasClassValue(false), generators); }


DIRECT_OPERATOR(skip_missing_classes)
{ TFilter_hasClassValue fhcv(false);
  return filterExamples(PFilter(fhcv), generator); 
}



CONSTRUCTOR(only_missing_classes)
{}


OPERATOR(only_missing_classes)
{ addFilterAdapter(mlnew TFilter_hasClassValue(true), generators); }


DIRECT_OPERATOR(only_missing_classes)
{ TFilter_hasClassValue fhcv(true);
  return filterExamples(PFilter(fhcv), generator); 
}



void varProbabilities(const TIdList &args, PDomain domain, vector<float> &probs)
{
  float overall = -2;
  probs = vector<float>(domain->variables->size(), -2);

  const_ITERATE(TIdList, ai, args) {
    string::const_iterator aii((*ai).begin());
    for(; (aii!=(*ai).end()) && (*aii!='='); aii++);
    if (aii==(*ai).end()) {
      if (overall>0)
        raiseError("varProbabilities: invalid probabilities (%s).", (*ai).c_str());
      char *ep;
      overall = float(strtod((*ai).c_str(), &ep));
      if (*ep || (overall>1) || (overall<-1))
        raiseError("varProbabilities: invalid probability (%s)", (*ai).c_str());
    }
    else {
      string varName=string((*ai).begin(), aii);
      int vnum = domain->getVarNum(varName, false);
      if (vnum<0)
        raiseError("varProbabilities: attribute '%s' does not exist", varName.c_str());
      else {
        if (probs[vnum]>=0)
          raiseError("varProbabilities: probability for '%s' defined more than once", varName.c_str());

        char *ep;
        string pstr(string(aii+1, (*ai).end()));
        float p = float(strtod(pstr.c_str(), &ep));
        if (*ep || (p>1) || (p<-1))
          raiseError("varProbabilities: invalid probability for '%s'", varName.c_str());
        else probs[vnum] = p;
      }
    }
  }

  if (overall<0)
    overall=0;
  vector<float>::iterator pi(probs.begin());
  for(vector<float>::iterator pe(probs.end()-1); pi!=pe; pi++)
    if (*pi==-2)
      *pi=overall;
  if (*pi==-2)
    *pi=0;
}


void varProbabilities(const string &parameters, vector<float> &probabilities, PDomain domain)
{
  vector<string> probstrs;
  string2atoms(parameters, probstrs);
  varProbabilities(probstrs, domain, probabilities);
}


void decodeProbabilities(const string &parameters, PNameProb probabilities, float &overall)
{ vector<string> probstrs;
  string2atoms(parameters, probstrs);
  
  overall = -2;

  const_ITERATE(vector<string>, ai, probstrs) {
    string::const_iterator aii((*ai).begin());
    for(; (aii!=(*ai).end()) && (*aii!='='); aii++);
    string varName = string((*ai).begin(), aii);
    if (!varName.length() | (varName=="overall")) {
      // no name, setting overall probability
      if (overall>0)
        raiseError("decodeProbabilities: invalid probabilities (%s).", (*ai).c_str());
      char *ep;
      overall = float(strtod((*ai).c_str(), &ep));
      if (*ep || (overall>1) || (overall<-1))
        raiseError("decodeProbabilities: invalid overall probability (%s)", (*ai).c_str());
    }
    else {
      if (probabilities->find(varName)!=probabilities->end())
        raiseError("decodeProbabilities: probability for '%s' defined more than once", varName.c_str());

      char *ep;
      string pstr(string(aii+1, (*ai).end()));
      float p = float(strtod(pstr.c_str(), &ep));
      if (*ep || (p>1) || (p<-1))
        raiseError("decodeProbabilities: invalid probability for '%s'", varName.c_str());
      probabilities->operator[](varName)=p;
    }
  }

  if (overall==-2)
    overall=0;
}


void decodeDeviations(const string &parameters, PNameProb deviations, float &overall)
{ vector<string> probstrs;
  string2atoms(parameters, probstrs);
  
  overall = -2;

  const_ITERATE(vector<string>, ai, probstrs) {
    string::const_iterator aii((*ai).begin());
    for(; (aii!=(*ai).end()) && (*aii!='='); aii++);
    string varName = string((*ai).begin(), aii);
    if (!varName.length() | (varName=="overall")) {
      // no name, setting overall probability
      if (overall>0)
        raiseError("decodeDeviations: invalid deviation (%s).", (*ai).c_str());
      char *ep;
      overall = float(strtod((*ai).c_str(), &ep));
      if (*ep || (overall>1) || (overall<-1))
        raiseError("decodeDeviations: invalid overall deviation (%s)", (*ai).c_str());
    }
    else {
      if (deviations->find(varName)!=deviations->end())
        raiseError("decodeDeviations: deviation for '%s' defined more than once", varName.c_str());

      char *ep;
      string pstr(string(aii+1, (*ai).end()));
      float p = float(strtod(pstr.c_str(), &ep));
      if (*ep || (p<0.0))
        raiseError("decodeDeviations: invalid deviation for '%s'", varName.c_str());
      deviations->operator[](varName) = p;
    }
  }

  if (overall==-2)
    overall=0;
}


void varProbabilities(const PNameProb nameprobs, float &overall,
                      PDomain domain, vector<float> &probs)
{ 
  probs = vector<float>(domain->variables->size(), -2);
  const_PITERATE(TNameProb, ni, nameprobs) {
    int vnum = domain->getVarNum((*ni).first, false);
    if (vnum<0)
      raiseError("varProbabilities: variable '%s' does not exist", (*ni).first.c_str());
    else if (probs[vnum]>=0)
      raiseError("varProbabilities: probability for '%s' defined more than once", (*ni).first.c_str());
    else
      probs[vnum]=(*ni).second;
  }

  for(vector<float>::iterator pi(probs.begin()), pe(probs.end()-1); pi!=pe; pi++)
    if (*pi==-2)
      *pi = overall;
  if (probs.back()==-2)
    probs.back() = 0.0;
}
    


#define RANDOMGENERATOR (randseed>0) ? PRandomGenerator(mlnew TRandomGenerator(randseed)) \
                                     : PRandomGenerator()
CONSTRUCTOR(noise)
{ decodeProbabilities(parameters, probabilities, defaultNoise); }


void TPreprocessor_noise::addNoise(TExampleTable *table)
{
  vector<float> probs;
  varProbabilities(probabilities, defaultNoise, table->domain, probs);

  int N = table->numberOfExamples();
  
  PRandomGenerator rgen = RANDOMGENERATOR;
  if (!rgen)
    rgen = globalRandom;

  int lastNN = 0, tn = 0;
  {
  ITERATE(vector<float>, pi, probs) {
    if (*pi!=0.0)
      lastNN = tn;
    tn++;
  }
  }

  vector<PRandomIndices> changeWhat;
  vector<float>::iterator pi = probs.begin();
  for(tn=0; tn<=lastNN; tn++, pi++)
    changeWhat.push_back(*pi!=0.0 ? TMakeRandomIndices2(fabs(*pi), TMakeRandomIndices::NOT_STRATIFIED, rgen)(N)
                                  : PRandomIndices());

  N = 0;
  vector<PRandomIndices>::iterator cwi, ewi=changeWhat.end();
  TVarList::iterator vi;
  TExample::iterator iei;
  PEITERATE(ei, table) {
    for(cwi = changeWhat.begin(), pi = probs.begin(), iei = (*ei).begin(), vi = table->domain->variables->begin();
        cwi!=ewi;
        cwi++, pi++, iei++, vi++)
      if (*cwi && !(*cwi)->at(N))
        if ((*iei = (*vi)->randomValue(rgen->randint())).isDC())
          raiseError("attribute '%s' cannot compute random values.", (*vi)->name.c_str());
    N++;
  }
}


OPERATOR(noise)
{ storeToTable(generators);
  addNoise(generators.back().AS(TExampleTable));
}


DIRECT_OPERATOR(noise)
{ PExampleGenerator table = mlnew TExampleTable(generator);
  addNoise(table.AS(TExampleTable));
  return table;
}



CONSTRUCTOR(gaussian_noise)
{ decodeDeviations(parameters, deviations, defaultDeviation); }


OPERATOR(gaussian_noise)
{ vector<float> probs;
  varProbabilities(deviations, defaultDeviation, generators.back()->domain, probs);
  generators.push_back(PExampleGenerator(mlnew TGaussianNoiseGenerator(probs, generators.back(), RANDOMGENERATOR)));
}


DIRECT_OPERATOR   (gaussian_noise) {
  vector<float> probs;
  varProbabilities(deviations, defaultDeviation, generator->domain, probs);
  TGaussianNoiseGenerator gg = TGaussianNoiseGenerator(probs, generator, RANDOMGENERATOR);
  return PExampleGenerator(mlnew TExampleTable(PExampleGenerator(gg)));
}



CONSTRUCTOR(missing)
: probabilities(mlnew map<string, float>())
{ decodeProbabilities(parameters, probabilities, defaultMissing);  }


void TPreprocessor_missing::addMissing(TExampleTable *table)
{ vector<float> probs;
  varProbabilities(probabilities, defaultMissing, table->domain, probs);

  int N = table->numberOfExamples();
  
  PRandomGenerator rgen=RANDOMGENERATOR;
  if (!rgen)
    rgen = globalRandom;

  int lastNN = 0, tn = 0;
  {
  ITERATE(vector<float>, pi, probs) {
    if (*pi!=0.0)
      lastNN = tn;
    tn++;
  }}

  vector<float>::iterator pi;
  vector<PRandomIndices> changeWhat;
  pi = probs.begin();
  for(tn = 0; tn<=lastNN; tn++, pi++)
    changeWhat.push_back(*pi!=0.0 ? TMakeRandomIndices2(fabs(*pi), TMakeRandomIndices::NOT_STRATIFIED, rgen)(N)
                                 : PRandomIndices());

  N = 0;
  vector<PRandomIndices>::iterator cwi, ewi = changeWhat.end();
  TExample::iterator iei;
  PEITERATE(ei, table) {
    for(cwi = changeWhat.begin(), pi = probs.begin(), iei = (*ei).begin(); cwi!=ewi; cwi++, pi++, iei++)
      if (*cwi && !(*cwi)->at(N))
        if (*pi<0)
          (*iei).setDC();
        else
          (*iei).setDK();
    N++;
  }
}


OPERATOR(missing) {
  storeToTable(generators);
  addMissing(generators.back().AS(TExampleTable));
}


DIRECT_OPERATOR(missing) {
  PExampleGenerator table = mlnew TExampleTable(generator);
  addMissing(table.AS(TExampleTable));
  return table;
}



CONSTRUCTOR(class_noise) {
  vector<string> pstr;
  string2atoms(parameters, pstr);
  if (pstr.size()>1)
    raiseError("invalid number of parameters");
  else if (pstr.size()==1) {
    char *ep;
    classNoise = float(strtod(pstr.front().c_str(), &ep));
    if (*ep || (classNoise<-1) || (classNoise>1))
      raiseError("invalid probability");
  }
  else
    classNoise = 0;
}


void TPreprocessor_class_noise::addNoise(PExampleGenerator table)
{
  PRandomGenerator rgen = RANDOMGENERATOR;
  if (!rgen)
    rgen = globalRandom;

  PRandomIndices rindi = TMakeRandomIndices2(classNoise, TMakeRandomIndices::STRATIFIED_IF_POSSIBLE, rgen)(table);

  TFoldIndices::const_iterator cwi(rindi->begin());
  PVariable classVar=table->domain->classVar;
  PEITERATE(ei, table)
    if (!*(cwi++)) {
      (*ei).setClass(classVar->randomValue(rgen->randint()));
      if ((*ei).getClass().isDC())
        raiseError("attribute '%s' cannot give randomValues.", classVar->name.c_str());
    }
}


OPERATOR(class_noise)
{ storeToTable(generators);
  addNoise(generators.back());
}

DIRECT_OPERATOR(class_noise)
{
  PExampleGenerator table = mlnew TExampleTable(generator);
  addNoise(table);
  return table;
}


CONSTRUCTOR(class_gaussian_noise) {
  vector<string> pstr;
  string2atoms(parameters, pstr);
  if (pstr.size()>1)
    raiseError("invalid number of arguments");
  else if (pstr.size()==1) {
    char *ep;
    classDeviation = float(strtod(pstr.front().c_str(), &ep));
    if (*ep || (classDeviation<0.0))
      raiseError("invalid deviation");
  }
  else
    classDeviation = 0.0;
}


OPERATOR(class_gaussian_noise)
{ vector<float> probabilities(generators.back()->domain->variables->size(), 0);
  probabilities.back() = classDeviation;
  generators.push_back(mlnew TGaussianNoiseGenerator(probabilities, generators.back(), RANDOMGENERATOR));
}


DIRECT_OPERATOR(class_gaussian_noise)
{ vector<float> probabilities(generator->domain->variables->size(), 0);
  probabilities.back() = classDeviation;
  TGaussianNoiseGenerator gngen(probabilities, generator, RANDOMGENERATOR);
  return PExampleGenerator(mlnew TExampleTable(PExampleGenerator(gngen)));
}



CONSTRUCTOR(class_missing)
{ vector<string> pstr;
  string2atoms(parameters, pstr);
  if (pstr.size()>1)
    raiseError("invalid number of arguments");
  else if (pstr.size()==1) {
    char *ep;
    classMissing = float(strtod(pstr.front().c_str(), &ep));
    if (*ep || (classMissing<0) || (classMissing>1))
      raiseError("invalid probability");
  }
  else
    classMissing=0;
}


void TPreprocessor_class_missing::addMissing(PExampleGenerator table)
{
  bool insertDC = (classMissing<0);

  PRandomIndices rindi = TMakeRandomIndices2(fabs(classMissing), TMakeRandomIndices::STRATIFIED_IF_POSSIBLE, RANDOMGENERATOR)(table);

  TFoldIndices::const_iterator cwi(rindi->begin());
  PVariable classVar = table->domain->classVar;
  PEITERATE(ei, table)
    if (!*(cwi++))
      (*ei).setClass(insertDC ? classVar->DC() : classVar->DK());
}


OPERATOR(class_missing)
{ storeToTable(generators);
  addMissing(generators.back());
}


DIRECT_OPERATOR(class_missing)
{ PExampleGenerator table = mlnew TExampleTable(generator);
  addMissing(generator);
  return generator;
}


void multiplyCostVectors(PFloatList dest, const TFloatList &mult)
{
  if (!dest && mult.size()) 
    dest = mlnew TFloatList;

  TFloatList::iterator       ci1(dest->begin()), ce1(dest->end());
  TFloatList::const_iterator ci2(mult.begin()),  ce2(mult.end());
  for( ; (ci1!=ce1) && (ci2!=ce2); *(ci1++) *= *(ci2++));
  while(ci2!=ce2)
    dest->push_back(*(ci2++));
}


CONSTRUCTOR(cost_weight)
: equalize(false)
{}



OPERATOR(cost_weight) {
  if (!equalize && !classWeights->size()) return;

  storeToTable(generators);

  int newei = getMetaID();

  if (equalize) {
    PDistribution dist(getClassDistribution(generators.back(), weightID));
    const TDiscDistribution &ddist = CAST_TO_DISCDISTRIBUTION(dist);
    float N = ddist.abs;
    int nocl = generators.back()->domain->classVar->noOfValues();
    TDiscDistribution dv;
    const_ITERATE(TDiscDistribution, di, ddist)
      dv.push_back(N / nocl / *di);

    PEITERATE(ei, generators.back())
      (*ei).meta.setValue(newei, TValue(WEIGHT(*ei) * dv[(*ei).getClass().intV]));

    weightID=newei;
  }

  if (classWeights && classWeights->size()) {
    const TFloatList &weights = classWeights.getReference();
    PEITERATE(ei, generators.back())
      (*ei).meta.setValue(newei, TValue(WEIGHT(*ei) * weights[(*ei).getClass().intV]));

    weightID=newei;
  }
}



DIRECT_OPERATOR(cost_weight)
{
  if (!equalize && !classWeights->size()) 
    return generator;

  PExampleGenerator table = mlnew TExampleTable(generator);

  if (equalize) {
    PDistribution dist(getClassDistribution(table, weightID));
    const TDiscDistribution &ddist = CAST_TO_DISCDISTRIBUTION(dist);
    float N = ddist.abs;
    int nocl = table->domain->classVar->noOfValues();
    TDiscDistribution dv;
    const_ITERATE(TDiscDistribution, di, ddist)
      dv.push_back(N / nocl / *di);

    PEITERATE(ei, table)
      (*ei).meta.setValue(weightID, TValue(WEIGHT(*ei) * dv[(*ei).getClass().intV]));
  }

  if (classWeights && classWeights->size()) {
    const TFloatList &weights = classWeights.getReference();
    PEITERATE(ei, table)
      (*ei).meta.setValue(weightID, TValue(WEIGHT(*ei) * weights[(*ei).getClass().intV]));
  }

  return table;
}


    
CONSTRUCTOR(censor_weight)
: maxTime(-1),
  weightName("surv_weight")
{}


OPERATOR(censor_weight) {
  if (! (eventValue.size() && timeVar.size() ) )
    raiseError("'eventValue' and/or 'timeVar' not set");

  storeToTable(generators);

  /* Prepare parameters such as outComeIndex, timeIndex and eventVal */

  int outcomeIndex, timeIndex;
  TValue eventVal;

  PDomain domain = generators.back()->domain;

  timeIndex = domain->getMetaNum(timeVar);
  outcomeIndex = outcomeVar.length() ? domain->getVarNum(outcomeVar) : domain->attributes->size();
  domain->variables->at(outcomeIndex)->str2val(eventValue, eventVal);

  /* Construct an appropriate sweightvar and getWeight (and km and p_max, if method=="split") */

  PVariable sweightvar = mlnew TFloatVariable(weightName);
  TKaplanMeier *km = NULL;
  PClassifier getWeight;
  float p_max = -1.0;

  if ((method=="km") || (method=="nmr") || (method=="split")) {
    km = mlnew TKaplanMeier(generators.back(), outcomeIndex, eventVal.intV, timeIndex, weightID);

    if (method!="split") {
      if (method=="km")
        km->toFailure();
      else
        km->toLog();
      km->normalizedCut(maxTime);
    }
    else 
      p_max = (*km)(maxTime);

    getWeight = mlnew TClassifierForKMWeight(sweightvar, km, timeIndex, domain->variables->at(outcomeIndex), eventVal.intV);
  }
  else if (method=="linear") {
    float thisMaxTime = maxTime;
    if (thisMaxTime<=0.0)
      PEITERATE(ei, generators.back()) {
        float tw = (*ei).meta[timeIndex];
        if (tw>thisMaxTime)
          thisMaxTime=tw;
      }

    if (thisMaxTime<=0.0)
      raiseError("invalid time values (max<=0)");

    getWeight = mlnew TClassifierForLinearWeight(sweightvar, thisMaxTime, timeIndex, domain->variables->at(outcomeIndex), eventVal.intV);
  }
  
  else
    raiseError("unknown method (%s)", method.c_str());


  /* Prepare a new domain with a weighted meta-variable */

  PDomain newDomain = CLONE(TDomain, domain);

  int newWeightID = getMetaID();
  newDomain->metas.push_back(TMetaDescriptor(newWeightID, sweightvar));
  weightID = newWeightID;

  /* Prepare the weighted example table */

  if (method=="split") {
    TExampleTable *newTable = mlnew TExampleTable(newDomain);
    PExampleGenerator wnewTable = PExampleGenerator(newTable);
    int clss = domain->classVar->noOfValues();
    { 
      PEITERATE(ei, generators.back()) {
        if ((*ei).getClass().intV!=eventVal.intV) {
          float p_surv = p_max/getWeight->operator()(*ei).floatV;
          if (p_surv>=1.0) {
            TExample newExample(newDomain, *ei);
            newExample.meta.setValue(weightID, TValue(1.0F));
            newTable->addExample(newExample);
          }
          else {
            for(int clv = 0; clv<clss; clv++) {
              TExample newExample(newDomain, *ei);
              newExample.setClass(TValue(clv));
              newExample.meta.setValue(weightID, TValue(clv==eventVal.intV ? 1-p_surv : p_surv));
              newTable->addExample(newExample);
            }
          }
        }
        else {
          TExample newExample(newDomain, *ei);
          newExample.meta.setValue(weightID, TValue(1.0F));
          newTable->addExample(newExample);
        }
      }
    }
    replaceWithTable(generators, wnewTable);
  }

  else {
    sweightvar->getValueFrom=getWeight;
    generators.back().AS(TExampleTable)->changeDomain(newDomain);
  }
}

DIRECT_OPERATOR(censor_weight)
{ raiseError("not implemented"); 
  throw 0;
}


CONSTRUCTOR(discretize) 
: noOfIntervals(4),
  vnames(string2atoms(parameters)),
  notClass(true)
{}


PDomain TPreprocessor_discretize::discretizedDomain(PExampleGenerator generator, long &weightID)
{
  PDomain oldDomain = generator->domain;

  vector<int> discretizeId;
  if (vnames && !vnames->empty())
    PITERATE(vector<string>, ni, vnames) {
      int vn = oldDomain->getVarNum(*ni, false);
      if (vn<0)
        raiseError("attribute '%s' not found", (*ni).c_str());
      else
        discretizeId.push_back(vn);
    }
  else
    for(TVarList::iterator bi(oldDomain->variables->begin()),
                           ei(oldDomain->variables->end()-((notClass && oldDomain->classVar) ? 1 : 0)),
                           vi(bi);
        vi!=ei; vi++)
      if ((*vi)->varType==TValue::FLOATVAR) discretizeId.push_back(vi-bi);
    
  return PDomain(mlnew TDiscretizedDomain(generator, discretizeId, weightID, noOfIntervals, method));
}


OPERATOR(discretize)
{
  storeToTable(generators);
  storeToTable(generators, discretizedDomain(generators.back(), weightID));
}


DIRECT_OPERATOR(discretize)
{ return PExampleGenerator(mlnew TExampleTable(discretizedDomain(generator, weightID), generator)); }



int prop_min_max(const float &prop, const int &min, const int &max, const int &available)
{ int ret;

  if ((prop==-1) && (max==-1))
    ret = available>30 ? 10 : (available+2)/3;
  else {
    ret = available;
    if (prop>=0)
      ret=int(ret*prop);
    if ((max>=0) && (ret>max))
        ret=max;
  }

  if (ret<min)
    ret=min;

  return (ret>available) ? available : ret;
}


CONSTRUCTOR(move_to_table)
{}


OPERATOR(move_to_table)
{ storeToTable(generators); }


DIRECT_OPERATOR(move_to_table)
{ raiseError("not implemented"); 
  throw 0;
}


CONSTRUCTOR(filter)
{}


OPERATOR(filter)
{ if (filter) 
    addFilterAdapter(filter, generators);
}


DIRECT_OPERATOR(filter)
{ raiseError("not implemented"); 
  throw 0;
}
