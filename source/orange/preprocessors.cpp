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
#include "learn.hpp"

#include <string>
#include "preprocessors.ppp"

DEFINE_TOrangeMap_classDescription(TOrangeMap_KV, PVariable, PValueFilter, "VariableFilterMap")
DEFINE_TOrangeMap_classDescription(TOrangeMap_K, PVariable, float, "VariableFloatMap")

#ifdef _MSC_VER
  #pragma warning (disable : 4100) // unreferenced local parameter (macros name all arguments)
#endif


PExampleGenerator TPreprocessor::filterExamples(PFilter filter, PExampleGenerator generator)
{ TFilteredGenerator fg(filter, generator);
  return PExampleGenerator(mlnew TExampleTable(PExampleGenerator(fg))); 
}


PBoolList TPreprocessor::filterSelectionVector(PFilter filter, PExampleGenerator generator)
{
  TBoolList *selection = new TBoolList;
  PBoolList pselection = selection;

  const int nex = generator->numberOfExamples();
  if (nex > 0)
    selection->reserve(nex);

  TFilter &filt = filter.getReference();
  PEITERATE(ei, generator)
    selection->push_back(filt(*ei));

  return pselection;
}


PBoolList TPreprocessor::selectionVector(PExampleGenerator, const int &)
{ 
  raiseError("this class doesn't support method 'selectionVector'");
  return NULL;
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
        raiseError("attribute '%s' not found", (*vi)->get_name().c_str());

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




PFilter TPreprocessor_take::constructFilter(PVariableFilterMap values, PDomain dom, bool conj, bool negate)
{ 
  TValueFilterList *dropvalues = mlnew TValueFilterList();
  PValueFilterList wdropvalues = dropvalues;
  const TDomain &domain = dom.getReference();
  PITERATE(TVariableFilterMap, vi, values) {
    TValueFilter *vf = CLONE(TValueFilter, (*vi).second);
    dropvalues->push_back(vf); // this wraps it!
    vf->position = domain.getVarNum((*vi).first);
  }

  return mlnew TFilter_values(wdropvalues, conj, negate, dom);
}



TPreprocessor_take::TPreprocessor_take()
: values(mlnew TVariableFilterMap()),
  conjunction(true)
{}


TPreprocessor_take::TPreprocessor_take(PVariableFilterMap avalues, bool aconj)
: values(avalues),
  conjunction(aconj)
{}


PExampleGenerator TPreprocessor_take::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  newWeight = weightID;
  return filterExamples(constructFilter(values, gen->domain, conjunction, false), gen);
}


PBoolList TPreprocessor_take::selectionVector(PExampleGenerator gen, const int &)
{ 
  return filterSelectionVector(constructFilter(values, gen->domain, conjunction, false), gen);
}



TPreprocessor_drop::TPreprocessor_drop()
: values(mlnew TVariableFilterMap()),
  conjunction(true)
{}


TPreprocessor_drop::TPreprocessor_drop(PVariableFilterMap avalues, bool aconj)
: values(avalues),
  conjunction(aconj)
{}


PExampleGenerator TPreprocessor_drop::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  newWeight = weightID;
  return filterExamples(TPreprocessor_take::constructFilter(values, gen->domain, conjunction, true), gen);
}

  
PBoolList TPreprocessor_drop::selectionVector(PExampleGenerator gen, const int &)
{ 
  return filterSelectionVector(TPreprocessor_take::constructFilter(values, gen->domain, conjunction, true), gen);
}




PExampleGenerator TPreprocessor_removeDuplicates::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ PExampleGenerator table = mlnew TExampleTable(gen);

  if (weightID)
    newWeight = weightID;
  else {
    newWeight = getMetaID();
    table.AS(TExampleTable)->addMetaAttribute(newWeight, TValue(float(1.0)));
  }

  table.AS(TExampleTable)->removeDuplicates(newWeight);
  return table;
}



PExampleGenerator TPreprocessor_dropMissing::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ newWeight = weightID;
  return filterExamples(mlnew TFilter_hasSpecial(true), gen);
}


PBoolList TPreprocessor_dropMissing::selectionVector(PExampleGenerator gen, const int &)
{ 
  return filterSelectionVector(mlnew TFilter_hasSpecial(true), gen);
}


PExampleGenerator TPreprocessor_takeMissing::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ newWeight = weightID;
  return filterExamples(mlnew TFilter_hasSpecial(false), gen);
}


PBoolList TPreprocessor_takeMissing::selectionVector(PExampleGenerator gen, const int &)
{ 
  return filterSelectionVector(mlnew TFilter_hasSpecial(false), gen);
}


PExampleGenerator TPreprocessor_dropMissingClasses::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ newWeight = weightID;
  return filterExamples(mlnew TFilter_hasClassValue(false), gen);
}


PBoolList TPreprocessor_dropMissingClasses::selectionVector(PExampleGenerator gen, const int &)
{ 
  return filterSelectionVector(mlnew TFilter_hasClassValue(false), gen);
}


PExampleGenerator TPreprocessor_takeMissingClasses::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ newWeight = weightID;
  return filterExamples(mlnew TFilter_hasClassValue(true), gen);
}


PBoolList TPreprocessor_takeMissingClasses::selectionVector(PExampleGenerator gen, const int &)
{ 
  return filterSelectionVector(mlnew TFilter_hasClassValue(true), gen);
}


TPreprocessor_shuffle::TPreprocessor_shuffle()
: attributes(mlnew TVarList())
{}


TPreprocessor_shuffle::TPreprocessor_shuffle(PVarList attrs)
: attributes(attrs)
{}


PExampleGenerator TPreprocessor_shuffle::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  vector<int> indices;
  PITERATE(TVarList, vi, attributes) {
    const int idx = gen->domain->getVarNum(*vi, false);
    if (idx == ILLEGAL_INT)
      raiseError("attribute '%s' not found", (*vi)->get_name().c_str());
    indices.push_back(idx);
  }
    
  newWeight = weightID;

  TExampleTable *newData = mlnew TExampleTable(gen);
  PExampleGenerator wdata = newData;
  const int tlen = newData->size();
  if (!tlen || !indices.size())
    return wdata;
    
  PRandomGenerator rg = randomGenerator ? randomGenerator : mlnew TRandomGenerator;

  const_ITERATE(vector<int>, ii, indices) {
    for(int i = tlen; --i; )
      swap((*newData)[i][*ii], (*newData)[rg->randint(i)][*ii]);
  }
      
  return wdata;
}



void addNoise(const int &index, const float &proportion, TMakeRandomIndicesN &mri, TExampleTable *table)
{ 
  const int nvals = table->domain->variables->at(index)->noOfValues();
  const int N = table->size();
  const int changed = N*proportion;
  const int cdiv = (changed+(nvals-1)) / nvals;
  mri.p = mlnew TFloatList(nvals, cdiv);
  
  PLongList rind(mri(N));
  TLongList::const_iterator ri(rind->begin());
  PEITERATE(ei, table) {
    if (*ri < nvals)
        (*ei)[index] = TValue(int(*ri));
    ri++;
  }
}


TPreprocessor_addClassNoise::TPreprocessor_addClassNoise(const float &cn)
: proportion(cn)
{}


PExampleGenerator TPreprocessor_addClassNoise::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  if (!gen->domain->classVar)
    raiseError("Class-less domain");
  if (gen->domain->classVar->varType != TValue::INTVAR)
    raiseError("Discrete class value expected");
  if ((proportion<0.0) || (proportion>1.0))
    raiseError("invalid 'proportion'");

  TExampleTable *table = mlnew TExampleTable(gen);
  PExampleGenerator wtable = table;

  if (proportion>0.0) {
    TMakeRandomIndicesN mri;
    mri.randomGenerator = randomGenerator ? randomGenerator : mlnew TRandomGenerator;
    addNoise(table->domain->attributes->size(), proportion, mri, table);
  }

  newWeight = weightID;
  return wtable;
}



TPreprocessor_addNoise::TPreprocessor_addNoise()
: proportions(mlnew TVariableFloatMap()),
  defaultProportion(0.0)
{}


TPreprocessor_addNoise::TPreprocessor_addNoise(PVariableFloatMap probs, const float &defprob)
: proportions(probs),
  defaultProportion(defprob)  
{}



// props should be initialized to length of domain.attributes->size(), with defaultProportions
void getProportions(PVariableFloatMap &proportions, const TDomain &domain, vector<float> &props)
{
  if (proportions) {
    PITERATE(TVariableFloatMap, vi, proportions) {
      const int idx = domain.getVarNum((*vi).first);
      // class is included if this is explicitly requested
      if (idx >= props.size())
        props.push_back((*vi).second);
      else
        props[idx] = (*vi).second;
    }
  }
}



PExampleGenerator TPreprocessor_addNoise::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  newWeight = weightID;

  if (!proportions && (defaultProportion<=0.0))
    return mlnew TExampleTable(gen);

  const TDomain &domain = gen->domain.getReference();

  TExampleTable *table = mlnew TExampleTable(gen);
  PExampleGenerator wtable = table;

  // We mustn't allow MakeRandomIndicesN to initalize a new generator each time it's called since we'd than always select the same examples
  const int n = table->size();
  PRandomGenerator rg = randomGenerator ? randomGenerator : mlnew TRandomGenerator;
  TMakeRandomIndicesN makerind;
  makerind.randomGenerator = rg;

  // this will not assign the defaultProportion to the class
  vector<float> props(domain.attributes->size(), defaultProportion > 0.0 ? defaultProportion : 0.0);
  getProportions(proportions, domain, props);

  int idx = 0;
  vector<float>::const_iterator pi(props.begin()), pe(props.end());
  for(; pi!=pe; pi++, idx++) {
    if (*pi > 0.0) {
      const PVariable &var = domain.variables->at(idx);
      if (var->varType != TValue::INTVAR)
        raiseError("Cannot add noise to non-discrete attribute '%s'", var->get_name().c_str());
      addNoise(idx, *pi, makerind, table);
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


int cmp1st(const pair<int, float> &o1, const pair<int, float> &o2)
{
  return o1.first < o2.first;
}

/* For Gaussian noise we use TGaussianNoiseGenerator; the advantage against going
   attribute by attribute (like in addNoise) is that it might require less paging
   on huge datasets. */
PExampleGenerator TPreprocessor_addGaussianNoise::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  newWeight = weightID;

  if (!deviations && (defaultDeviation<=0.0))
    return mlnew TExampleTable(gen);

  const TDomain &domain = gen->domain.getReference();
  vector<pair<int, float> > ps;
  vector<bool> attributeUsed(domain.attributes->size(), false);
  

  if (deviations)
    PITERATE(TVariableFloatMap, vi, deviations) {
      PVariable var = (*vi).first;
      if (var->varType != TValue::FLOATVAR)
        raiseError("attribute '%s' is not continuous", var->get_name().c_str());

      const int pos = domain.getVarNum(var);
      ps.push_back(pair<int, float>(pos, (*vi).second));

      if ((pos >= 0) && (pos < attributeUsed.size()))
        attributeUsed[pos] = true;
    }
  
  if (defaultDeviation) {
    TVarList::const_iterator vi(domain.attributes->begin());
    const vector<bool>::const_iterator bb = attributeUsed.begin();
    const_ITERATE(vector<bool>, bi, attributeUsed) {
      if (!*bi && ((*vi)->varType == TValue::FLOATVAR))
        ps.push_back(pair<int, float>(bi-bb, defaultDeviation));
      vi++;
    }
  }

  sort(ps.begin(), ps.end(), cmp1st);
  TGaussianNoiseGenerator gg = TGaussianNoiseGenerator(ps, gen, randomGenerator);
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

  TExampleTable *table = mlnew TExampleTable(gen);
  PExampleGenerator wtable = table;

  // We mustn't allow MakeRandomIndices2 to initalize a new generator each time it's called since we'd than always select the same examples
  const int n = table->size();
  TMakeRandomIndices2 makerind;
  makerind.randomGenerator = randomGenerator ? randomGenerator : mlnew TRandomGenerator;;

  // this will not assign the defaultProportion to the class
  vector<float> props(domain.attributes->size(), defaultProportion > 0.0 ? defaultProportion : 0.0);
  getProportions(proportions, domain, props);

  int idx = 0;
  vector<float>::const_iterator pi(props.begin()), pe(props.end());
  for(; pi != pe; idx++, pi++)
    if (*pi > 0.0) {
      PLongList rind = makerind(n, 1 - *pi);
      const unsigned char &varType = domain.variables->at(idx)->varType;
      int eind = 0;
      PITERATE(TLongList, ri, rind) {
        if (*ri)
          (*table)[eind][idx] = TValue(varType, specialType);
        eind++;
      }
    }

  return wtable;
}



TPreprocessor_addGaussianClassNoise::TPreprocessor_addGaussianClassNoise(const float &dev)
: deviation(dev)
{}


PExampleGenerator TPreprocessor_addGaussianClassNoise::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  PVariable classVar = gen->domain->classVar;

  if (!classVar)
    raiseError("Class-less domain");
  if (classVar->varType != TValue::FLOATVAR)
    raiseError("Class '%s' is not continuous", gen->domain->classVar->get_name().c_str());

  newWeight = weightID;

  if (deviation>0.0) {
    vector<pair<int, float> > deviations;
    deviations.push_back(pair<int, float>(gen->domain->attributes->size(), deviation));
    TGaussianNoiseGenerator gngen(deviations, gen, randomGenerator);
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
    mri2.randomGenerator = randomGenerator;
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



TPreprocessor_addClassWeight::TPreprocessor_addClassWeight()
: classWeights(mlnew TFloatList),
  equalize(false)
{}


TPreprocessor_addClassWeight::TPreprocessor_addClassWeight(PFloatList cw, const bool &eq)
: equalize(eq),
  classWeights(cw)
{}


PExampleGenerator TPreprocessor_addClassWeight::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
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

  if (classWeights && classWeights->size() && (classWeights->size() != nocl))
    raiseError("size of classWeights should equal the number of classes");


  vector<float> weights;

  if (equalize) {
    PDistribution dist(getClassDistribution(gen, weightID));
    const TDiscDistribution &ddist = CAST_TO_DISCDISTRIBUTION(dist);
    if (ddist.size() > nocl)
      raiseError("there are out-of-range classes in the data (attribute descriptor has too few values)");

    if (classWeights && classWeights->size()) {
      float tot_w = 0.0;
      TFloatList::const_iterator cwi(classWeights->begin());
      TDiscDistribution::const_iterator di(ddist.begin()), de(ddist.end());
      for(; di!=de; di++, cwi++)
        if (*di > 0.0)
          tot_w += *cwi;

      if (tot_w == 0.0) {
        newWeight = 0;
        return wtable;
      }

      float fact = tot_w * ddist.abs;
      di = ddist.begin();
      PITERATE(TFloatList, wi, classWeights)
        weights.push_back(*wi / *(di++) * fact);
    }

    else { // no class weights, only equalization
      int noNullClasses = 0;
      { const_ITERATE(TDiscDistribution, di, ddist)
          if (*di>0.0)
            noNullClasses++;
      }
      const float N = ddist.abs;
      const_ITERATE(TDiscDistribution, di, ddist)
        if (*di>0.0)
          weights.push_back(N / noNullClasses / *di);
        else
          weights.push_back(1.0);
    }
  }

  else  // no equalization, only weights
    weights = classWeights.getReference();

  newWeight = getMetaID();
  PEITERATE(ei, table)
    (*ei).setMeta(newWeight, TValue(WEIGHT(*ei) * weights[(*ei).getClass().intV]));

  return wtable;
}



PDistribution kaplanMeier(PExampleGenerator gen, const int &outcomeIndex, TValue &failValue, const int &timeIndex, const int &weightID);
PDistribution bayesSurvival(PExampleGenerator gen, const int &outcomeIndex, TValue &failValue, const int &timeIndex, const int &weightID, const float &maxTime);

TPreprocessor_addCensorWeight::TPreprocessor_addCensorWeight()
: outcomeVar(),
  timeVar(),
  eventValue(),
  method(km),
  maxTime(0.0),
  addComplementary(false)
{}


TPreprocessor_addCensorWeight::TPreprocessor_addCensorWeight(PVariable ov, PVariable tv, const TValue &ev, const int &me, const float &mt)
: outcomeVar(ov),
  timeVar(tv),
  eventValue(ev),
  method(me),
  maxTime(0.0),
  addComplementary(false)
{}

void TPreprocessor_addCensorWeight::addExample(TExampleTable *table, const int &weightID, const TExample &example, const float &weight, const int &complementary, const float &compWeight)
{ 
  TExample ex = example;

  ex.setMeta(weightID, TValue(weight));
  table->addExample(ex);

  if ((complementary >= 0) && (compWeight>0.0)) {
    ex.setClass(TValue(complementary));
    ex.setMeta(weightID, TValue(compWeight));
    table->addExample(ex);
  }
}


PExampleGenerator TPreprocessor_addCensorWeight::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  if (eventValue.isSpecial())
    raiseError("'eventValue' not set");

  if (eventValue.varType != TValue::INTVAR)
    raiseError("'eventValue' invalid (discrete value expected)");

  const int failIndex = eventValue.intV;

  int outcomeIndex;
  if (outcomeVar) {
    outcomeIndex = gen->domain->getVarNum(outcomeVar, false);
    if (outcomeIndex==ILLEGAL_INT)
      raiseError("outcomeVar not found in domain");
  }
  else
    if (gen->domain->classVar)
      outcomeIndex = gen->domain->attributes->size();
    else
      raiseError("'outcomeVar' not set and the domain is class-less");

  int complementary = addComplementary ? eventValue.intV : -1;

  checkProperty(timeVar);
  int timeIndex = gen->domain->getVarNum(timeVar, false);
  if (timeIndex==ILLEGAL_INT)
    raiseError("'timeVar' not found in domain");

  TExampleTable *table = mlnew TExampleTable(gen->domain);
  PExampleGenerator wtable = table;

  if (method == linear) {
    float thisMaxTime = maxTime;
    if (thisMaxTime<=0.0)
      PEITERATE(ei, table) {
        const TValue &tme = (*ei)[timeIndex];
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
    PEITERATE(ei, gen) {
      if (!(*ei)[outcomeIndex].isSpecial() && (*ei)[outcomeIndex].intV==failIndex)
        addExample(table, newWeight, *ei, WEIGHT(*ei), complementary);
      else {
        const TValue &tme = (*ei)[timeIndex];
        // need to check it again -- the above check is only run if maxTime is not given
        if (tme.varType != TValue::FLOATVAR)
          raiseError("invalid time (continuous attribute expected)");

        if (!tme.isSpecial())
          addExample(table, newWeight, *ei, WEIGHT(*ei) * (tme.floatV>thisMaxTime ? 1.0 : tme.floatV / thisMaxTime), complementary);
      }
    }
  }

  else if ((method == km) || (method == bayes)) {
    if ((km==bayes) && (maxTime<=0.0))
      raiseError("'maxTime' should be set when 'method' is 'Bayes'");
      
    PDistribution KM = (method == km) ? kaplanMeier(gen, outcomeIndex, eventValue, timeIndex, weightID)
                                      : bayesSurvival(gen, outcomeIndex, eventValue, timeIndex, weightID, maxTime);

    float KM_max = maxTime>0.0 ? KM->p(maxTime) : (*KM.AS(TContDistribution)->distribution.rbegin()).second;

    newWeight = getMetaID();
    PEITERATE(ei, gen) {
      if (!(*ei)[outcomeIndex].isSpecial() && (*ei)[outcomeIndex].intV==failIndex)
        addExample(table, newWeight, *ei, WEIGHT(*ei), -1);
      else {
        const TValue &tme = (*ei)[timeIndex];
        if (tme.varType != TValue::FLOATVAR)
          raiseError("invalid time (continuous attribute expected)");
        if (tme.varType != TValue::FLOATVAR)
          raiseError("invalid time (continuous value expected)");
        if (!tme.isSpecial()) {
          if (tme.floatV > maxTime)
            addExample(table, newWeight, *ei, WEIGHT(*ei), -1);
          else {
            float KM_t = KM->p(tme.floatV);
            if (method==km) {
              if (KM_t>0) {
                float origw = WEIGHT(*ei);
                float fact = KM_max/KM_t;
                addExample(table, newWeight, *ei, origw*fact, complementary, origw*(1-fact));
              }
            }
            else {
              float origw = WEIGHT(*ei);
              addExample(table, newWeight, *ei, origw*KM_t, complementary, origw*(1-KM_t));
            }
          }
        }
      }
    }
  }

  else
    raiseError("unknown weighting method");

  return wtable;
}
  


TPreprocessor_discretize::TPreprocessor_discretize()
: attributes(),
  discretizeClass(false),
  method()
{}


TPreprocessor_discretize::TPreprocessor_discretize(PVarList attrs, const bool &nocl, PDiscretization meth)
: attributes(attrs),
  discretizeClass(nocl),
  method(meth)
{}


PExampleGenerator TPreprocessor_discretize::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ 
  checkProperty(method);

  TVarList discretized;
  vector<int> discretizedMetas;
  TDomain *newDomain = mlnew TDomain();
  PDomain wdomain(newDomain);
  
  const TDomain &domain = gen->domain.getReference();
  
  const_PITERATE(TVarList, vi, domain.attributes)
    if (   ((*vi)->varType == TValue::FLOATVAR)
        && (   !attributes || !attributes->size() 
            || exists(attributes->begin(), attributes->end(), *vi))) {
      PVariable evar = method->operator()(gen, *vi);
      newDomain->variables->push_back(evar);
      newDomain->attributes->push_back(evar);
      discretized.push_back(*vi);
    }
    else {
      newDomain->variables->push_back(*vi);
      newDomain->attributes->push_back(*vi);
    }

  // classVar discretization
  if (domain.classVar){
	  if (domain.classVar->varType == TValue::FLOATVAR
		  && (   !attributes || !attributes->size()
		             || exists(attributes->begin(), attributes->end(), domain.classVar))
		  && discretizeClass) {
		       PVariable evar = method->operator()(gen, domain.classVar);
		       newDomain->variables->push_back(evar);
		       newDomain->classVar = evar;
		       discretized.push_back(evar);
		     }
		     else {
		       newDomain->variables->push_back(domain.classVar);
		       newDomain->classVar = domain.classVar;
		     }
  }
  
  if (attributes)
    PITERATE(TVarList, ai, attributes)
      if (!exists(discretized.begin(), discretized.end(), *ai)) {
        long varNum = domain.getVarNum(*ai);
        if (varNum == ILLEGAL_INT)
          raiseError("Attribute '%s' is not found", (*ai)->get_name().c_str());
        else if ((varNum >= 0) || ((*ai)->varType != TValue::FLOATVAR))
          raiseError("Attribute '%s' is not continuous", (*ai)->get_name().c_str());
        else {
          PVariable evar = method->operator()(gen, *ai);
          TMetaDescriptor ndsc(varNum, evar);
          newDomain->metas.push_back(ndsc);
          discretizedMetas.push_back(varNum);
        }
      }

  const_ITERATE(TMetaVector, mi, domain.metas)
    if (!exists(discretizedMetas.begin(), discretizedMetas.end(), (*mi).id))
      newDomain->metas.push_back(*mi);
      
  newWeight = weightID;
  return mlnew TExampleTable(newDomain, gen);
}


TImputeClassifier::TImputeClassifier(PVariable newVar, PVariable oldVar)
: TClassifier(newVar),
  classifierFromVar(mlnew TClassifierFromVar(newVar, oldVar))
{}

TImputeClassifier::TImputeClassifier(const TImputeClassifier &old)
: TClassifier(old),
  classifierFromVar(old.classifierFromVar),
  imputer(old.imputer)
{}


TValue TImputeClassifier::operator ()(const TExample &ex)
{
  checkProperty(classifierFromVar);
  checkProperty(imputer);

  const TValue res = classifierFromVar->call(ex);

  return res.isSpecial() ? imputer->call(ex) : res;
}


PExampleGenerator TPreprocessor_imputeByLearner::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{
  checkProperty(learner);

  TDomain &domain = gen->domain.getReference();

  // determine the attributes with unknown values
  vector<int> knowns;
  for(int i = 0, e = domain.attributes->size(); i<e; i++)
    knowns.push_back(i);
  vector<int> unknowns;

  PEITERATE(ei, gen) {
    for(int rei = 1; rei--; )
      ITERATE(vector<int>, ui, knowns)
        if ((*ei)[*ui].isSpecial()) {
          unknowns.push_back(*ui);
          knowns.erase(ui);
          rei = 1;
          break; // break out of this ITERATE since the vector has changed, but set rei to 1 to enter it once again...
        }
    if (!knowns.size())
      break;
  }

  TVarList newVars = domain.attributes.getReference();
  TVarList::iterator nvi(newVars.begin());
  ITERATE(vector<int>, ki, unknowns) {
    PVariable &oldVar = domain.attributes->at(*ki);
    PVariable newVar = CLONE(TVariable, oldVar);

    TVarList learnAttrs = domain.attributes.getReference();
    learnAttrs.erase(learnAttrs.begin() + *ki);
    PDomain learnDomain = mlnew TDomain(oldVar, learnAttrs);
    PExampleGenerator data = mlnew TExampleTable(learnDomain, gen);

    TImputeClassifier *imputeClassifier = mlnew TImputeClassifier(newVar, oldVar);
    PClassifier wimputeClassifier = imputeClassifier;

    imputeClassifier->imputer = learner->call(data, weightID);

    newVar->getValueFrom = wimputeClassifier;

    newVars[*ki] = newVar;
  }

  newWeight = weightID;
  PDomain newDomain = mlnew TDomain(domain.classVar, newVars);
  return mlnew TExampleTable(newDomain, gen);
}



TPreprocessor_filter::TPreprocessor_filter(PFilter filt)
: filter(filt)
{}

PExampleGenerator TPreprocessor_filter::operator()(PExampleGenerator gen, const int &weightID, int &newWeight)
{ checkProperty(filter);
  newWeight = weightID;
  return filterExamples(filter, gen);
}


PExampleTable TTableAverager::operator()(PExampleGeneratorList tables) const
{
    if (tables->size() == 0)
        return PExampleTable();
      
    PExampleGenerator &firstTable = tables->front();

    if (tables->size() == 1)
        return new TExampleTable(firstTable);
      
    const PDomain &firstDomain = firstTable->domain;
    TExampleGeneratorList::iterator ti(tables->begin()), te(tables->end());
    TVarList::const_iterator d1i, d2i, d1e(firstDomain->variables->end());
    while(++ti != te) {
        const PDomain &thisDomain = (*ti)->domain;
        if (thisDomain == firstDomain)
            continue;
        if (   (thisDomain->attributes->size() != firstDomain->attributes->size())
            || (thisDomain->classVar != firstDomain->classVar))
            raiseError("Cannot average data from different domains");
        for(d1i = firstDomain->variables->begin(), d2i = thisDomain->variables->begin();
            d1i != d1e; d1i++, d2i++) {
            if (*d1i != *d2i)
                raiseError("Cannot average data from different domains");
        }
    }
    
    TExampleTable *table = new TExampleTable(firstDomain);
    TExampleTable *pfirst = dynamic_cast<TExampleTable *>(firstTable.getUnwrappedPtr());
    if (pfirst)
        table->reserve(pfirst->size());
    PExampleTable wtable = table;
    
    if (tables->size() == 2) {
        for(TExampleIterator e1i = firstTable->begin(), e2i = tables->back()->begin(); e1i; ++e1i, ++e2i) {
              TExample::iterator nei = table->new_example().begin();
              d1i = firstDomain->variables->begin();
              for(TExample::const_iterator ee1i((*e1i).begin()), ee2i((*e2i).begin());
                  d1i != d1e; d1i++, ee1i++, ee2i++, nei++) {
                  if ((*d1i)->varType != TValue::FLOATVAR) {
                      *nei = *ee1i;
                  }
                  else {
                      // ee1i's special value is used when both are special
                      if (ee2i->isSpecial())
                          *nei = *ee1i;
                      else if (ee1i->isSpecial())
                          *nei = *ee2i;
                      else
                          *nei = TValue((ee1i->floatV + ee2i->floatV)/2);
                 }
             }
        }
        return wtable;
    }
    
    float *values = new float[tables->size()], *ve;
    vector<TExampleIterator> iterators;
    for(ti = tables->begin(); ti != te; ti++)
        iterators.push_back((*ti)->begin());
    TExampleIterator &firstIter = iterators.front();
    vector<TExampleIterator>::iterator ib(iterators.begin()), ii, ie(iterators.end());
    for(;;) {
        for(ii = ib; (ii != ie) && *ii; ii++);
        if (ii != ie)
            break;
        TExample::iterator nei = table->new_example().begin();
        int attrNo = 0;
        for(d1i = firstDomain->variables->begin(); d1i != d1e; d1i++, attrNo++, nei++) {
            if ((*d1i)->varType != TValue::FLOATVAR) {
               *nei = (*firstIter)[attrNo];
            }
            else {
                ve = values;
                for(ii = ib; ii != ie; ii++) {
                  if (!(**ii)[attrNo].isSpecial())
                    *ve++ = TValue((**ii)[attrNo].floatV);
                }
                if (ve == values)
                    *nei = TValue((*firstIter)[attrNo]);
                else if (ve-values == 1)
                    *nei = TValue(*values);
                else if (ve-values == 2)
                    *nei = TValue((values[0] + values[1]) / 2);
                else {
                    float *mid = values+(ve-values)/2;
                    nth_element(values, mid, ve);
                    if ((ve-values) % 2) {
                      *nei = TValue(*mid);
                    }
                    else {
                      *nei = TValue((*mid + *max_element(values, mid)) / 2);
                    }
                }
            }
        }      

        for(ii = ib; ii != ie; ++*ii++);
    }
    delete values;
    
    for(ii = ib; ii != ie; ii++)
        if (*ii) {
            raiseError("Cannot average tables of different lengths");
        }
        
    return wtable;
}
