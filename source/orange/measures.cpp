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


// to include Python.h before STL defines a template set (doesn't work with VC >6.0)
#include "garbage.hpp" 

#include <math.h>
#include <set>

#include "stladdon.hpp"
#include "student.hpp"
#include "random.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "distance.hpp"
#include "contingency.hpp"
#include "classify.hpp"
#include "symmatrix.hpp"

#include "cost.hpp"
#include <vector>

#include "relief.ppp"
#include "measures.ppp"


void checkDiscrete(const PContingency &cont, char *measure)
{ if (cont->varType!=TValue::INTVAR)
    if (cont->outerVariable)
      raiseErrorWho(measure, "cannot evaluate the non-discrete attribute '%s'", cont->outerVariable->name.c_str());
    else
      raiseErrorWho(measure, "cannot evaluate continuous attributes");

  if (cont->innerVariable) {
    if (cont->innerVariable->varType != TValue::INTVAR)
      raiseErrorWho(measure, "cannot work with continuous outcome '%s'", cont->innerVariable->name.c_str());
  }
  else
    if (!cont->innerDistribution.is_derived_from(TDiscDistribution))
      raiseErrorWho(measure, "expects discrete class attribute");
}


void checkDiscreteContinuous(const PContingency &cont, char *measure)
{ if (cont->varType!=TValue::INTVAR)
    if (cont->outerVariable)
      raiseErrorWho(measure, "cannot evaluate the non-discrete attribute '%s'", cont->outerVariable->name.c_str());
    else
      raiseErrorWho(measure, "cannot evaluate continuous attributes");

  if (cont->innerVariable) {
    if (cont->innerVariable->varType != TValue::FLOATVAR)
      raiseErrorWho(measure, "cannot work with discrete outcome '%s'", cont->innerVariable->name.c_str());
  }
  else
    if (!cont->innerDistribution.is_derived_from(TContDistribution))
      raiseErrorWho(measure, "expects continuous outcome");
}


/* Prepares the common stuff for binarization through attribute quality assessment:
   - a binary attribute 
   - a contingency matrix for this attribute
   - a DomainContingency that contains this matrix at position newpos (the last)
   - dis0 and dis1 (or con0 and con1, if the class is continuous) that point to distributions
     for the left and the right branch
*/
PContingency prepareBinaryCheat(PDistribution classDistribution, PContingency origContingency,
                                PVariable &bvar, 
                                TDiscDistribution *&dis0, TDiscDistribution *&dis1,
                                TContDistribution *&con0, TContDistribution *&con1)
{
  TEnumVariable *ebvar = mlnew TEnumVariable("");
  bvar = ebvar;
  ebvar->addValue("0");
  ebvar->addValue("1");

  /* An ugly cheat that is prone to cause problems when Contingency class is changed.
     It is fast, though :) */
  TContingencyClass *cont = mlnew TContingencyAttrClass(bvar, classDistribution->variable);
  cont->innerDistribution = classDistribution;
  cont->operator[](1);

  TDiscDistribution *outerDistribution = cont->outerDistribution.AS(TDiscDistribution);
  outerDistribution->cases = origContingency->outerDistribution->cases;
  outerDistribution->abs = origContingency->outerDistribution->abs;
  outerDistribution->normalized = origContingency->outerDistribution->normalized;

  if (classDistribution->variable->varType == TValue::INTVAR) {
    dis0 = cont->discrete->front().AS(TDiscDistribution);
    dis1 = cont->discrete->back().AS(TDiscDistribution);
    con0 = con1 = NULL;
  }
  else {
    con0 = cont->discrete->front().AS(TContDistribution);
    con1 = cont->discrete->back().AS(TContDistribution);
    dis0 = dis1 = NULL;
  }

  return cont;
}



TMeasureAttribute::TMeasureAttribute(const int aneeds, const bool hd, const bool hc, const bool ts)
: needs(aneeds),
  handlesDiscrete(hd),
  handlesContinuous(hc),
  computesThresholds(ts)
{}


float TMeasureAttribute::operator()(PContingency, PDistribution, PDistribution)
{ raiseError("cannot evaluate attribute from contingencies only"); 
  return 0.0;
}


float TMeasureAttribute::operator()(int attrNo, PDomainContingency domainContingency, PDistribution apriorClass)
{ if (needs>Contingency_Class) 
    raiseError("cannot evaluate attribute from domain contingency only");
  if (attrNo>int(domainContingency->size()))
    raiseError("attribute index out of range");
  return operator()(domainContingency->operator[](attrNo), domainContingency->classes, apriorClass ? apriorClass : domainContingency->classes); 
}


float TMeasureAttribute::operator()(int attrNo, PExampleGenerator gen, PDistribution apriorClass, int weightID)
{ 
  if (needs>DomainContingency)
    return operator()(gen->domain->attributes->at(attrNo), gen, apriorClass, weightID);

  _ASSERT(gen && gen->domain);
  if (!gen->domain->classVar)
    raiseError("can't evaluate attributes on class-less domains");
  if (attrNo>int(gen->domain->attributes->size()))
    raiseError("attribute index out of range");

  if (needs==Contingency_Class) {
    TContingencyAttrClass contingency(gen, attrNo, weightID);

    PDistribution classDistribution = CLONE(TDistribution, contingency.innerDistribution);
    classDistribution->operator+= (contingency.innerDistributionUnknown);

    return operator()(PContingency(contingency), classDistribution, apriorClass ? apriorClass : classDistribution);
  }
   
 TDomainContingency domcont(gen, weightID);
 return operator()(attrNo, PDomainContingency(domcont), apriorClass ? apriorClass : domcont.classes);
}


float TMeasureAttribute::operator ()(PVariable var, PExampleGenerator gen, PDistribution apriorClass, int weightID)
{ if (!gen->domain->classVar)
    raiseError("can't evaluate attributes on class-less domains");
  
  if (needs>DomainContingency)
   raiseError("invalid 'needs'");

  int attrNo=gen->domain->getVarNum(var, false);
  if (attrNo != ILLEGAL_INT)
    return operator()(attrNo, gen, apriorClass, weightID);

  if (needs>Contingency_Class)
    raiseError("invalid 'needs'");

  TContingencyAttrClass contingency(gen, var, weightID);

  PDistribution classDistribution = CLONE(TDistribution, contingency.innerDistribution);
  classDistribution->operator+= (contingency.innerDistributionUnknown);

  return operator()(PContingency(contingency), PDistribution(classDistribution), apriorClass ? apriorClass : classDistribution);
}


float TMeasureAttribute::operator ()(PDistribution dist) const
{ TDiscDistribution *discdist = dist.AS(TDiscDistribution);
  if (discdist)
    return operator()(*discdist);
  
  TContDistribution *contdist = dist.AS(TContDistribution);
  if (contdist)
    return operator()(*contdist);
    
  raiseError("invalid distribution");
  return 0.0;
}

float TMeasureAttribute::operator ()(const TDiscDistribution &) const
{ raiseError("cannot evaluate discrete attributes");
  return 0.0;
}

float TMeasureAttribute::operator ()(const TContDistribution &) const
{ raiseError("cannot evaluate continuous attributes");
  return 0.0;
}


void TMeasureAttribute::thresholdFunction(TFloatFloatList &res, PVariable var, PExampleGenerator gen, PDistribution apriorClass, int weightID)
{ 
  if (!computesThresholds || (needs > Contingency_Class))
    raiseError("cannot compute thresholds");
  if (!gen->domain->classVar)
    raiseError("can't evaluate attributes on class-less domains");

  TContingencyAttrClass contingency(gen, var, weightID);

  PDistribution classDistribution = CLONE(TDistribution, contingency.innerDistribution);
  classDistribution->operator+= (contingency.innerDistributionUnknown);

  thresholdFunction(res, PContingency(contingency), classDistribution, apriorClass ? apriorClass : classDistribution);
}


float TMeasureAttribute::bestThreshold(PDistribution &left_right, float &score, PVariable var, PExampleGenerator gen, PDistribution apriorClass, int weightID, const float &minSubset)
{ 
  if (needs > Contingency_Class) {
    TFloatFloatList res;
    thresholdFunction(res, var, gen, apriorClass, weightID);
    if (!res.size()) {
      score = 0;
      return ILLEGAL_FLOAT;
    }
    float &score = res.front().second;
    float &bestThresh = res.front().first;
    ITERATE(TFloatFloatList, ii, res) {
      if ((*ii).second > score) {
        bestThresh = (*ii).first;
        score = (*ii).second;
      }
    }
    return score;
  }
  
  if (!gen->domain->classVar)
    raiseError("can't evaluate attributes on class-less domains");

  TContingencyAttrClass contingency(gen, var, weightID);

  PDistribution classDistribution = CLONE(TDistribution, contingency.innerDistribution);
  classDistribution->operator+= (contingency.innerDistributionUnknown);

  return bestThreshold(left_right, score, PContingency(contingency), classDistribution, apriorClass ? apriorClass : classDistribution, minSubset);
}


template<class TRecorder>
bool traverseThresholds(TMeasureAttribute *measure, TRecorder &recorder, PVariable &bvar, PContingency origContingency, PDistribution classDistribution, PDistribution apriorClass)
{
  if (measure->needs > measure->Contingency_Class)
    raiseError("cannot compute thresholds from contingencies");

  PVariable var = origContingency->outerVariable;
  if (var->varType != TValue::FLOATVAR)
    raiseError("cannot search for thresholds of a non-continuous variable");

  if (origContingency->continuous->size() < 2)
    return false;

  TDiscDistribution *dis0, *dis1;
  TContDistribution *con0, *con1;
  PContingency cont = prepareBinaryCheat(classDistribution, origContingency, bvar, dis0, dis1, con0, con1);
  TDiscDistribution *outerDistribution = cont->outerDistribution.AS(TDiscDistribution);
  
  const TDistributionMap &distr = *(origContingency->continuous);

  TMeasureAttributeFromProbabilities *mp = dynamic_cast<TMeasureAttributeFromProbabilities *>(measure);
  if (mp && (mp->unknownsTreatment == mp->IgnoreUnknowns))
    classDistribution = cont->innerDistribution;

  if (dis0) { // class is discrete
    *dis0 = TDiscDistribution();
    *dis1 = CAST_TO_DISCDISTRIBUTION(origContingency->innerDistribution);
    const float &left = dis0->abs, &right = dis1->abs;
  
    const_ITERATE(TDistributionMap, threshi, distr) {
      *dis0 += threshi->second;
      *dis1 -= threshi->second;

      if (!recorder.acceptable(threshi->first, left, right))
        continue;

      outerDistribution->distribution[0] = left;
      outerDistribution->distribution[1] = right;

      recorder.record(threshi->first, measure->call(cont, classDistribution, apriorClass), left, right);
    }
  }

  else { // class is continuous
    *con0 = TContDistribution();
    *con1 = CAST_TO_CONTDISTRIBUTION(origContingency->innerDistribution);
    const float &left = con0->abs, &right = con1->abs;

    const_ITERATE(TDistributionMap, threshi, distr) {
      *con0 += threshi->second;
      *con1 -= threshi->second;

      if (!recorder.acceptable(threshi->first, left, right))
        continue;

      cont->outerDistribution->setint(0, left);
      cont->outerDistribution->setint(1, right);
        
      recorder.record(threshi->first, measure->call(cont, classDistribution, apriorClass), left, right);
    }
  }

  return true;
}


class TRecordThresholds {
public:
  TFloatFloatList &res;

  TRecordThresholds(TFloatFloatList &ares)
  : res(ares)
  {}

  inline bool acceptable(const float &, const float &, const float &)
  { return true; }

  inline void record(const float &threshold, const float &score, const float &left, const float &right)
  { if (res.size())
      res.back().first = (res.back().first + threshold) / 2.0;
    res.push_back(make_pair(threshold, score)); 
  }
};


void TMeasureAttribute::thresholdFunction(TFloatFloatList &res, PContingency origContingency, PDistribution classDistribution, PDistribution apriorClass)
{
  PVariable bvar;
  TRecordThresholds recorder(res);
  if (!traverseThresholds(this, recorder, bvar, origContingency, classDistribution, apriorClass))
    res.clear();
  res.erase(res.end()-1);
}


class TRecordMaximalThreshold {
public:
  float minSubset;

  int wins;
  float bestThreshold, bestScore, bestLeft, bestRight;
  //float lastThreshold;
  bool fixLast;
  TRandomGenerator &rgen;

  TRecordMaximalThreshold(TRandomGenerator &rg, const float &minSub = -1)
  : minSubset(minSub),
    wins(0),
    rgen(rg)
  {}

  inline bool acceptable(const float &threshold, const float &left, const float &right)
  { 
    if (fixLast) {
      bestThreshold = (bestThreshold + threshold) / 2.0;
      fixLast = false;
    }
    return (left >= minSubset) && (right >= minSubset);
  }

  void record(const float &threshold, const float &score, const float &left, const float &right)
  {
    if (   (!wins || (score > bestScore)) && ((wins=1)==1)
        || (score == bestScore) && rgen.randbool(++wins)) {
        bestThreshold = threshold;
        fixLast = true;
      bestScore = score;
      bestLeft = left;
      bestRight = right;
    }
  }
};


float TMeasureAttribute::bestThreshold(PDistribution &subsetSizes, float &score, PContingency origContingency, PDistribution classDistribution, PDistribution apriorClass, const float &minSubset)
{
  PVariable bvar;
  TRandomGenerator rgen(classDistribution->abs);
  TRecordMaximalThreshold recorder(rgen, minSubset);
  if (   !traverseThresholds(this, recorder, bvar, origContingency, classDistribution, apriorClass)
      || !recorder.wins)
    return ILLEGAL_FLOAT;

  subsetSizes = mlnew TDiscDistribution(bvar);
  subsetSizes->addint(0, recorder.bestLeft);
  subsetSizes->addint(1, recorder.bestRight);

  score = recorder.bestScore;
  return recorder.bestThreshold;
}


PIntList TMeasureAttribute::bestBinarization(PDistribution &, float &score, PContingency origContingency, PDistribution classDistribution, PDistribution apriorClass, const float &minSubset)
{
  if (needs > Contingency_Class)
    raiseError("cannot compute thresholds from contingencies");

  PVariable var = origContingency->outerVariable;
  if (var->varType != TValue::INTVAR)
    raiseError("cannot search for thresholds of a non-continuous variable");

  if (origContingency->continuous->size() < 2)
    return NULL;

  raiseError("this has not been implemented yet");
  return NULL;
}


PIntList TMeasureAttribute::bestBinarization(PDistribution &subsets, float &score, PVariable var, PExampleGenerator gen, PDistribution apriorClass, int weightID, const float &minSubset)
{ 
  if (!computesThresholds || (needs > Contingency_Class))
    raiseError("cannot compute binarization");
  if (!gen->domain->classVar)
    raiseError("can't evaluate attributes on class-less domains");

  TContingencyAttrClass contingency(gen, var, weightID);

  PDistribution classDistribution = CLONE(TDistribution, contingency.innerDistribution);
  classDistribution->operator+= (contingency.innerDistributionUnknown);

  return bestBinarization(subsets, score, PContingency(contingency), classDistribution, apriorClass ? apriorClass : classDistribution, minSubset);
}

int TMeasureAttribute::bestValue(PDistribution &, float &score, PContingency origContingency, PDistribution classDistribution, PDistribution apriorClass, const float &minSubset)
{
  raiseError("bestValue is not supported by the selected attribute measure");
  return 0;
}

int TMeasureAttribute::bestValue(PDistribution &, float &score, PVariable, PExampleGenerator, PDistribution apriorClass, int weightID, const float &minSubset)
{
  raiseError("bestValue is not supported by the selected attribute measure");
  return 0;
}

bool TMeasureAttribute::checkClassType(const int &varType)
{
  return    ((varType==TValue::INTVAR) && handlesDiscrete)
         || ((varType==TValue::FLOATVAR) && handlesContinuous);
}


void TMeasureAttribute::checkClassTypeExc(const int &varType)
{
  if (varType==TValue::INTVAR) {
    if (!handlesDiscrete)
      raiseError("cannot work with discrete classes");
  }
  else if (varType==TValue::FLOATVAR) {
    if (!handlesContinuous)
      raiseError("cannot work with continuous classes");
  }
}


TMeasureAttributeFromProbabilities::TMeasureAttributeFromProbabilities(const bool hd, const bool hc, const int unkTreat)
: TMeasureAttribute(Contingency_Class, hd, hc, hd), // we can compute thresholds, if we handle discrete attributes...
  unknownsTreatment(unkTreat)
{}


float TMeasureAttributeFromProbabilities::operator()(PContingency cont, PDistribution classDistribution, PDistribution aprior)
{ 
  // if unknowns are ignored, we only take the class distribution for examples for which the attribute's value was defined
  if (unknownsTreatment == IgnoreUnknowns)
    classDistribution = cont->innerDistribution;

  if (estimatorConstructor) {
    classDistribution = estimatorConstructor->call(classDistribution, aprior)->call();
    if (!classDistribution)
      raiseError("'estimatorConstructor' cannot return the distribution");
  }

  if (conditionalEstimatorConstructor) {
    PContingency cont_e = conditionalEstimatorConstructor->call(cont, aprior)->call();
    if (!cont_e)
      raiseError("'conditionalEstimatorConstructor cannot return contingency matrix");
    cont_e->outerDistribution = cont->outerDistribution;
    cont_e->innerDistribution = classDistribution;

    cont = cont_e;
  }

  TDiscDistribution *dcDist = classDistribution.AS(TDiscDistribution);
  if (!dcDist)
    raiseError("discrete class expected");

  return operator()(cont, *dcDist);
}


inline float round0(const float &x)
{ 
  return (x > -1e-6) && (x < 1e-6) ? 0.0 : x;
}


float getEntropy(const vector<float> &vf)
{ 
  float n = 0.0, sum = 0.0;
  int noDif0 = 0;
  const_ITERATE(vector<float>, vi, vf)
    if (*vi>0) {
      sum += (*vi)*log(*vi);
      n += *vi;
      noDif0++;
    }
  return (noDif0>1) ? (log(float(n))-sum/n) / log(2.0) : 0;
}


float getEntropy(PContingency cont, int unknownsTreatment)
{ 
  checkDiscrete(cont, "getEntropy");

  float sum = 0.0, N = 0.0;
  const TDiscDistribution &outer = CAST_TO_DISCDISTRIBUTION(cont->outerDistribution);
  TDistributionVector::const_iterator mostCommon(
     unknownsTreatment == TMeasureAttribute::UnknownsToCommon
       ? cont->discrete->begin() + outer.highestProbIntIndex()
       : cont->discrete->end());

  const_ITERATE(TDistributionVector, ci, *cont->discrete) {
    const TDiscDistribution &dist = CAST_TO_DISCDISTRIBUTION(*ci);
    if (ci == mostCommon) {
      TDiscDistribution dist2 = dist;
      dist2 += cont->innerDistributionUnknown;
      N += dist2.cases;
      sum += dist2.cases * getEntropy(dist2);
    }
    else {
      N += dist.cases;
      sum += dist.cases * getEntropy(dist.distribution);
    }
  }

  if (unknownsTreatment == TMeasureAttribute::UnknownsAsValue) {
    const float &cases = cont->innerDistributionUnknown->cases;
    N += cases;
    sum += cases * getEntropy(CAST_TO_DISCDISTRIBUTION(cont->innerDistributionUnknown));
  }

  return N ? sum/N : 0.0;
}



TMeasureAttribute_info::TMeasureAttribute_info(const int &unk)
: TMeasureAttributeFromProbabilities(true, false, unk)
{}


float TMeasureAttribute_info::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  // checkDiscrete is called by getEntropy

  const TDistribution &outer = probabilities->outerDistribution.getReference();
  if (!outer.cases)
    return 0.0;

  float info = getEntropy(classProbabilities) - getEntropy(probabilities, unknownsTreatment);
  if (unknownsTreatment == ReduceByUnknowns)
    info *= outer.cases / (outer.unknowns + outer.cases);

  return round0(info);
}


float TMeasureAttribute_info::operator()(const TDiscDistribution &dist) const
{ 
  return -getEntropy(dist);
}


TMeasureAttribute_gainRatio::TMeasureAttribute_gainRatio(const int &unk)
: TMeasureAttributeFromProbabilities(true, false, unk)
{}



float TMeasureAttribute_gainRatio::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  checkDiscrete(probabilities, "MeasureAttribute_gainRatio");

  const TDiscDistribution &outer = CAST_TO_DISCDISTRIBUTION(probabilities->outerDistribution);
  if (!outer.cases)
    return 0.0;

  float attributeEntropy;
  if (unknownsTreatment == UnknownsAsValue) {
    vector<float> dist(outer);
    dist.push_back(probabilities->innerDistributionUnknown->cases);
    attributeEntropy = getEntropy(dist);
  }
  else
    attributeEntropy = getEntropy(outer);

  if (attributeEntropy<1e-20)
    return 0.0;
  
  float gain = getEntropy(classProbabilities) - getEntropy(probabilities, unknownsTreatment);
  if (gain<1e-20)
    return 0.0;
  
  gain /= attributeEntropy;

  if (unknownsTreatment == ReduceByUnknowns)
    gain *= outer.cases / (outer.unknowns + outer.cases);

  return round0(gain);
}



float TMeasureAttribute_gainRatioA::operator()(const TDiscDistribution &dist) const
{ return round0(-getEntropy(dist) * log(float(dist.size()))); }




float getGini(const vector<float> &vf)
{ float sum = 0.0, N = 0.0;
  const_ITERATE(vector<float>, vi, vf) {
    N += *vi;
    sum += (*vi)*(*vi);
  }
  return N ? (1 - sum/N/N)/2 : 0.0;
}


float getGini(PContingency cont, int unknownsTreatment)
{ 
  float sum = 0.0, N = 0.0;
  const TDiscDistribution &outer = CAST_TO_DISCDISTRIBUTION(cont->outerDistribution);
  TDistributionVector::const_iterator mostCommon(
    unknownsTreatment == TMeasureAttribute::UnknownsToCommon
      ? cont->discrete->begin() + outer.highestProbIntIndex()
      : cont->discrete->end());

  const_ITERATE(TDistributionVector, ci, *cont->discrete) {
    const TDiscDistribution &dist = CAST_TO_DISCDISTRIBUTION(*ci);
    if (ci == mostCommon) {
      TDiscDistribution dist2 = dist;
      dist2 += cont->innerDistributionUnknown;
      N += dist2.cases;
      sum += dist2.cases * getGini(dist2);
    }
    else {
      N += dist.cases;
      sum += dist.cases * getGini(dist.distribution);
    }
  }

  if (unknownsTreatment == TMeasureAttribute::UnknownsAsValue) {
    const float &cases = cont->innerDistributionUnknown->cases;
    N += cases;
    sum += cases * getGini(CAST_TO_DISCDISTRIBUTION(cont->innerDistributionUnknown));
  }

  return N ? sum/N : 0.0;
}


TMeasureAttribute_gini::TMeasureAttribute_gini(const int &unk)
: TMeasureAttributeFromProbabilities(true, false, unk)
{}


float TMeasureAttribute_gini::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  checkDiscrete(probabilities, "MeasureAttribute_gini");
  
  const TDistribution &outer = probabilities->outerDistribution.getReference();
  if ((unknownsTreatment == ReduceByUnknowns) && (outer.unknowns == outer.cases))
    return 0.0;
 
  float gini = getGini(classProbabilities) - getGini(probabilities, unknownsTreatment);
  if (unknownsTreatment == ReduceByUnknowns)
    gini *= (outer.cases / (outer.unknowns + outer.cases));

  return round0(gini);
}


float TMeasureAttribute_gini::operator()(const TDiscDistribution &dist) const
{ return round0(-getGini(dist)); }



TMeasureAttribute_relevance::TMeasureAttribute_relevance(const int &unk)
: TMeasureAttributeFromProbabilities(true, false, unk)
{}



float TMeasureAttribute_relevance::valueRelevance(const TDiscDistribution &dval, const TDiscDistribution &classDist)
{ 

  TDiscDistribution::const_iterator ci(classDist.begin()), ce(classDist.end()), hci(ci);
  TDiscDistribution::const_iterator di(dval.begin()), de(dval.end());

  for (; (di!=de) && (ci!=ce) && (*ci<1e-20); ci++, di++);
  if ((ci==ce) || (di==de))
    return 0.0;

  /* 'leftout' is the element for the most probable class encountered so far
      If a more probable class appears, 'leftout' is added and new leftout taken
      If there is more than one most probable class, the one with higher aprior probability
      is taken (as this gives higher relevance). If there is more than one such class,
      it doesn't matter which one we take. */  
  float relev = 0.0;
  float highestProb = *di;
  float leftout = *di / *ci;

  while(++ci!=ce && ++di!=de) 
    if (*ci>=1e-20) {
      const float &tras = *di / *ci;
      if (   (*di >  highestProb)
          || (*di == highestProb) && (leftout < tras)) {
        relev += leftout;
        leftout = tras;
        highestProb = *di;
      }
      else
        relev += tras;
    }

  return relev;
}


float TMeasureAttribute_relevance::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  checkDiscrete(probabilities, "MeasureAttribute_relevance");
  
  const TDistribution &outer = probabilities->outerDistribution.getReference();
  if ((unknownsTreatment == ReduceByUnknowns) && (outer.unknowns == outer.cases))
    return 0.0;
 
  int C = 0;
  const_ITERATE(TDiscDistribution, di, classProbabilities)
    if (*di > 1e-20)
      C++;
  if (C<=1)
    return 0.0;

  TDistributionVector::const_iterator mostCommon (unknownsTreatment == UnknownsToCommon
    ? probabilities->discrete->begin() + outer.highestProbIntIndex()
    : probabilities->discrete->end());
 
  float relevance = 0.0;
  const_ITERATE(TDistributionVector, ci, *probabilities->discrete) {
    const TDiscDistribution &dist = CAST_TO_DISCDISTRIBUTION(*ci);
    if (ci == mostCommon) {
      TDiscDistribution dist2 = dist;
      dist2 += probabilities->innerDistributionUnknown;
      relevance += valueRelevance(dist2, classProbabilities);
    }
    else {
      relevance += valueRelevance(dist, classProbabilities);
    }
  }

  if (unknownsTreatment == TMeasureAttribute::UnknownsAsValue)
    relevance += valueRelevance(CAST_TO_DISCDISTRIBUTION(probabilities->innerDistributionUnknown), classProbabilities);

  relevance = 1.0 - relevance / float(C-1);

  if (unknownsTreatment == TMeasureAttribute::ReduceByUnknowns)
    relevance *= (outer.cases / (outer.unknowns + outer.cases));
  return round0(relevance);
}


TMeasureAttribute_logOddsRatio::TMeasureAttribute_logOddsRatio()
: TMeasureAttributeFromProbabilities(true, false, IgnoreUnknowns)
{}

float TMeasureAttribute_logOddsRatio::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  checkDiscrete(probabilities, "MeasureAttribute_chiSquare");
  if (probabilities->discrete->size() == 2) {
    const TDiscDistribution &pdist = CAST_TO_DISCDISTRIBUTION(probabilities->discrete->back());
    const TDiscDistribution &qdist = CAST_TO_DISCDISTRIBUTION(probabilities->discrete->front());
    if ((pdist.size() == 2) && (qdist.size() == 2)) {
      const float p = pdist.p(1);
      const float q = qdist.p(1);
      if ((p < 1e-6) || (1 - q < 1e-6))
        return -999999;
      if ((1 - p < 1e-6) || (q < 1e-6))
        return 999999;
      return log(p/(1-p) / (q/(1-q)));
    }
  }

  raiseError("this measure is defined for binary attribute and class");
  return 0;
}  

#include "stat.hpp"

TMeasureAttribute_chiSquare::TMeasureAttribute_chiSquare(const int &unk, const bool probs)
: TMeasureAttributeFromProbabilities(true, false, unk),
  computeProbabilities(probs)
{}


float TMeasureAttribute_chiSquare::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  checkDiscrete(probabilities, "MeasureAttribute_chiSquare");
  
  const TDistribution &outer = probabilities->outerDistribution.getReference();
  if (   (classProbabilities.size() <= 0)
      || (probabilities->discrete->size() <= 0)
      || (unknownsTreatment == ReduceByUnknowns) && (outer.unknowns == outer.cases))
    return 0.0;

  TDiscDistribution expClass(classProbabilities);
  expClass.normalize();

  float df_in = -1.0, df_out = -1.0;
  ITERATE(TDiscDistribution, pi, expClass)
    if (*pi > 1e-6)
      df_in += 1.0;

  if (df_in <= 0.0)
    return computeProbabilities ? 1.0 : 0.0;

  float chisq = 0.0;
  const_ITERATE(TDistributionVector, ci, *probabilities->discrete) {
    float n0 = 0.0, psum = 0.0;
    const TDiscDistribution &dist = CAST_TO_DISCDISTRIBUTION(*ci);
    for(TDiscDistribution::const_iterator oi(dist.begin()), oe(dist.end()), pi(expClass.begin()), pe(expClass.end());
        (oi != oe) && (pi != pe); oi++, pi++) {
      if (*pi > 1e-6) {
        psum += *oi * *oi / *pi;
        n0 += *oi;
      }
    }
    if (n0 > 1e-6) {
      chisq += psum/n0 - n0;
      df_out += 1.0;
    }
  }

  if (df_out <= 0.0)
    return computeProbabilities ? 1.0 : 0.0;

  return computeProbabilities ? chisqprob(chisq, df_in * df_out) : chisq;
}




TMeasureAttribute_cost::TMeasureAttribute_cost(PCostMatrix costs)
: TMeasureAttributeFromProbabilities(true, false),
  cost(costs)
{}


float TMeasureAttribute_cost::majorityCost(const TDiscDistribution &dval)
{ float cost;
  TValue cclass;
  majorityCost(dval, cost, cclass);
  return cost;
}


void TMeasureAttribute_cost::majorityCost(const TDiscDistribution &dval, float &ccost, TValue &cclass)
{ 
  checkProperty(cost);

  int dsize = dval.size();
  if (dsize > cost->dimension)
    raiseError("cost matrix is too small");

  TRandomGenerator srgen(dval.sumValues());

  ccost = numeric_limits<float>::max();
  int wins = 0, bestPrediction;

  for(int predicted = 0; predicted < dsize; predicted++) {
    float thisCost = 0;
    for(int correct = 0; correct < dsize; correct++)
      thisCost += dval[correct] * cost->cost(predicted, correct);

    if (   (thisCost<ccost) && ((wins=1)==1)
        || (thisCost==ccost) && srgen.randbool(++wins)) {
      bestPrediction = predicted;
      ccost = thisCost; 
    }
  }
  
  ccost /= dval.abs;
  cclass = TValue(bestPrediction);
}


float TMeasureAttribute_cost::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  checkDiscrete(probabilities, "MeasureAttribute_cost");

  const TDistribution &outer = probabilities->outerDistribution.getReference();
  if ((unknownsTreatment == ReduceByUnknowns) && (outer.unknowns == outer.cases))
    return 0.0;
 
  checkProperty(cost);

  float stopCost = majorityCost(classProbabilities);
  
  TDistributionVector::const_iterator mostCommon = (unknownsTreatment == UnknownsToCommon)
    ? probabilities->discrete->begin() + outer.highestProbIntIndex()
    : probabilities->discrete->end();

  float continueCost = 0;
  float N = 0;
  const_ITERATE(TDistributionVector, ci, *probabilities->discrete) {
    const TDiscDistribution &dist = CAST_TO_DISCDISTRIBUTION(*ci);
    if (ci == mostCommon) {
      TDiscDistribution dist2 = dist;
      dist2 += probabilities->innerDistributionUnknown;
      if (dist2.cases && dist2.abs) {
        N += dist2.cases;
        continueCost += dist2.cases * majorityCost(dist2);
      }
    }
    else {
      if (dist.cases && dist.abs) {
        N += dist.cases;
        continueCost += dist.cases * majorityCost(dist.distribution);
      }
    }
  }

  if (unknownsTreatment == UnknownsAsValue) {
    const float &cases = probabilities->innerDistributionUnknown->cases;
    if (cases) {
      N += cases;
      continueCost += cases * majorityCost(CAST_TO_DISCDISTRIBUTION(probabilities->innerDistributionUnknown));
    }
  }

  if (N)
    continueCost /= N;

  float cost = stopCost - continueCost;
  if ((unknownsTreatment == ReduceByUnknowns) && outer.unknowns) // to avoid div by zero if !cases, too
    cost *= (outer.cases / (outer.unknowns + outer.cases));

  return round0(cost);
}



TMeasureAttribute_MSE::TMeasureAttribute_MSE(const int &unkTreat)
: TMeasureAttribute(Contingency_Class, false, true),
  m(0),
  unknownsTreatment(unkTreat)
{}


float TMeasureAttribute_MSE::operator()(PContingency cont, PDistribution classDistribution, PDistribution apriorClass)
{
  checkDiscreteContinuous(cont, "MeasureAttribute_MSE");

  const TDistribution &outer = CAST_TO_DISCDISTRIBUTION(cont->outerDistribution);
  
  if (cont->innerVariable->varType!=TValue::FLOATVAR)
    raiseError("cannot evaluate attribute in domain with discrete classes");
  if (cont->outerVariable->varType!=TValue::INTVAR)
    raiseError("cannot evaluate continuous attributes");

  const TContDistribution &classDist = CAST_TO_CONTDISTRIBUTION(classDistribution);

  float W=classDist.abs;
  if (W<=0)
    return 0.0;

  float I_orig=(classDist.sum2-classDist.sum*classDist.sum/W)/W;
  if (I_orig<=0.0)
    return 0.0;

  TDistributionVector::const_iterator mostCommon = (unknownsTreatment == UnknownsToCommon)
    ? cont->discrete->begin() + outer.highestProbIntIndex()
    : cont->discrete->end();

  float I=0;
  float downW=0;
  const_ITERATE(TDistributionVector, ci, *cont->discrete) {
    const TContDistribution &tdist = CAST_TO_CONTDISTRIBUTION(*ci);
    if (ci==mostCommon) {
      const float ssum2 = tdist.sum2 + cont->innerDistribution.AS(TContDistribution)->sum2;
      const float ssum = tdist.sum + cont->innerDistribution.AS(TContDistribution)->sum;
      const float sabs = tdist.abs + cont->innerDistribution.AS(TContDistribution)->abs;
      I += ssum2  -  ssum*ssum / sabs;
      downW += sabs;
    }
    else {
      if (tdist.abs>0) {
        I += tdist.sum2 - tdist.sum*tdist.sum/tdist.abs;
        downW += tdist.abs;
      }
    }
  }

  if (unknownsTreatment == UnknownsAsValue) {
    const TContDistribution &tdist = CAST_TO_CONTDISTRIBUTION(cont->innerDistributionUnknown);
    I += tdist.sum2 - tdist.sum*tdist.sum/tdist.abs;
    downW += tdist.abs;
  }

  if (apriorClass && (m>0)) {
    const TContDistribution &tdist = CAST_TO_CONTDISTRIBUTION(apriorClass);
    I =   (I + m * (tdist.sum2 - tdist.sum * tdist.sum/tdist.abs) / tdist.abs)
        / (downW + m);
  }
  else 
    I /= downW;

  float mse = (I_orig - I)/I_orig;
  if (unknownsTreatment == ReduceByUnknowns)
    mse *= (outer.cases / (outer.unknowns + outer.cases));
  
  return round0(mse);
}



TMeasureAttribute_relief::TMeasureAttribute_relief(int ak, int am)
: TMeasureAttribute(Generator, true, false), 
  k(ak),
  m(am),
  checkCachedData(true),
  prevExamples(-1),
  prevWeight(0),
  prevChecksum(0),
  prevK(-1),
  prevM(-1)
{}




inline bool compare2nd(const pair<int, float> &o1, const pair<int, float> &o2)
{ return o1.second < o2.second; }


void TMeasureAttribute_relief::prepareNeighbours(PExampleGenerator gen, const int &weightID)
{
  neighbourhood.clear();

  if (!gen->domain->classVar)
    raiseError("classless domain");

  const bool regression = gen->domain->classVar->varType == TValue::FLOATVAR;

  if (!regression && (gen->domain->classVar->varType != TValue::INTVAR))
    raiseError("cannot compute ReliefF of a class that is neither discrete nor continuous");
  
  if (gen.is_derived_from(TExampleTable)) {
    storedExamples = mlnew TExampleTable(gen, 1); // must store lock!
  }
  else {
    storedExamples = mlnew TExampleTable(gen->domain);
  }
  TExampleTable &table = dynamic_cast<TExampleTable &>(storedExamples.getReference());
  PEITERATE(ei, gen)
    if (!(*ei).getClass().isSpecial())
      table.addExample(*ei);

  const int N = table.numberOfExamples();
  if (!N)
    raiseError("no examples with known class");

  const int classIdx = table.domain->attributes->size();

  vector<vector<int> > examplesByClasses(regression ? 1 : table.domain->classVar->noOfValues());
  vector<vector<int > >::iterator ebcb, ebci, ebce;

  float minCl, maxCl;

  if (table.domain->classVar->varType==TValue::INTVAR) {
    int index;
    TExampleIterator ei;

    for(ei = table.begin(), index = 0; ei; ++ei, index++)
      examplesByClasses.at(int((*ei).getClass())).push_back(index);

    for(ebcb = examplesByClasses.begin(), ebci = ebcb, ebce = examplesByClasses.end(); ebci != ebce; ) {
      const int sze = (*ebci).size();
      if (sze)
        ebci++;
      else {
        examplesByClasses.erase(ebci);
        ebce = examplesByClasses.end();
      }
    }
  }
  else {
    ebcb = examplesByClasses.begin(), ebce = examplesByClasses.end();
    ebcb->resize(N);
    int i = 0;
    for(vector<int>::iterator c0i(ebcb->begin()), c0e(ebcb->end()); c0i != c0e; *c0i++ = i++);

    TExampleIterator ei(table.begin());
    minCl = maxCl = (*ei).getClass().floatV;
    while(++ei) {
      const float tex = (*ei).getClass().floatV;
	    if (tex > maxCl)
        maxCl = tex;
		  else if (tex < minCl)
        minCl = tex;
		}
  }


  distance = TExamplesDistanceConstructor_Relief()(gen);
  const TExamplesDistance_Relief &rdistance = dynamic_cast<const TExamplesDistance_Relief &>(distance.getReference());

  TRandomGenerator rgen(N);
  int referenceIndex = 0;
  const bool useAll = (m==-1) || (!weightID && (m>N));
  float referenceExamples, referenceWeight;

  for(referenceExamples = 0; useAll ? (referenceIndex < N) : (referenceExamples < m); referenceExamples += referenceWeight, referenceIndex++) {
    if (!useAll)
      referenceIndex = rgen.randlong(N);
    TExample &referenceExample = table[referenceIndex];
    referenceWeight = WEIGHT(referenceExample);

    const TValue &referenceValue = referenceExample.getClass();
    const int referenceClass= regression ? 0 : referenceExample.getClass().intV;

    neighbourhood.push_back(referenceIndex);
    vector<TNeighbourExample> &refNeighbours = neighbourhood.back().neighbours;

    ndC = 0.0;

    ITERATE(vector<vector<int> >, cli, examplesByClasses) {
      const float inCliClass = (*cli).size();
      vector<pair<int, float> > distances(inCliClass);
      vector<pair<int, float> >::iterator disti = distances.begin(), diste;
      ITERATE(vector<int> , clii, *cli)
        *disti++ = make_pair(*clii, rdistance(referenceExample, table[*clii]));

      diste = distances.end();
      disti = distances.begin();
      sort(disti, diste, compare2nd);

      int startNew = refNeighbours.size();

      while(disti != diste && (disti->second <= 0))
        disti++;

      float inWeight, needwei;
      for(needwei = k; (disti != diste) && (needwei > 1e-6); ) {
        const float thisDist = disti->second;
        inWeight = 0.0;
        const int inAdded = refNeighbours.size();
        do {
          TExample &neighbourExample = table[disti->first];

          const float neighbourWeight = WEIGHT(neighbourExample);
          const float weightEE = neighbourWeight * referenceWeight;
          inWeight += neighbourWeight;

          if (regression) {
            const float classDist = rdistance(classIdx, neighbourExample.getClass(), referenceValue);
            refNeighbours.push_back(TNeighbourExample(disti->first,
                                                      weightEE * classDist,
                                                      weightEE));
            ndC += weightEE * classDist;
          }
          else {
            const int neighbourClass = neighbourExample.getClass().intV;
            refNeighbours.push_back(TNeighbourExample(disti->first, weightEE * (neighbourClass == referenceClass ? -1 : float(inCliClass) / (N - examplesByClasses[neighbourClass].size()))));
          }
        } while ((++disti != diste) && (disti->second == thisDist));

        needwei -= inWeight;
      }

      if (k-needwei > 1) {
        const float adj = 1.0 / (k - needwei);
        if (regression)
          for(vector<TNeighbourExample>::iterator ai(refNeighbours.begin() + startNew), ae(refNeighbours.end()); ai != ae; ai++) {
            ai->weight *= adj;
            ai->weightEE *= adj;
          }
        else
          for(vector<TNeighbourExample>::iterator ai(refNeighbours.begin() + startNew), ae(refNeighbours.end()); ai != ae; ai++)
            ai->weight *= adj;
      }
    }
  }


  if (regression)
    m_ndC = referenceExamples - ndC;
  else
    ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
      const float adj = 1.0 / referenceExamples;
      ITERATE(vector<TNeighbourExample>, nei, rei->neighbours)
        nei->weight *= adj;
    }
}


void TMeasureAttribute_relief::checkNeighbourhood(PExampleGenerator gen, const int &weightID)
{
  if (!gen->domain->classVar)
    raiseError("class-less domain");

  int newChecksum;
  bool renew = false;
  if ((prevExamples != gen->version) || (weightID != prevWeight) || (k != prevK) || (m != prevM)) {
    newChecksum = gen->checkSum(true);
    renew = true;
  }
  else if (checkCachedData) {
    newChecksum = gen->checkSum(true);
    renew = newChecksum != prevChecksum;
  }

  if (renew)  {
    measures.clear();
    prepareNeighbours(gen, weightID);
    prevExamples = gen->version;
    prevWeight = weightID;
    prevChecksum = newChecksum;
    prevK = k;
    prevM = m;
  }
}


float *tabulateContinuousValues(PExampleGenerator gen, const int &weightID, TVariable &variable,
                                float &min, float &max, float &avg, float &N)
{
  float *pc, *precals;
  precals = pc = new float[gen->numberOfExamples()];
  avg = N = 0.0;

  PEITERATE(ei, gen) {
    const TValue &val = variable.computeValue(*ei);
    if (val.isSpecial())
      *pc++ = ILLEGAL_FLOAT;
    else {
      *pc++ = val.floatV;
      if (N == 0.0)
        max = min = val.floatV;
      else if (val.floatV > max)
        max = val.floatV;
      else if (val.floatV < min)
        min = val.floatV;

      const float w = WEIGHT(*ei);
      avg += w * val.floatV;
      N += w;
    }
  }

  if (N > 1e-6)
    avg /= N;

  return precals;
}


int *tabulateDiscreteValues(PExampleGenerator gen, const int &weightID, TVariable &variable,
                            float *&unk, float &bothUnk)
{
  const int noVal = dynamic_cast<TEnumVariable &>(variable).noOfValues();

  int *pc, *precals = pc = new int[gen->numberOfExamples()];
  unk = new float[noVal];

  try {
    float *ui, *ue = unk + noVal;
    for(ui = unk; ui != ue; *ui++ = 0.0);
       
    int *pc = precals;
    PEITERATE(ei, gen) {
      const TValue &val = variable.computeValue(*ei);
      if (val.isSpecial() || (val.intV >= noVal) || (val.intV < 0))
        *pc++ = ILLEGAL_INT;
      else {
        *pc++ = val.intV;
        unk[val.intV] += WEIGHT(*ei);
      }
    }

    bothUnk = 1.0;
    for(ui = unk; ui != ue; ui++) {
      bothUnk -= *ui * *ui;
      *ui = 1 - *ui;
    }
  }
  catch (...) {
    delete unk;
    unk = NULL;
    delete precals;
    precals = NULL;
    throw;
  }

  return precals;
}


float TMeasureAttribute_relief::operator()(PVariable var, PExampleGenerator gen, PDistribution aprior, int weightID)
{
  checkNeighbourhood(gen, weightID);

  // the attribute is in the domain
  const int attrIdx = gen->domain->getVarNum(var, false);
  if (attrIdx != ILLEGAL_INT) {
    if (measures.empty()) {
      const TExamplesDistance_Relief &rdistance = dynamic_cast<const TExamplesDistance_Relief &>(distance.getReference());

      const TExampleTable &table = dynamic_cast<const TExampleTable &>(gen.getReference());
      const int nAttrs = gen->domain->attributes->size();
      measures = vector<float>(nAttrs, 0.0);
      vector<float>::iterator mb(measures.begin()), mi;
      const vector<float>::const_iterator me(measures.end());
      TExample::const_iterator e1i, e1b, e2i;
      int attrNo;

      if (gen->domain->classVar->varType == TValue::FLOATVAR) {
        vector<float> ndA(nAttrs, 0.0);
        vector<float> ndCdA(nAttrs, 0.0);
        vector<float>::iterator ndAb(ndA.begin()), ndAi;
        const vector<float>::const_iterator ndAe(ndA.end());
        vector<float>::iterator ndCdAb(ndCdA.begin()), ndCdAi;

        ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
          const TExample &referenceExample = table[rei->index];
          e1b = referenceExample.begin();
          ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
            const float &weight = nei->weight;
            const float &weightEE = nei->weightEE;
            for(attrNo = 0, e1i = e1b, e2i = table[nei->index].begin(), ndAi = ndAb, ndCdAi = ndCdAb; ndAi != ndAe; ndAi++, ndCdAi++, e1i++, e2i++, attrNo++) {
              const float attrDist = rdistance(attrNo, *e1i, *e2i);
              *ndAi += weightEE * attrDist;
              *ndCdAi += weight * attrDist;
            }
          }
        }
        for(ndAi = ndAb, ndCdAi = ndCdAb, mi = mb; mi != me; mi++, ndAi++, ndCdAi++)
          *mi = *ndCdAi / ndC - (*ndAi - *ndCdAi) / m_ndC;
      }
      else {
        ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
          const TExample &referenceExample = table[rei->index];
          e1b = referenceExample.begin();
          ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
            const float &weight = nei->weight;
            for(attrNo = 0, e1i = e1b, e2i = table[nei->index].begin(), mi = mb; mi != me; e1i++, e2i++, mi++, attrNo++)
              *mi += weight * rdistance(attrNo, *e1i, *e2i);
          }
        }
      }
    }

    return measures[attrIdx];
  }


  // the attribute is not in the domain
  else {
    if (!var->getValueFrom)
      raiseError("attribute is not among the domain attributes and cannot be computed from them");
  
    const TExampleTable &table = dynamic_cast<const TExampleTable &>(gen.getReference());
    TVariable &variable = var.getReference();
    const int nExamples = gen->numberOfExamples();

    PExamplesDistance distance;


    // continuous attribute
    if (variable.varType == TValue::FLOATVAR) {
      float avg, min, max, N;
      float *precals = tabulateContinuousValues(gen, weightID, variable, min, max, avg, N);

      try {
        if ((min == max) || (N < 1e-6)) {
          delete precals;
          return 0.0;
        }

        const float nor = 1.0 / (min-max);

        // continuous attribute, continuous class
        if (gen->domain->classVar->varType == TValue::FLOATVAR) {
          float ndA = 0.0, ndCdA = 0.0;
          ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
            const float refVal = precals[rei->index];
            if (refVal == ILLEGAL_FLOAT)
              ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
                const float neiVal = precals[nei->index];
                const float attrDist = (neiVal == ILLEGAL_FLOAT) ? 0.5 : fabs(avg - neiVal) * nor;
                ndA += nei->weightEE * attrDist;
                ndCdA += nei->weight * attrDist;
              }
            else {
              ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
                const float neiVal = precals[nei->index];
                const float attrDist = fabs(refVal - (neiVal == ILLEGAL_FLOAT ? avg : neiVal)) * nor;
                ndA += nei->weightEE * attrDist;
                ndCdA += nei->weight * attrDist;
              }
            }
          }

          delete precals;
          return ndCdA / ndC - (ndA - ndCdA) / m_ndC;
        }

        // continuous attribute, discrete class
        else {
          float relf = 0.0;

          ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
            const float refVal = precals[rei->index];
            if (refVal == ILLEGAL_FLOAT)
              ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
                const float neiVal = precals[nei->index];
                const float attrDist = (neiVal == ILLEGAL_FLOAT) ? 0.5 : fabs(avg - neiVal) * nor;
                relf += nei->weight * attrDist;
              }
            else {
              ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
                const float neiVal = precals[nei->index];
                const float attrDist = fabs(refVal - (neiVal == ILLEGAL_FLOAT ? avg : neiVal)) * nor;
                relf += nei->weight * attrDist;
              }
            }
          }

          delete precals;
          return relf;
        }

      }
      catch (...) {
        delete precals;
        throw;
      }
    }


    // discrete attribute
    else {
      float *unk, bothUnk;
      int *precals = tabulateDiscreteValues(gen, weightID, var.getReference(), unk, bothUnk);

      try {
        // discrete attribute, continuous class
        if (gen->domain->classVar->varType == TValue::FLOATVAR) {
          float ndA = 0.0, ndCdA = 0.0;
          ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
            const int refVal = precals[rei->index];
            if (refVal == ILLEGAL_INT)
              ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
                const int neiVal = precals[nei->index];
                const float attrDist = (neiVal == ILLEGAL_INT) ? bothUnk : unk[neiVal];
                ndA += nei->weightEE * attrDist;
                ndCdA += nei->weight * attrDist;
              }
            else {
              ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
                const int neiVal = precals[nei->index];
                const float attrDist = (neiVal == ILLEGAL_INT) ? unk[refVal] : (refVal != neiVal ? 1.0 : 0.0);
                ndA += nei->weightEE * attrDist;
                ndCdA += nei->weight * attrDist;
              }
            }
          }

          delete unk;
          delete precals;
          return ndCdA / ndC - (ndA - ndCdA) / m_ndC;
        }

        // discrete attribute, discrete class
        else {
          float relf = 0.0;

          ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
            const int refVal = precals[rei->index];
            if (refVal == ILLEGAL_FLOAT)
              ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
                const int neiVal = precals[nei->index];
                relf += nei->weight * ((neiVal == ILLEGAL_INT) ? bothUnk : unk[neiVal]);
              }
            else {
              ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
                const int neiVal = precals[nei->index];
                relf += nei->weight * ((neiVal == ILLEGAL_INT) ? unk[refVal] : (refVal != neiVal ? 1.0 : 0.0));
              }
            }
          }

          delete unk;
          delete precals;
          return relf;
        }
      }
      catch (...) {
        delete unk;
        delete precals;
        throw;
      }
    }
  }
}



void TMeasureAttribute_relief::thresholdFunction(TFloatFloatList &res, PVariable var, PExampleGenerator gen, PDistribution, int weightID)
{
  TFunctionAdder divs;
  thresholdFunction(var, gen, divs, weightID);

  res.clear();
  float score = 0;
  for(TFunctionAdder::const_iterator di(divs.begin()), de(divs.end()); di != de; di++)
    res.push_back(make_pair(di->first, score += di->second));
}


float TMeasureAttribute_relief::bestThreshold(PDistribution &subsetSizes, float &bestScore, PVariable var, PExampleGenerator gen, PDistribution, int weightID, const float &minSubset)
{
  TFunctionAdder divs;
  int wins = 0;
  float score = 0.0, bestThreshold;
  TRandomGenerator rgen(gen->numberOfExamples());

  if (minSubset > 0) {
    float *attrVals;
    thresholdFunction(var, gen, divs, weightID, &attrVals);

    TContDistribution *valueDistribution;
    PDistribution wvd;

    if (attrVals) {
      try {
        float *vali = attrVals, *vale;
        wvd = valueDistribution = new TContDistribution(var);
        if (weightID)
          for(TExampleIterator ei(gen->begin()); ei; ++ei, vali++)
            if (*vali != ILLEGAL_FLOAT)
              valueDistribution->addfloat(*vali, WEIGHT(*ei));
        else
           for(vali = attrVals, vale = attrVals + gen->numberOfExamples(); vali != vale; vali++)
             if (*vali != ILLEGAL_FLOAT)
               valueDistribution->addfloat(*vali);
      }
      catch (...) {
        delete attrVals;
        throw;
      }

      delete attrVals;
      attrVals = NULL;
    }
    else {
      wvd = new TContDistribution(gen, var, weightID);
      valueDistribution = wvd.AS(TContDistribution);
    }

    float left = 0.0, right = valueDistribution->abs;
    float bestLeft, bestRight;

    map<float, float>::iterator distb(valueDistribution->begin()), diste(valueDistribution->end()), disti = distb, disti2;
    for(TFunctionAdder::const_iterator di(divs.begin()), de(divs.end()); di != de; di++) {
      score += di->second;
      if (!wins || (score > bestScore) || (score == bestScore) && rgen.randbool(++wins)) {
        for(; (disti != diste) && (disti->first <= di->first); disti++) {
          left += disti->second;
          right -= disti->second;
        }
        if ((left < minSubset))
          continue;
        if ((right < minSubset) || (disti == diste))
          break;
  
        if (!wins || (score > bestScore))
          wins = 1;
  
        bestScore = score;
        bestLeft = left;
        bestRight = right;

        // disti cannot be distb (contemplate the above for)
        disti2 = disti;
        bestThreshold = (disti->first + (--disti2)->first) / 2.0;
      }
    }

    if (!wins) {
      subsetSizes = NULL;
      return ILLEGAL_FLOAT;
    }

    subsetSizes = new TDiscDistribution(2);
    subsetSizes->addint(0, bestLeft);
    subsetSizes->addint(1, bestRight);
    return bestThreshold;
  }

  else {
    thresholdFunction(var, gen, divs, weightID);

    for(TFunctionAdder::const_iterator db(divs.begin()), de(divs.end()), di = db, di2; di != de; di++) {
      score += di->second;
      if (   (!wins || (score > bestScore)) && ((wins=1) == 1)
          || (score == bestScore) && rgen.randbool(++wins)) {
        di2 = di;
        bestThreshold = (++di2 == de) && (--di2 == db) ? di->first : (di->first + di2->first) / 2.0;
        bestScore = score;
      }
    }

    subsetSizes = NULL;
    return wins ? bestThreshold : ILLEGAL_FLOAT;
  }
}


PSymMatrix TMeasureAttribute_relief::gainMatrix(PVariable var, PExampleGenerator gen, PDistribution, int weightID, int **attrVals, float **attrDistr)
{
  TEnumVariable *evar = var.AS(TEnumVariable);
  if (!evar)
    raiseError("thresholdFunction can only be computed for continuous attributes");

  checkNeighbourhood(gen, weightID);

  TSymMatrix *gains = new TSymMatrix(evar->noOfValues());
  PSymMatrix wgains = gains;

  const int attrIdx = gen->domain->getVarNum(var, false);
  const bool regression = gen->domain->classVar->varType == TValue::FLOATVAR;

  if (attrIdx != ILLEGAL_INT) {
    if (attrVals)
      *attrVals = NULL;
    if (attrDistr)
      *attrDistr = NULL;

    const TExamplesDistance_Relief &rdistance = dynamic_cast<const TExamplesDistance_Relief &>(distance.getReference());
    const TExampleTable &table = dynamic_cast<const TExampleTable &>(gen.getReference());

    ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
      const TValue &refVal = table[rei->index][attrIdx];
      if (refVal.isSpecial())
        continue;
      const int &refValI = refVal.intV;

      ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
        const TValue &neiVal = table[nei->index][attrIdx];
        if (neiVal.isSpecial())
          continue;

        const float attrDist = rdistance(attrIdx, refVal, neiVal);
        if (regression) {
          const float dCdA = nei->weight * attrDist;
          const float dA = nei->weightEE * attrDist;
          gains->getref(refValI, neiVal.intV) += dCdA / ndC - (dA - dCdA) / m_ndC;
        }
        else
          gains->getref(refValI, neiVal.intV) += nei->weight * attrDist;
      }
    }
  }

  else {
    if (!var->getValueFrom)
      raiseError("attribute is not among the domain attributes and cannot be computed from them");

    float *unk, bothUnk;
    int *precals = tabulateDiscreteValues(gen, weightID, var.getReference(), unk, bothUnk);
    if (attrVals)
      *attrVals = precals;
    if (attrDistr) {
      const int noVal = evar->noOfValues();
      *attrDistr = new float[noVal];
      for(float *ai = *attrDistr, *ui = unk, *ue = unk + noVal; ui != ue; *ai++ = 1 - *ui++);
    }

    try {
      ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
        const int refValI = precals[rei->index];
        ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
          const int neiVal = precals[nei->index];
          const int attrDist = (refValI == ILLEGAL_INT) ? ((neiVal == ILLEGAL_INT) ? bothUnk : unk[neiVal])
                                                        : ((neiVal == ILLEGAL_INT) ? unk[refValI] : (refValI != neiVal ? 1.0 : 0.0));
          if (attrDist == 0.0)
            continue;
          if (regression) {
            const float dCdA = nei->weight * attrDist;
            const float dA = nei->weightEE * attrDist;
            gains->getref(refValI, neiVal) += dCdA / ndC - (dA - dCdA) / m_ndC;
          }
          else
            gains->getref(refValI, neiVal) += nei->weight * attrDist;
        }
      }

      delete unk;
      if (!attrVals)
        delete precals;
    }
    catch (...) {
      if (unk)
        delete unk;
      if (precals)
        delete precals;
      throw;
    }
  }

  return wgains;
}


PIntList TMeasureAttribute_relief::bestBinarization(PDistribution &subsetSizes, float &bestScore, PVariable var, PExampleGenerator gen, PDistribution apriorClass, int weightID, const float &minSubset)
{
  TEnumVariable *evar = var.AS(TEnumVariable);
  if (!evar)
    raiseError("cannot discretly binarize a continuous attribute");

  const int noVal = evar->noOfValues();
  if (noVal > 16)
    raiseError("cannot binarize an attribute with more than 16 values (it would take too long)");

  float *attrDistr = NULL;
  PSymMatrix wgain = gainMatrix(var, gen, apriorClass, weightID, NULL, &attrDistr);
  TSymMatrix &gain = wgain.getReference();

  float *gains = new float[noVal * noVal], *gi = gains, *ge;

  int wins = 0, bestSubset;
  float bestLeft, bestRight;

  try {
    float thisScore = 0.0;
    int i, j;
    for(i = 0; i < noVal; i++)
      for(j = 0; j < noVal; j++)
        *gi++ = gain.getitem(i, j);

    float thisLeft = 0.0, thisRight = 0.0;
    float *ai, *ae;
    if (!attrDistr) {
      TDiscDistribution dd(gen, var, weightID);
      attrDistr = new float[noVal];
      ai = attrDistr;
      ae = attrDistr + noVal;
      for(vector<float>::const_iterator di(dd.distribution.begin()); ai != ae; thisLeft += (*ai++ = *di++));
    }
    else
      for(ai = attrDistr, ae = attrDistr + noVal; ai != ae; thisLeft += *ai++);

    if (thisLeft < minSubset)
      return NULL;

    bestSubset = 0;
    wins = 0;
    bestLeft = thisLeft;
    bestRight = 0.0;
    bestScore = 0;

    TRandomGenerator rgen(gen->numberOfExamples());

    // if a bit in gray is 0, the corresponding value is on the left
    for(int cnt = (1 << (noVal-1)) - 1, gray = 0; cnt; cnt--) {
      int prevgray = gray;
      gray = cnt ^ (cnt >> 1);
      int graydiff = gray ^ prevgray;
      int diffed;
      for(diffed = 0; !(graydiff & 1); graydiff >>= 1, diffed++);

      if (gray > prevgray) { // something went to the right; subtract all the gains for being different from values on the right
        /* prevgray = gray; */   //  unneeded: they only differ in the bit representing this group
        for(gi = gains + diffed*noVal, ge = gi + noVal; gi != ge; thisScore += prevgray & 1 ? -*gi++ : *gi++, prevgray >>= 1);
        thisLeft -= attrDistr[diffed];
        thisRight += attrDistr[diffed];
      }
      else {
        /* prevgray = gray; */   //  unneeded: they only differ in the bit representing this group
        for(gi = gains + diffed*noVal, ge = gi + noVal; gi != ge; thisScore += prevgray & 1 ? *gi++ : -*gi++, prevgray >>= 1);
        thisLeft += attrDistr[diffed];
        thisRight -= attrDistr[diffed];
      }

      if (   (thisLeft >= minSubset) && (thisRight >= minSubset)
          && (   (!wins || (thisScore > bestScore)) && ((wins=1) == 1)
              || (thisScore == bestScore) && rgen.randbool(++wins))) {
        bestScore = thisScore;
        bestSubset = gray;
        bestLeft = thisLeft;
        bestRight = thisRight;
      }
    }

    delete gains;
    gains = NULL;

    if (!wins || !bestSubset) {
      delete attrDistr;
      return false;
    }
    
    ai = attrDistr;
    TIntList *rightSide = new TIntList();
    for(i = noVal; i--; bestSubset = bestSubset >> 1, ai++)
      rightSide->push_back(*ai > 0 ? bestSubset & 1 : -1);

    delete attrDistr;
    attrDistr = NULL;

    subsetSizes = new TDiscDistribution(2);
    subsetSizes->addint(0, bestLeft);
    subsetSizes->addint(1, bestRight);

    return rightSide;
  }
  catch (...) {
    if (gains)
      delete gains;
    if (attrDistr)
      delete attrDistr;
    throw;
  }
}




int TMeasureAttribute_relief::bestValue(PDistribution &subsetSizes, float &bestScore, PVariable var, PExampleGenerator gen, PDistribution apriorClass, int weightID, const float &minSubset)
{
  TEnumVariable *evar = var.AS(TEnumVariable);
  if (!evar)
    raiseError("cannot discretly binarize a continuous attribute");

  const int noVal = evar->noOfValues();

  float *attrDistr = NULL;
  PSymMatrix wgain = gainMatrix(var, gen, apriorClass, weightID, NULL, &attrDistr);
  TSymMatrix &gain = wgain.getReference();

  float *gains = new float[noVal * noVal], *gi = gains, *ge;

  int wins = 0;

  try {
    float thisScore = 0.0;
    int i, j;
    for(i = 0; i < noVal; i++)
      for(j = 0; j < noVal; j++)
        *gi++ = gain.getitem(i, j);

    float *ai, *ae;
    float nExamples;
    if (!attrDistr) {
      TDiscDistribution dd(gen, var, weightID);
      attrDistr = new float[noVal];
      ai = attrDistr;
      ae = attrDistr + noVal;
      for(vector<float>::const_iterator di(dd.distribution.begin()); ai != ae; *ai++ = *di++);
      nExamples = dd.abs;
    }
    else {
      nExamples = 0;
      for(ai = attrDistr, ae = attrDistr + noVal; ai != ae; nExamples += *ai++);
    }   
   
    float maxSubset = nExamples - minSubset;
    if (maxSubset < minSubset)
      return -1;

    int bestVal = -1;
    wins = 0;
    bestScore = 0;
    TRandomGenerator rgen(gen->numberOfExamples());
    float *gi = gains;
    ai = attrDistr;
    for(int thisValue = 0; thisValue < noVal; thisValue++, ai++) {
      if ((*ai < minSubset) || (*ai > maxSubset)) {
        gi += noVal;
        continue;
      }
      
      float thisScore = -2*gi[thisValue]; // have to subtract this, we'll add it once below
      for(ge = gi + noVal; gi != ge; thisScore += *gi++);
      if (    (!wins || (thisScore > bestScore)) && ((wins=1) == 1)
          || (thisScore == bestScore) && rgen.randbool(++wins)) {
        bestScore = thisScore;
        bestVal = thisValue;
      }
    }

    delete gains;
    gains = NULL;

    if (!wins) {
      delete attrDistr;
      return -1;
    }
    
    subsetSizes = new TDiscDistribution(2);
    subsetSizes->addint(0, nExamples - attrDistr[bestVal]);
    subsetSizes->addint(1, attrDistr[bestVal]);

    delete attrDistr;
    attrDistr = NULL;

    return bestVal;
  }
  catch (...) {
    if (gains)
      delete gains;
    if (attrDistr)
      delete attrDistr;
    throw;
  }
}
