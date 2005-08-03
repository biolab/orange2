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

    Authors: Martin Mozina, Janez Demsar, Blaz Zupan, 1996--2004
    Contact: martin.mozina@fri.uni-lj.si
*/

#include "filter.hpp"
#include "table.hpp"
#include "stat.hpp"
#include "measures.hpp"
#include "discretize.hpp"
#include "distvars.hpp"
#include "classfromvar.hpp"

#include "rulelearner.ppp"


DEFINE_TOrangeVector_classDescription(PRule, "TRuleList", true, ORANGE_API)


TRule::TRule()
: weightID(0),
  quality(ILLEGAL_FLOAT),
  complexity(ILLEGAL_FLOAT),
  coveredExamples(NULL),
  coveredExamplesLength(-1)
{}


TRule::TRule(PFilter af, PClassifier cl, PLearner lr, PDistribution dist, PExampleTable ce, const int &w, const float &qu)
: filter(af),
  classifier(cl),
  learner(lr),
  classDistribution(dist),
  examples(ce),
  weightID(w),
  quality(qu),
  coveredExamples(NULL),
  coveredExamplesLength(-1)
{}


TRule::TRule(const TRule &other, bool copyData)
: filter(other.filter? other.filter->deepCopy() : PFilter()),
  classifier(other.classifier),
  learner(other.learner),
  complexity(other.complexity),
  classDistribution(copyData ? other.classDistribution: PDistribution()),
  examples(copyData ? other.examples : PExampleTable()),
  weightID(copyData ? other.weightID : 0),
  quality(copyData ? other.quality : ILLEGAL_FLOAT),
  coveredExamples(copyData && other.coveredExamples && (other.coveredExamplesLength >= 0) ? (int *)memcpy(new int[other.coveredExamplesLength], other.coveredExamples, other.coveredExamplesLength) : NULL),
  coveredExamplesLength(copyData ? other.coveredExamplesLength : -1)
{}


TRule::~TRule()
{ delete coveredExamples; }

bool TRule::operator ()(const TExample &ex)
{
  checkProperty(filter);
  return filter->call(ex);
}


#define HIGHBIT 0x80000000

PExampleTable TRule::operator ()(PExampleTable gen, const bool ref, const bool negate)
{
  checkProperty(filter);

  TExampleTable *table = ref ? mlnew TExampleTable(gen, 1) : mlnew TExampleTable(PExampleGenerator(gen));
  PExampleGenerator wtable = table;

  PEITERATE(ei, gen)
    if (filter->call(*ei) != negate)
      table->addExample(*ei);

  return wtable;
}


void TRule::filterAndStore(PExampleTable gen, const int &wei, const int &targetClass, const int *prevCovered, const int anExamples)
{
  checkProperty(filter);
  examples=this->call(gen);
  weightID = wei;
  classDistribution = getClassDistribution(examples, wei);
  if (classDistribution->abs==0)
    return;

  if (learner) {
    classifier = learner->call(examples,wei);
  }
  else if (targetClass>=0)
    classifier = mlnew TDefaultClassifier(gen->domain->classVar, TValue(targetClass), classDistribution);
  else
    classifier = mlnew TDefaultClassifier(gen->domain->classVar, classDistribution); 
/*  if (anExamples > 0) {
    const int bitsInInt = sizeof(int)*8;
    coveredExamplesLength = anExamples/bitsInInt + 1;
    coveredExamples = (int *)malloc(coveredExamplesLength);
    if (prevCovered) {
      memcpy(coveredExamples, prevCovered, coveredExamplesLength);

      int *cei = coveredExamples-1;
      int mask = 0;
      int inBit = 0;

      PEITERATE(ei, gen) {
        if (!(*cei & mask)) {
          if (inBit)
            *cei = *cei << inBit;
          while(!*++cei);
          mask = -1;
          inBit = bitsInInt;
        }

        while( (*cei & HIGHBIT) == 0) {
          *cei = *cei << 1;
          *cei = mask << 1;
          inBit--;
        }

        if (filter->call(*ei)) {
          *cei = (*cei << 1) | 1;
          table->addExample(*ei);
        }
        else
          *cei = *cei << 1;

        mask = mask << 1;
        inBit--;
      }
    }

    else {
      int *cei = coveredExamples;
      int inBit = bitsInInt;

      PEITERATE(ei, gen) {
        if (filter->call(*ei)) {
          *cei = (*cei << 1) | 1;
          table->addExample(*ei);
        }
        else
          *cei = *cei << 1;

        if (!--inBit) {
          inBit = bitsInInt;
          cei++;
        }
      }
      *cei = *cei << inBit;
    }
  } */
}



bool haveEqualValues(const TRule &r1, const TRule &r2)
{
  const TDefaultClassifier *clsf1 = r1.classifier.AS(TDefaultClassifier);
  const TDefaultClassifier *clsf2 = r2.classifier.AS(TDefaultClassifier);
  if (!clsf1 || !clsf2)
    return false;

  const TDiscDistribution *dist1 = dynamic_cast<const TDiscDistribution *>(clsf1->defaultDistribution.getUnwrappedPtr());
  const TDiscDistribution *dist2 = dynamic_cast<const TDiscDistribution *>(clsf2->defaultDistribution.getUnwrappedPtr());

  float high1 = dist1->highestProb();
  float high2 = dist2->highestProb();

  for(TDiscDistribution::const_iterator d1i(dist1->begin()), d1e(dist1->end()), d2i(dist2->begin()), d2e(dist2->end());
      (d1i!=d1e) && (d2i!=d2e);
      d1i++, d2i++)
    if ((*d1i == high1) && (*d2i == high2))
      return true;

  return false;
}


bool TRule::operator <(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  bool different = false;
  
  if (coveredExamples && other.coveredExamples) {
    int *c1i = coveredExamples;
    int *c2i = other.coveredExamples;
    for(int i = coveredExamplesLength; i--; c1i++, c2i++) {
      if (*c1i & ~*c2i)
        return false;
      if (*c1i != *c2i)
        different = true;
    }
  }
  else {
    raiseError("operator not implemented yet");
  }

  return different;
}   


bool TRule::operator <=(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  if (coveredExamples && other.coveredExamples) {
    int *c1i = coveredExamples;
    int *c2i = other.coveredExamples;
    for(int i = coveredExamplesLength; i--; c1i++, c2i++) {
      if (*c1i & ~*c2i)
        return false;
    }
  }

  else {
    raiseError("operator not implemented yet");
  }

  return true;
}


bool TRule::operator >(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  bool different = false;
  if (coveredExamples && other.coveredExamples) {
    int *c1i = coveredExamples;
    int *c2i = other.coveredExamples;
    for(int i = coveredExamplesLength; i--; c1i++, c2i++) {
      if (~*c1i & *c2i)
        return false;
      if (*c1i != *c2i)
        different = true;
    }
  }

  else {
    raiseError("operator not implemented yet");
  }

  return different;
}   


bool TRule::operator >=(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  if (coveredExamples && other.coveredExamples) {
    int *c1i = coveredExamples;
    int *c2i = other.coveredExamples;
    for(int i = coveredExamplesLength; i--; c1i++, c2i++) {
      if (~*c1i & *c2i)
        return false;
    }
  }

  else {
    raiseError("operator not implemented yet");
  }

  return true;
}


bool TRule::operator ==(const TRule &other) const
{
  if (!haveEqualValues(*this, other))
    return false;

  if (coveredExamples && other.coveredExamples) {
    return !memcmp(coveredExamples, other.coveredExamples, coveredExamplesLength);
  }

  else {
    raiseError("operator not implemented yet");
  }

  return false;
}



TRuleValidator_LRS::TRuleValidator_LRS(const float &a, const float &min_coverage, const float &max_rule_complexity)
: alpha(a),
  min_coverage(min_coverage),
  max_rule_complexity(max_rule_complexity)
{}

bool TRuleValidator_LRS::operator()(PRule rule, PExampleTable, const int &, const int &targetClass, PDistribution apriori) const
{
  const TDiscDistribution &obs_dist = dynamic_cast<const TDiscDistribution &>(rule->classDistribution.getReference());
  if (!obs_dist.cases)
    return false;
  
  if (min_coverage>0.0 && obs_dist.cases < min_coverage)
    return false;

  if (max_rule_complexity > 0.0 && rule->complexity > max_rule_complexity)
    return false;


  const TDiscDistribution &exp_dist = dynamic_cast<const TDiscDistribution &>(apriori.getReference());

  if (targetClass == -1) {
    float lrs = 0.0;
    for(TDiscDistribution::const_iterator odi(obs_dist.begin()), ode(obs_dist.end()), edi(exp_dist.begin()), ede(exp_dist.end());
        (odi!=ode); odi++, edi++) {
      if ((edi!=ede) && (*edi) && (*odi))
        lrs += *odi * log(*odi / ((edi != ede) & (*edi > 0.0) ? *edi : 1e-5));
    }

    lrs = 2 * (lrs - obs_dist.abs * log(obs_dist.abs / exp_dist.abs));

    return (lrs > 0.0) && (chisqprob(lrs, float(obs_dist.size()-1)) <= alpha);
  }

  float p = (targetClass < obs_dist.size()) ? obs_dist[targetClass] : 1e-5;
  const float P = (targetClass < exp_dist.size()) && (exp_dist[targetClass] > 0.0) ? exp_dist[targetClass] : 1e-5;

  float n = obs_dist.abs - p;
  float N = exp_dist.abs - P;

  if (N<=0.0)
    N = 1e-6f;
  if (p<=0.0)
    p = 1e-6f;
  if (n<=0.0)
    n = 1e-6f;


  
  float lrs = 2 * (p*log(p/P) + n*log(n/N) - obs_dist.abs * log(obs_dist.abs/exp_dist.abs));

  return (lrs > 0.0) && (chisqprob(lrs, 1.0f) <= alpha);
}


float TRuleEvaluator_Entropy::operator()(PRule rule, PExampleTable, const int &, const int &targetClass, PDistribution apriori) const
{
  if (targetClass == -1)
    return -getEntropy(dynamic_cast<TDiscDistribution &>(rule->classDistribution.getReference()));

  const TDiscDistribution &exp_dist = dynamic_cast<const TDiscDistribution &>(apriori.getReference());

  const TDiscDistribution &obs_dist = dynamic_cast<const TDiscDistribution &>(rule->classDistribution.getReference());
  if (!obs_dist.cases)
    return false;

  float p = (targetClass < obs_dist.size()) ? obs_dist[targetClass] : 0.0;
  const float P = (targetClass < exp_dist.size()) && (exp_dist[targetClass] > 0.0) ? exp_dist[targetClass] : 1e-5;

  float n = obs_dist.abs - p;
  float N = exp_dist.abs - P;
  if (N<=0.0)
    N = 1e-6f;
  if (p<=0.0)
    p = 1e-6f;
  if (n<=0.0)
    n = 1e-6f;

  return ((p*log(p) + n*log(n) - obs_dist.abs * log(obs_dist.abs)) / obs_dist.abs);
}

float TRuleEvaluator_Laplace::operator()(PRule rule, PExampleTable, const int &, const int &targetClass, PDistribution apriori) const
{
  const TDiscDistribution &obs_dist = dynamic_cast<const TDiscDistribution &>(rule->classDistribution.getReference());
  if (!obs_dist.cases)
    return 0;

  float p;
  if (targetClass == -1) {
    p = float(obs_dist.highestProb());
    return (p+1)/(obs_dist.abs+obs_dist.size());
  }
  p = float(obs_dist[targetClass]);
  return (p+1)/(obs_dist.abs+2);
}


bool worstRule(const PRule &r1, const PRule &r2)
{ return    (r1->quality > r2->quality) 
          || (r1->quality==r2->quality 
          && r1->complexity < r2->complexity);
}
/*         || (r1->quality==r2->quality) 
            && (   (r1->complexity < r2->complexity)
                || (r1->complexity == r2->complexity) 
                   && ((int(r1.getUnwrappedPtr()) ^ int(r2.getUnwrappedPtr())) & 16) != 0
               ); }  */

TRuleBeamFilter_Width::TRuleBeamFilter_Width(const int &w)
: width(w)
{}


void TRuleBeamFilter_Width::operator()(PRuleList rules, PExampleTable, const int &)
{
  if (rules->size() > width) {
    // Janez poglej
    TRuleList::const_iterator ri(rules->begin()), re(rules->end());
    sort(rules->begin(), rules->end(), worstRule);
    rules->erase(rules->begin()+width, rules->end());
  }
}


inline void _selectBestRule(PRule &rule, PRule &bestRule, int &wins, TRandomGenerator &rgen)
{
  if ((rule->quality > bestRule->quality) || (rule->complexity < bestRule->complexity)) {
    bestRule = rule;
    wins = 1;
  }
  else if ((rule->complexity == bestRule->complexity) && rgen.randbool(++wins))
    bestRule = rule;
}



PRuleList TRuleBeamInitializer_Default::operator()(PExampleTable data, const int &weightID, const int &targetClass, PRuleList baseRules, PRuleEvaluator evaluator, PDistribution apriori, PRule &bestRule)
{
  checkProperty(evaluator);

  TRuleList *ruleList = mlnew TRuleList();
  PRuleList wruleList = ruleList;

  TRandomGenerator rgen(data->numberOfExamples());
  int wins;

  if (baseRules && baseRules->size())
    PITERATE(TRuleList, ri, baseRules) {
      TRule *newRule = mlnew TRule((*ri).getReference(), false);
      PRule wNewRule = newRule;
      ruleList->push_back(wNewRule);
      newRule->filterAndStore(data,weightID,targetClass);
      newRule->quality = evaluator->call(wNewRule, data, weightID, targetClass, apriori);
      if (!bestRule || (newRule->quality > bestRule->quality)) {
        bestRule = wNewRule;
        wins = 1;
      }
      else 
        if (newRule->quality == bestRule->quality)
          _selectBestRule(wNewRule, bestRule, wins, rgen);
    }

  else {
     TRule *ubestRule = mlnew TRule();
     bestRule = ubestRule;
     ruleList->push_back(bestRule);
     ubestRule->filter = new TFilter_values();
     ubestRule->filter->domain = data->domain;
     ubestRule->filterAndStore(data, weightID,targetClass);
     ubestRule->complexity = 0;
  }

  return wruleList;
}


PRuleList TRuleBeamRefiner_Selector::operator()(PRule wrule, PExampleTable data, const int &weightID, const int &targetClass)
{
  if (!discretization) {
    discretization = mlnew TEntropyDiscretization();
    dynamic_cast<TEntropyDiscretization *>(discretization.getUnwrappedPtr())->forceAttribute = true;
  }

  TRule &rule = wrule.getReference();
  TFilter_values *filter = wrule->filter.AS(TFilter_values);
  if (!filter)
    raiseError("a filter of type 'Filter_values' expected");

  TRuleList *ruleList = mlnew TRuleList;
  PRuleList wRuleList = ruleList;

  TDomainDistributions ddist(wrule->examples, wrule->weightID);

  const TVarList &attributes = rule.examples->domain->attributes.getReference();

  vector<bool> used(attributes.size(), false);
  PITERATE(TValueFilterList, vfi, filter->conditions)
    used[(*vfi)->position] = true;

  vector<bool>::const_iterator ui(used.begin());
  TDomainDistributions::const_iterator di(ddist.begin());
  TVarList::const_iterator vi(attributes.begin()), ve(attributes.end());
  int pos = 0;
  for(; vi != ve; vi++, ui++, pos++, di++) {
    if ((*vi)->varType == TValue::INTVAR) {
      if (!*ui) {
        vector<float>::const_iterator idi((*di).AS(TDiscDistribution)->begin());
        for(int v = 0, e = (*vi)->noOfValues(); v != e; v++)
          if (*idi>0) {
            TRule *newRule = mlnew TRule(rule, false);
            ruleList->push_back(newRule);
            newRule->complexity++;

            filter = newRule->filter.AS(TFilter_values);

            TValueFilter_discrete *newCondition = mlnew TValueFilter_discrete(pos, *vi, 0);
            filter->conditions->push_back(newCondition);

            TValue value = TValue(v);
            newCondition->values->push_back(value);
            newRule->filterAndStore(rule.examples, rule.weightID,targetClass);
          }
      }
    }

    else if (((*vi)->varType == TValue::FLOATVAR)) {
      if (discretization) {
        PVariable discretized;
        try {
        discretized = discretization->call(rule.examples, *vi, weightID);
        } catch(...) {
          continue;
        }
        TClassifierFromVar *cfv = discretized->getValueFrom.AS(TClassifierFromVar);
        TDiscretizer *discretizer = cfv ? cfv->transformer.AS(TDiscretizer) : NULL;
        if (!discretizer)
          raiseError("invalid or unrecognized discretizer");

        vector<float> cutoffs;
        discretizer->getCutoffs(cutoffs);
        if (cutoffs.size()) {
          TRule *newRule;
          newRule = mlnew TRule(rule, false);
          PRule wnewRule = newRule;
          newRule->complexity++;

          newRule->filter.AS(TFilter_values)->conditions->push_back(mlnew TValueFilter_continuous(pos,  TValueFilter_continuous::LessEqual, cutoffs.front(), 0, 0));
          newRule->filterAndStore(rule.examples, rule.weightID,targetClass);
          if (wrule->classDistribution->cases > wnewRule->classDistribution->cases)
            ruleList->push_back(newRule);

          for(vector<float>::const_iterator ci(cutoffs.begin()), ce(cutoffs.end()-1); ci != ce; ci++) {
            newRule = mlnew TRule(rule, false);
            PRule wnewRule = newRule;
            newRule->complexity++;

            filter = newRule->filter.AS(TFilter_values);
            filter->conditions->push_back(mlnew TValueFilter_continuous(pos,  TValueFilter_continuous::Greater, *ci, 0, 0));
            filter->conditions->push_back(mlnew TValueFilter_continuous(pos,  TValueFilter_continuous::LessEqual, *(ci+1), 0, 0));
            newRule->filterAndStore(rule.examples, rule.weightID,targetClass);
            if (wrule->classDistribution->cases > wnewRule->classDistribution->cases)
              ruleList->push_back(newRule);
          }

          newRule = mlnew TRule(rule, false);
          ruleList->push_back(newRule);
          newRule->complexity++;

          newRule->filter.AS(TFilter_values)->conditions->push_back(mlnew TValueFilter_continuous(pos,  TValueFilter_continuous::Greater, cutoffs.back(), 0, 0));
          newRule->filterAndStore(rule.examples, rule.weightID,targetClass);
        }
      }
      else
        raiseWarning("discretizer not given, continuous attributes will be skipped");
    }
  }
  if (!discretization)
    discretization = PDiscretization();
  return wRuleList;
}


PRuleList TRuleBeamCandidateSelector_TakeAll::operator()(PRuleList existingRules, PExampleTable, const int &)
{
  PRuleList candidates = mlnew TRuleList(existingRules.getReference());
//  existingRules->clear();
  existingRules->erase(existingRules->begin(), existingRules->end());
  return candidates;
}


PRule TRuleBeamFinder::operator()(PExampleTable data, const int &weightID, const int &targetClass, PRuleList baseRules)
{
  // set default values if value not set
  bool tempInitializer = !initializer;
  if (tempInitializer)
    initializer = mlnew TRuleBeamInitializer_Default;
  bool tempCandidateSelector = !candidateSelector;
  if (tempCandidateSelector)
    candidateSelector = mlnew TRuleBeamCandidateSelector_TakeAll;
  bool tempRefiner = !refiner;
  if (tempRefiner)
    refiner = mlnew TRuleBeamRefiner_Selector;
/*  bool tempValidator = !validator;
  if (tempValidator) 
    validator = mlnew TRuleValidator_LRS((float)0.01);
  bool tempRuleStoppingValidator = !ruleStoppingValidator;
  if (tempRuleStoppingValidator) 
    ruleStoppingValidator = mlnew TRuleValidator_LRS((float)0.05); */
  bool tempEvaluator = !evaluator;
  if (tempEvaluator)
    evaluator = mlnew TRuleEvaluator_Entropy;
  bool tempRuleFilter = !ruleFilter;
  if (tempRuleFilter)
    ruleFilter = mlnew TRuleBeamFilter_Width;

  checkProperty(initializer);
  checkProperty(candidateSelector);
  checkProperty(refiner);
  checkProperty(evaluator);
  checkProperty(ruleFilter);
  checkProperty(ruleStoppingValidator);

  PDistribution apriori = getClassDistribution(data, weightID);

  TRandomGenerator rgen(data->numberOfExamples());
  int wins = 1;

  PRule bestRule;
  PRuleList ruleList = initializer->call(data, weightID, targetClass, baseRules, evaluator, apriori, bestRule);

  {
  PITERATE(TRuleList, ri, ruleList) {
    if (!(*ri)->examples)
      (*ri)->filterAndStore(data, weightID,targetClass);
    if ((*ri)->quality == ILLEGAL_FLOAT)
      (*ri)->quality = evaluator->call(*ri, data, weightID, targetClass, apriori);
  }
  }

  if (!bestRule->examples)
    bestRule->filterAndStore(data, weightID,targetClass);
  if (bestRule->quality == ILLEGAL_FLOAT)
    bestRule->quality = evaluator->call(bestRule, data, weightID, targetClass, apriori);

  int bestRuleLength = 0;
  while(ruleList->size()) {
    PRuleList candidateRules = candidateSelector->call(ruleList, data, weightID);
    if (ruleList->size()>0)
      raiseWarning("ruleList length to large.");
    PITERATE(TRuleList, ri, candidateRules) {
      PRuleList newRules = refiner->call(*ri, data, weightID, targetClass);      
      PITERATE(TRuleList, ni, newRules) {
        if (!ruleStoppingValidator || ruleStoppingValidator->call(*ni, data, weightID, targetClass, apriori)) {
          (*ni)->quality = evaluator->call(*ni, data, weightID, targetClass, apriori);
          ruleList->push_back(*ni);
          if ((*ni)->quality >= bestRule->quality && (!validator || validator->call(*ni, data, weightID, targetClass, apriori)))
            _selectBestRule(*ni, bestRule, wins, rgen);
        }
      }  
    } 
    ruleFilter->call(ruleList,data,weightID);
  }  

  // set empty values if value was not set (used default)
  if (tempInitializer)
    initializer = PRuleBeamInitializer();
  if (tempCandidateSelector)
    candidateSelector = PRuleBeamCandidateSelector();
  if (tempRefiner)
    refiner = PRuleBeamRefiner();
/*  if (tempValidator)
    validator = PRuleValidator();
  if (tempRuleStoppingValidator)
    ruleStoppingValidator = PRuleValidator();  */
  if (tempEvaluator)
    evaluator = PRuleEvaluator();
  if (tempRuleFilter)
    ruleFilter = PRuleBeamFilter();

  return bestRule;
}


TRuleLearner::TRuleLearner(bool se, int tc, PRuleList rl)
: storeExamples(se),
  targetClass(tc),
  baseRules(rl)
{}


PClassifier TRuleLearner::operator()(PExampleGenerator gen, const int &weightID)
{
  return this->call(gen,weightID,targetClass,baseRules);
}

PClassifier TRuleLearner::operator()(PExampleGenerator gen, const int &weightID, const int &targetClass, PRuleList baseRules)
{
  // Initialize default values if values not set
  bool tempDataStopping = !dataStopping && !ruleStopping;
  if (tempDataStopping) 
    dataStopping = mlnew TRuleDataStoppingCriteria_NoPositives;

  bool tempRuleFinder = !ruleFinder;
  if (tempRuleFinder)
    ruleFinder = mlnew TRuleBeamFinder;

  bool tempCoverAndRemove = !coverAndRemove;
  if (tempCoverAndRemove)
    coverAndRemove = mlnew TRuleCovererAndRemover_Default;

  checkProperty(ruleFinder);
  checkProperty(coverAndRemove);

  TExampleTable *data = mlnew TExampleTable(gen);
  PExampleTable wdata = data;

  if (!dataStopping && !ruleStopping)
    raiseError("no stopping criteria; set 'dataStopping' and/or 'ruleStopping'");

  TRuleList *ruleList = mlnew TRuleList;
  PRuleList wruleList = ruleList;

  int currWeightID = weightID;

  while (!dataStopping || !dataStopping->call(wdata, currWeightID, targetClass)) {
    PRule rule = ruleFinder->call(wdata, currWeightID, targetClass, baseRules);
    if (!rule)
      raiseError("'ruleFinder' didn't return a rule");

    if (ruleStopping && ruleStopping->call(ruleList, rule, wdata, currWeightID))
      break;

    wdata = coverAndRemove->call(rule, wdata, currWeightID, currWeightID, targetClass);
    ruleList->push_back(rule);
  } 

  // Restore values
  if (tempDataStopping) 
    dataStopping = PRuleDataStoppingCriteria();
  if (tempRuleFinder)
    ruleFinder = PRuleFinder();
  if (tempCoverAndRemove)
    coverAndRemove = PRuleCovererAndRemover();

  PRuleClassifierConstructor clConstructor = 
    classifierConstructor ? classifierConstructor : 
    PRuleClassifierConstructor(mlnew TRuleClassifierConstructor_firstRule());
  return clConstructor->call(ruleList, gen, weightID);
};


bool TRuleDataStoppingCriteria_NoPositives::operator()(PExampleTable data, const int &weightID, const int &targetClass) const
{
  PDistribution classDist = getClassDistribution(data, weightID);
  TDiscDistribution *ddist = classDist.AS(TDiscDistribution);

  return (targetClass >= 0 ? ddist->atint(targetClass) : ddist->abs) == 0.0;
}

bool TRuleStoppingCriteria_NegativeDistribution::operator()(PRuleList ruleList, PRule rule, PExampleTable data, const int &weightID) const
{
  if (rule && rule->classifier) 
  {
    const TDefaultClassifier *clsf = rule->classifier.AS(TDefaultClassifier);
    if (!clsf)
      return false;
    const TDiscDistribution *dist = dynamic_cast<const TDiscDistribution *>(clsf->defaultDistribution.getUnwrappedPtr());
    const int classVal = clsf->defaultVal.intV;
    if (classVal<0 || classVal>=dist->size())
      return false;
    float defProb = dist->atint(clsf->defaultVal.intV);

    for(TDiscDistribution::const_iterator di(dist->begin()), de(dist->end());
        (di!=de); di++)
      if ((*di > defProb))
        return true;
  }
  return false;
}


PExampleTable TRuleCovererAndRemover_Default::operator()(PRule rule, PExampleTable data, const int &weightID, int &newWeight, const int &targetClass) const
{
  TExampleTable *table = mlnew TExampleTable(data, 1);
  PExampleGenerator wtable = table;

  TFilter &filter = rule->filter.getReference();

  if (targetClass < 0)
  {
    PEITERATE(ei, data)
      if (!filter(*ei))
        table->addExample(*ei);
  }
  else 
    PEITERATE(ei, data)
      if (!filter(*ei) || (*ei).getClass().intV!=targetClass)
        table->addExample(*ei);


  newWeight = weightID;
  return wtable;  
}

// classifiers
PRuleClassifier TRuleClassifierConstructor_firstRule::operator ()(PRuleList rules, PExampleTable table, const int &weightID)
{
  return mlnew TRuleClassifier_firstRule(rules, table, weightID);
}


TRuleClassifier::TRuleClassifier(PRuleList arules, PExampleTable anexamples, const int &aweightID)
: rules(arules),
  examples(anexamples),
  weightID(aweightID)
{}

TRuleClassifier::TRuleClassifier()
{}


TRuleClassifier_firstRule::TRuleClassifier_firstRule(PRuleList arules, PExampleTable anexamples, const int &aweightID)
: TRuleClassifier(arules, anexamples, aweightID)
{
  prior = getClassDistribution(examples, weightID);
}

TRuleClassifier_firstRule::TRuleClassifier_firstRule()
: TRuleClassifier()
{}

PDistribution TRuleClassifier_firstRule::classDistribution(const TExample &ex)
{
  checkProperty(rules);
  checkProperty(prior);

  PITERATE(TRuleList, ri, rules) {
    if ((*ri)->call(ex))
      return (*ri)->classDistribution;
  }
  return prior;
}
