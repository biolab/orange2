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

#include <math.h>

#include "vars.hpp"
#include "domain.hpp"
#include "distvars.hpp"
#include "examples.hpp"
#include "examplegen.hpp"

#include "classify.hpp"
#include "contingency.hpp"
#include "estimateprob.hpp"
#include "calibrate.hpp"

#include "bayes.ppp"


TBayesLearner::TBayesLearner() 
: normalizePredictions(true),
  adjustThreshold(false)
{}


TBayesLearner::TBayesLearner(const TBayesLearner &old)
: TLearner(old),
  estimatorConstructor(old.estimatorConstructor),
  conditionalEstimatorConstructor(old.conditionalEstimatorConstructor),
  conditionalEstimatorConstructorContinuous(old.conditionalEstimatorConstructorContinuous),
  normalizePredictions(old.normalizePredictions),
  adjustThreshold(old.adjustThreshold)
{}



PClassifier TBayesLearner::operator()(PExampleGenerator gen, const int &weight)
{ if (!gen->domain->classVar)
    raiseError("class-less domain");
  if (gen->domain->classVar->varType != TValue::INTVAR)
    raiseError("discrete class attribute expected");

  PProbabilityEstimatorConstructor estConst = 
     estimatorConstructor ? estimatorConstructor
                          : PProbabilityEstimatorConstructor(mlnew TProbabilityEstimatorConstructor_relative());

  PConditionalProbabilityEstimatorConstructor condEstConst =
     conditionalEstimatorConstructor ? conditionalEstimatorConstructor 
                                     : PConditionalProbabilityEstimatorConstructor(mlnew TConditionalProbabilityEstimatorConstructor_ByRows(estConst));

  PConditionalProbabilityEstimatorConstructor condEstConstCont =
     conditionalEstimatorConstructorContinuous ? conditionalEstimatorConstructorContinuous
                                               : PConditionalProbabilityEstimatorConstructor(mlnew TConditionalProbabilityEstimatorConstructor_loess());

  PDomainContingency stat(mlnew TDomainContingency(gen, weight));

  PProbabilityEstimator estimator = estConst->call(stat->classes, PDistribution(), gen, weight);
  PDistribution distribution = estimator->call();
  if (distribution)
    estimator = PProbabilityEstimator();

  int i = 0;
  bool haveContingencies = false;
  bool haveEstimators = false;
  PDomainContingency condProbs = mlnew TDomainContingency();
  condProbs->classes = distribution;

  PConditionalProbabilityEstimatorList condProbEstList = mlnew TConditionalProbabilityEstimatorList();

  PITERATE(TDomainContingency, di, stat) {
    PConditionalProbabilityEstimator condEst;
    PContingency condProp;
    try {
      condEst = (((*di)->varType==TValue::FLOATVAR) ? condEstConstCont : condEstConst) ->call(*di, stat->classes, gen, weight, i++);
      condProp = condEst->call();
	}
	catch (mlexception err) {
      if (strcmp(err.what(), "'orange.ConditionalProbabilityEstimatorConstructor_loess': distribution (of attribute values, probably) is empty or has only a single value"))
        throw;
    }

    condProbs->push_back(condProp);
    if (condProbs) {
      condProbEstList->push_back(PConditionalProbabilityEstimator());
      haveContingencies = true;
    }
    else {
      condProbEstList->push_back(condEst);
      haveEstimators = true;
    }
  }

  // Remove the list of contingency or estimators, if they have no elements
  if (!haveContingencies && !haveEstimators)
    raiseWarning("invalid conditional probability or no attributes (the classifier will use apriori probabilities)");

  TBayesClassifier *classifier = mlnew TBayesClassifier(
      gen->domain, 
      distribution, haveContingencies ? condProbs : PDomainContingency(), 
      estimator, haveEstimators ? condProbEstList : PConditionalProbabilityEstimatorList(),
      normalizePredictions);

  PClassifier wclassifier(classifier);

  if (adjustThreshold && (gen->domain->classVar.AS(TEnumVariable)->values->size() == 2)) {
    float optCA;
    classifier->threshold = TThresholdCA()(wclassifier, gen, weight, optCA);
  }

  return wclassifier;
}



TBayesClassifier::TBayesClassifier(const bool &anP) 
: TClassifierFD(true),
  normalizePredictions(anP),
  threshold(0.5)
{}


TBayesClassifier::TBayesClassifier(PDomain dom, PDistribution dist, PDomainContingency dcont, PProbabilityEstimator pest, PConditionalProbabilityEstimatorList cpest, const bool &anP, const float &thresh) 
: TClassifierFD(dom, true),
  distribution(dist),
  conditionalDistributions(dcont),
  estimator(pest),
  conditionalEstimators(cpest),
  normalizePredictions(anP),
  threshold(thresh)
{}


PDistribution TBayesClassifier::classDistribution(const TExample &origexam)
{ checkProperty(domain)

  TExample exam = TExample(domain, origexam);
  TDiscDistribution *result = CLONE(TDiscDistribution, distribution);
  if (!result)
    raiseError("cannot return distribution of classes (non-discrete class and/or wrong type of probability estimator)");

  PDiscDistribution wresult = result;
  result->normalize();

  /* Ensure there is no division by zero: if a certain class probability is zero,
     we can divide it by anything.
     This is needed because if some value for some attribute is missing,
     the conditional class distribution will be 1/#classes for each class,
     so P(C|A)/P(C) would be (1/#classes) / 0... Now, it's (1/#classes) / 1. */
  TDiscDistribution *classDistDiv = CLONE(TDiscDistribution, distribution);
  PDistribution wclassDistDiv = classDistDiv;
  PITERATE(TDiscDistribution, ci, classDistDiv)
    if (*ci < 1e-20)
      *ci = 1.0;


  TDomainContingency::iterator dci, dce;
  bool dciOK = conditionalDistributions;
  if (dciOK) {
    dci = conditionalDistributions->begin();
    dce = conditionalDistributions->end();
  }

  TConditionalProbabilityEstimatorList::iterator cei, cee;
  bool ceiOK = conditionalEstimators;
  if (ceiOK) {
    cei = conditionalEstimators->begin();
    cee = conditionalEstimators->end();
  }

  TExample::const_iterator ei(exam.begin());

  for ( ; dciOK && (dci!=dce) || ceiOK && (cei!=cee); ei++) {
    if (!(*ei).isSpecial()) {
      
      // If we have a contingency, that's great
      if (dciOK && *dci) {
        PDistribution cp = (*dci)->p(*ei);
        if (cp->cases > 1e-6) {
          *result *= cp;
          *result /= *classDistDiv;
        }
      }

      else if (ceiOK && *cei) {
        PConditionalProbabilityEstimator cest = (*cei)->call(*ei);
        PDistribution dist = cest->call(*ei);

        // If the estimator can return distributions, that's OK
        if (dist) {
          *result *= dist;
        }

        // If not, we'll have to go class value by class value
        else {
          TValue classVal;
          TDiscDistribution nd(classVar);
          if (classVar->firstValue(classVal))
            do
              nd.set(classVal, cest->call(classVal, *ei));
            while (classVar->nextValue(classVal));

          *result *= nd;
          *result /= *classDistDiv;
        }
      }

      result->normalize();
    }
    if (dciOK)
      dci++;
    if (ceiOK)
      cei++;
  }

  /* Check for overflows (these occur when there are many attributes and
       P(C|A) is too often much higher then P(C) - for instance, when you
       have a minority class, but the example that is being classified
       is a strong example of this class */
  if (result->abs == numeric_limits<float>::infinity()) {
    for(TDiscDistribution::iterator di(result->begin()), de(result->end()); di != de; di++)
      *di = (*di==numeric_limits<float>::infinity()) ? 1.0 : 0.0;
  }

  return wresult;
}


TValue TBayesClassifier::operator ()(const TExample &exam)
{ 
  if (classVar.AS(TEnumVariable)->values->size() == 2)
    return TValue(classDistribution(exam)->atint(1) >= threshold ? 1 : 0);
  else
    return classDistribution(exam)->highestProbValue(exam);
}


void TBayesClassifier::predictionAndDistribution(const TExample &ex, TValue &val, PDistribution &classDist)
{ 
  classDist = classDistribution(ex);
  if (classVar.AS(TEnumVariable)->values->size() == 2)
    val = TValue(classDist->atint(1) >= threshold ? 1 : 0);
  else
    val = classDist->highestProbValue(ex);
}


/* We define this separately because there are cases when TBayesClassifier::classDistribution
   cannot work while this can. Concretely, this happens when the (unconditional) estimator
   was not able to provide class distributions (i.e. this->distribution is NULL).
   This would usually occur for continuous classes. */

float TBayesClassifier::p(const TValue &classValue, const TExample &origexam)
{ TExample exam = TExample(domain, origexam);
  float res, c;
  if (distribution)
    c = res = distribution->p(classValue);
  if (!res)
    return 0.0;

  TDomainContingency::iterator dci, dce;
  bool dciOK = conditionalDistributions;
  if (dciOK) {
    dci = conditionalDistributions->begin();
    dce = conditionalDistributions->end();
  }

  TConditionalProbabilityEstimatorList::iterator cei, cee;
  bool ceiOK = conditionalEstimators;
  if (ceiOK) {
    cei = conditionalEstimators->begin();
    cee = conditionalEstimators->end();
  }

  TExample::const_iterator ei(exam.begin());
  for ( ; dciOK && (dci!=dce) || ceiOK && (cei!=cee); ei++) {
    if (!(*ei).isSpecial()) {
      if (dciOK && *dci) {
        PDistribution cp = (*dci)->p(*ei);
        if (cp->cases > 1e-6) {
          res *= (*dci)->p(*ei)->p(classValue)/c;
        }
      }
      else if (ceiOK && *cei) 
        res *= (*cei)->call(classValue, *ei)/c;
    }
    if (dciOK)
      dci++;
    if (ceiOK)
      cei++;
  }

  return res;
}
