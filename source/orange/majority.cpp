#include <math.h>

#include "random.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "distvars.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "classify.hpp"

#include "measures.hpp"
#include "estimateprob.hpp"
#include "cost.hpp"

#include "majority.ppp"


TMajorityLearner::TMajorityLearner()
{}


PClassifier TMajorityLearner::operator()(PExampleGenerator ogen, const int &weight)
{ if (!ogen->domain->classVar)
    raiseError("class-less domain");

  PDistribution classDistr = getClassDistribution(ogen, weight);
  if (estimatorConstructor)
    classDistr = estimatorConstructor->call(classDistr, aprioriDistribution, ogen, weight)->call();
    if (!classDistr)
      raiseError("invalid estimatorConstructor");
  else
    classDistr->normalize();

  return mlnew TDefaultClassifier(ogen->domain->classVar,
                                  classDistr->supportsContinuous ? TValue(classDistr->average()) : classDistr->highestProbValue(classDistr->cases),
                                  classDistr);
}

  
TCostLearner::TCostLearner(PCostMatrix acost)
: cost(acost)
{}


PClassifier TCostLearner::operator()(PExampleGenerator gen, const int &weight)
{ if (!gen->domain->classVar)
    raiseError("class-less domain");

  if (gen->domain->classVar->varType!=TValue::INTVAR)
    raiseError("cost-sensitive learning for continuous classes not supported");
  checkProperty(cost);
  
  PClassifier clsfr = TMajorityLearner::operator()(gen, weight);
  float missclassificationCost;
  TMeasureAttribute_cost(cost).majorityCost(clsfr.AS(TDefaultClassifier)->defaultDistribution,
                                                     missclassificationCost,
                                                     clsfr.AS(TDefaultClassifier)->defaultVal);
  return clsfr;
}



TRandomLearner::TRandomLearner()
{}


TRandomLearner::TRandomLearner(PDistribution dist)
: probabilities(dist)
{}


#include "basstat.hpp"
PClassifier TRandomLearner::operator()(PExampleGenerator gen, const int &weight)
{
  if (probabilities)
    return new TRandomClassifier(probabilities);

  PVariable &classVar = gen->domain->classVar;
  if (!classVar)
    raiseError("classless domain");

  if (classVar->varType == TValue::INTVAR)
    return new TRandomClassifier(getClassDistribution(gen, weight));

  if (classVar->varType == TValue::FLOATVAR) {
    TBasicAttrStat stat(gen, classVar, weight);
    return new TRandomClassifier(TGaussianDistribution(stat.avg, stat.dev));
  }

  raiseError("unsupported class type");
  return NULL;
}
