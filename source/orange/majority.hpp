#ifndef __MAJORITY_HPP
#define __MAJORITY_HPP

#include "learn.hpp"

WRAPPER(Distribution)
WRAPPER(ProbabilityEstimator);
WRAPPER(CostMatrix)

class ORANGE_API TMajorityLearner : public TLearner {
public:
  __REGISTER_CLASS

  PProbabilityEstimatorConstructor estimatorConstructor; //P constructs probability estimator
  PDistribution aprioriDistribution; //P apriori class distribution

  TMajorityLearner();
  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};



class ORANGE_API TCostLearner : public TMajorityLearner {
public:
  __REGISTER_CLASS

  PCostMatrix cost; //P cost matrix

  TCostLearner(PCostMatrix = PCostMatrix());
  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};



class ORANGE_API TRandomLearner : public TLearner {
public:
  __REGISTER_CLASS

  PDistribution probabilities; //P probabilities of predictions

  TRandomLearner();
  TRandomLearner(PDistribution);
  
  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};

#endif
