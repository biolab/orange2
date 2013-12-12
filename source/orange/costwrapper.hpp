#ifndef __COSTWRAPPER_HPP
#define __COSTWRAPPER_HPP

#include "learn.hpp"
#include "classify.hpp"

WRAPPER(CostMatrix)

class ORANGE_API TCostWrapperLearner : public TLearner {
public:
  __REGISTER_CLASS

  PLearner basicLearner; //P(+base_learner) basic learner
  PCostMatrix costMatrix; //P cost matrix

  TCostWrapperLearner(PCostMatrix =PCostMatrix(), PLearner = PLearner());

  virtual PClassifier operator()(PExampleGenerator gen, const int & =0);
};


class ORANGE_API TCostWrapperClassifier : public TClassifier {
public:
  __REGISTER_CLASS

  PClassifier classifier; //P basic classifier
  PCostMatrix costMatrix; //P cost matrix

  TCostWrapperClassifier(PCostMatrix =PCostMatrix(), PClassifier =PClassifier());

  virtual TValue operator()(const TExample &);
  virtual TValue operator ()(PDiscDistribution risks);

  virtual PDiscDistribution getRisks(const TExample &);
  virtual PDiscDistribution getRisks(PDistribution);
};

#endif
