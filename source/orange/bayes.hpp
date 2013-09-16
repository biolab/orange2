#ifndef __BAYES_HPP
#define __BAYES_HPP

#include "classify.hpp"
#include "estimateprob.hpp"
#include "learn.hpp"

WRAPPER(DomainContingency);
WRAPPER(ProbabilityEstimator);

class ORANGE_API TBayesLearner : public TLearner {
public:
  __REGISTER_CLASS

  PProbabilityEstimatorConstructor estimatorConstructor;                                 //P constructs a probability estimator for P(C)
  PConditionalProbabilityEstimatorConstructor conditionalEstimatorConstructor;           //P constructs a probability estimator for P(C|A) 
  PConditionalProbabilityEstimatorConstructor conditionalEstimatorConstructorContinuous; //P constructs a probability estimator for P(C|A) for continuous attributes
  bool normalizePredictions;  //P instructs learner to construct a classifier that normalizes probabilities
  bool adjustThreshold; //P adjust probability thresholds (for binary classes only)

  TBayesLearner();
  TBayesLearner(const TBayesLearner &old);

  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};


/* Naive Bayesian classifier.
   If classDistribution is given, it is used; if not, estimator is called to get class probabilities.
   Further, conditionalDistributions are used for attributes for which they are defined; for others,
   a corresponding conditionalEstimator is called. */
class ORANGE_API TBayesClassifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  PDistribution distribution; //P class distributions (P(C))
  PDomainContingency conditionalDistributions; //P conditional distributions, P(C|A)

  PProbabilityEstimator estimator; //P a probability estimator for P(C)
  PConditionalProbabilityEstimatorList conditionalEstimators; //P a probability estimator for P(C|A)
  bool normalizePredictions; //P if true, classifier will normalize predictions
  float threshold; //P threshold probability for class 1 (for binary classes only)

  TBayesClassifier(const bool &anP=true);
  TBayesClassifier(PDomain, PDistribution, PDomainContingency, PProbabilityEstimator = PProbabilityEstimator(), PConditionalProbabilityEstimatorList = PConditionalProbabilityEstimatorList(), const bool &anP=true, const float &thresh = 0.5);

  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &, TValue &, PDistribution &);

  virtual float p(const TValue &classVal, const TExample &exam);
};

#endif
