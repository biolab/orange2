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


#ifndef __BAYES_HPP
#define __BAYES_HPP

#include "classify.hpp"
#include "estimateprob.hpp"
#include "learn.hpp"

WRAPPER(DomainContingency);
WRAPPER(PProbabilityEstimator);

class TBayesLearner : public TLearner {
public:
  __REGISTER_CLASS

  PProbabilityEstimatorConstructor estimatorConstructor;                                 //P constructs a probability estimator for P(C)
  PConditionalProbabilityEstimatorConstructor conditionalEstimatorConstructor;           //P constructs a probability estimator for P(C|A) 
  PConditionalProbabilityEstimatorConstructor conditionalEstimatorConstructorContinuous; //P constructs a probability estimator for P(C|A) for continuous attributes
  bool normalizePredictions;  //P instructs learner to construct a classifier that normalizes probabilities

  TBayesLearner();
  TBayesLearner(const TBayesLearner &old);

  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};


/* Naive Bayesian classifier.
   If classDistribution is given, it is used; if not, estimator is called to get class probabilities.
   Further, conditionalDistributions are used for attributes for which they are defined; for others,
   a corresponding conditionalEstimator is called. */
class TBayesClassifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  PDistribution distribution; //P class distributions (P(C))
  PDomainContingency conditionalDistributions; //P conditional distributions, P(C|A)

  PProbabilityEstimator estimator; //P a probability estimator for P(C)
  PConditionalProbabilityEstimatorList conditionalEstimators; //P a probability estimator for P(C|A)
  bool normalizePredictions; //P if true, classifier will normalize predictions

  TBayesClassifier(const bool &anP=true);
  TBayesClassifier(PDomain, PDistribution, PDomainContingency, PProbabilityEstimator = PProbabilityEstimator(), PConditionalProbabilityEstimatorList = PConditionalProbabilityEstimatorList(), const bool &anP=true);

  virtual PDistribution classDistribution(const TExample &);
  virtual float p(const TValue &classVal, const TExample &exam);
};

#endif
