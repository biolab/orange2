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


#ifndef __ESTIMATEPROB_HPP
#define __ESTIMATEPROB_HPP

#include <map>
#include "root.hpp"
#include "orvector.hpp"

WRAPPER(ExampleGenerator);
WRAPPER(Distribution);
WRAPPER(Contingency);

WRAPPER(ProbabilityEstimator)
WRAPPER(ProbabilityEstimatorConstructor)
WRAPPER(ConditionalProbabilityEstimator)
WRAPPER(ConditionalProbabilityEstimatorConstructor)

#define TProbabilityEstimatorList TOrangeVector<PProbabilityEstimator>
VWRAPPER(ProbabilityEstimatorList)

#define TConditionalProbabilityEstimatorList TOrangeVector<PConditionalProbabilityEstimator>
VWRAPPER(ConditionalProbabilityEstimatorList)


typedef map<float, PDistribution> TDistributionMap;

class ORANGE_API TProbabilityEstimator : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  bool supportsDiscrete; //PR can estimate probabilities of discrete attributes
  bool supportsContinuous; //PR can estimate probabilities of continuous attributes

  TProbabilityEstimator(const bool &disc, const bool &cont);
  
  /* Estimates p(val) */
  virtual float operator()(const TValue &val) const =0;

  /* Returns probability distribution for all possible outcomes (p(val1), p(val2), ..., p(valn)).
     May return NULL! */
  virtual PDistribution operator()() const;
};


class ORANGE_API TProbabilityEstimatorConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PProbabilityEstimator operator()(PDistribution frequencies, PDistribution apriori = PDistribution(), PExampleGenerator = PExampleGenerator(), const long &weightID = 0, const int &attrNo = -1) const = 0;
};

class ORANGE_API TConditionalProbabilityEstimator : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  bool supportsDiscrete; //PR can estimate probabilities of discrete attributes
  bool supportsContinuous; //PR can estimate probabilities of continuous attributes

  TConditionalProbabilityEstimator(const bool &disc = true, const bool &cont = true);

  /* Estimates p(val|condition) */
  virtual float operator()(const TValue &val, const TValue &condition) const =0;

  /* Returns probabilities for all possible outcomes, i.e. [p(val_i|condition) for each i]
     E.g. naive Bayesian classifier calls this to get P(C|A) for all classes with a single call.
     May return NULL! */
  virtual PDistribution operator()(const TValue &condition) const =0;

  /* Returns a contingency matrix with probabilities; outer variable is 'condition' and inner is 'value'.
     Not necessarily supported.
     May return NULL! */
  virtual PContingency operator()() const;
};


class ORANGE_API TConditionalProbabilityEstimatorConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS
  virtual PConditionalProbabilityEstimator operator()(PContingency frequencies, PDistribution apriori = PDistribution(), PExampleGenerator = PExampleGenerator(), const long &weightID = 0, const int &attrNo = -1) const = 0;
};




/* This class has a list of pre-computed probabilities. They can be
   either discrete or continuous. If they are discrete, they should be
   normalized (i.e., they should sum to 1.0). 
   The class is constructed for many estimations, such as estimation
   by relative frequencies, by Laplace and m-estimation. */
class ORANGE_API TProbabilityEstimator_FromDistribution : public TProbabilityEstimator {
public:
  __REGISTER_CLASS

  PDistribution probabilities; //P probabilities
  
  TProbabilityEstimator_FromDistribution(PDistribution af = PDistribution());
  
  virtual float operator()(const TValue &val) const;
  virtual PDistribution operator()() const;
};


class ORANGE_API TProbabilityEstimatorConstructor_relative : public TProbabilityEstimatorConstructor {
public:
  __REGISTER_CLASS
  virtual PProbabilityEstimator operator()(PDistribution frequencies, PDistribution apriori = PDistribution(), PExampleGenerator = PExampleGenerator(), const long &weightID = 0, const int &attrNo = -1) const;
};


class ORANGE_API TProbabilityEstimatorConstructor_Laplace : public TProbabilityEstimatorConstructor {
public:
  __REGISTER_CLASS

  float l; //P number of examples added to each class (default: 1)
  bool renormalize; //P computes the estimate on the original (not the normalized) distribution

  TProbabilityEstimatorConstructor_Laplace(const float & = 1.0, const bool & = true);
  virtual PProbabilityEstimator operator()(PDistribution frequencies, PDistribution apriori = PDistribution(), PExampleGenerator = PExampleGenerator(), const long &weightID = 0, const int &attrNo = -1) const;
};


class ORANGE_API TProbabilityEstimatorConstructor_m : public TProbabilityEstimatorConstructor {
public:
  __REGISTER_CLASS

  float m; //P parameter m for m-estimation
  bool renormalize; //P computes the estimate on the original (not the normalized) distribution

  TProbabilityEstimatorConstructor_m(const float & = 2.0, const bool & = true);
  virtual PProbabilityEstimator operator()(PDistribution frequencies, PDistribution apriori = PDistribution(), PExampleGenerator = PExampleGenerator(), const long &weightID = 0, const int &attrNo = -1) const;
};


class ORANGE_API TProbabilityEstimatorConstructor_kernel : public TProbabilityEstimatorConstructor {
public:
  __REGISTER_CLASS

  float minImpact; //P Minimal impact the point must have to be counted
  float smoothing; //P Smoothing factor
  int nPoints;   //P Number of points for curve (negative means the given number of points is inserted in each interval)

  TProbabilityEstimatorConstructor_kernel(const float &minImpact = 0.01, const float &smoothing = 1.144, const int &nP = -3);
  virtual PProbabilityEstimator operator()(PDistribution frequencies, PDistribution apriori = PDistribution(), PExampleGenerator = PExampleGenerator(), const long &weightID = 0, const int &attrNo = -1) const;

};


class ORANGE_API TProbabilityEstimatorConstructor_loess : public TProbabilityEstimatorConstructor {
public:
  __REGISTER_CLASS
  CLASSCONSTANTS(DistributionMethod: Minimal=DISTRIBUTE_MINIMAL; Factor=DISTRIBUTE_FACTOR; Fixed=DISTRIBUTE_FIXED; Uniform=DISTRIBUTE_UNIFORM; Maximal=DISTRIBUTE_MAXIMAL)

  float windowProportion; //P The proportion of points in a window for LR
  int nPoints; //P The number of points on curve (negative means the given number of points is inserted in each interval)
  int distributionMethod; //P(&ProbabilityEstimatorConstructor_loess_DistributionMethod) Meaning of the 'nPoints'

  TProbabilityEstimatorConstructor_loess(const float &windowProp = 0.5, const int &nP = -1);
  virtual PProbabilityEstimator operator()(PDistribution frequencies, PDistribution apriori = PDistribution(), PExampleGenerator = PExampleGenerator(), const long &weightID = 0, const int &attrNo = -1) const;
};




class ORANGE_API TConditionalProbabilityEstimator_FromDistribution : public TConditionalProbabilityEstimator {
public:
  __REGISTER_CLASS
  
  PContingency probabilities; //P Contingency that stores conditional probabilities
  
  TConditionalProbabilityEstimator_FromDistribution(PContingency = PContingency());

  virtual float operator()(const TValue &val, const TValue &condition) const;
  virtual PDistribution operator()(const TValue &condition) const;
  virtual PContingency operator()() const;
};


class ORANGE_API TConditionalProbabilityEstimator_ByRows : public TConditionalProbabilityEstimator {
public:
  __REGISTER_CLASS
  PProbabilityEstimatorList estimatorList; //P A list of probability estimators

  virtual float operator()(const TValue &val, const TValue &condition) const;
  virtual PDistribution operator()(const TValue &condition) const;

  void checkCondition(const TValue &condition) const;
};


class ORANGE_API  TConditionalProbabilityEstimatorConstructor_ByRows : public TConditionalProbabilityEstimatorConstructor {
public:
  __REGISTER_CLASS
  
  PProbabilityEstimatorConstructor estimatorConstructor; //P ProbabilityEstimator to be used 
  
  TConditionalProbabilityEstimatorConstructor_ByRows(PProbabilityEstimatorConstructor = PProbabilityEstimatorConstructor());
  virtual PConditionalProbabilityEstimator operator()(PContingency frequencies, PDistribution apriori = PDistribution(), PExampleGenerator = PExampleGenerator(), const long &weightID = 0, const int &attrNo = -1) const;
};


class ORANGE_API TConditionalProbabilityEstimatorConstructor_loess : public TConditionalProbabilityEstimatorConstructor {
public:
  __REGISTER_CLASS

  CLASSCONSTANTS(DistributionMethod: Minimal=DISTRIBUTE_MINIMAL; Factor=DISTRIBUTE_FACTOR; Fixed=DISTRIBUTE_FIXED; Uniform=DISTRIBUTE_UNIFORM; Maximal=DISTRIBUTE_MAXIMAL)

  float windowProportion; //P The proportion of points in a window for LR
  int nPoints; //P The number of points on curve
  int distributionMethod; //P(&ConditionalProbabilityEstimatorConstructor_loess_DistributionMethod) Meaning of the 'nPoints'

  TConditionalProbabilityEstimatorConstructor_loess(const float &windowProp = 0.5, const int &nP = 50);
  virtual PConditionalProbabilityEstimator operator()(PContingency frequencies, PDistribution apriori = PDistribution(), PExampleGenerator = PExampleGenerator(), const long &weightID = 0, const int &attrNo = -1) const;
};

#endif
