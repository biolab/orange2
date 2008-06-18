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
