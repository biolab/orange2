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


#ifndef __BOOSTING_HPP
#define __BOOSTING_HPP

#include "classify.hpp"
#include "learn.hpp"

class TBoostLearner : public TLearner {
public:
  __REGISTER_CLASS

  PLearner weakLearner; //P learner
  int T; //P number of iterations

  TBoostLearner(int aT=10);

  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};


class TWeight_Classifier : public TOrange {
public:
  __REGISTER_CLASS

  float weight; //P classifier's weight
  PClassifier classifier; //P classifier
  TWeight_Classifier(const float &aw, PClassifier acl);
};

WRAPPER(Weight_Classifier)

#define TWeight_ClassifierList TOrangeVector<PWeight_Classifier> 
VWRAPPER(Weight_ClassifierList)


class TVotingClassifier : public TClassifier {
public:
  __REGISTER_CLASS

  PWeight_ClassifierList classifiers; //P a list of classifiers and corresponding weights
  virtual PDistribution classDistribution(const TExample &);
};

#endif
