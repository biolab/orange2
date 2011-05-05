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


#ifndef __COSTWRAPPER_HPP
#define __COSTWRAPPER_HPP

#include "learn.hpp"
#include "classify.hpp"

WRAPPER(CostMatrix)

class ORANGE_API TCostWrapperLearner : public TLearner {
public:
  __REGISTER_CLASS

  PLearner basicLearner; //P basic learner
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
