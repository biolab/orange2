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


#ifndef __C45_HPP
#define __C45_HPP

#include "classify.hpp"
#include "learn.hpp"


#include "c45/types.i"
#define __TYPES_I


class TC45Learner : public TLearner {
public:
    __REGISTER_CLASS

    bool gainRatio; //P use gain ration (instead of information gain)
    bool subset; //P use subsetting
    bool batch; //P batch
    bool probThresh; //P probability threshold
    int minObjs; //P minimal number of objects (examples) in leaves
    int window; //P window
    int increment; //P increment
    float cf; //P cf
    int trials; //P trials

    bool prune; //P return pruned tree

    bool clearDomain();
    bool clearExamples();

    bool convertDomain(PDomain);
    bool convertExamples(PExampleGenerator);
    bool convertGenerator(PExampleGenerator);

    bool convertParameters();
    bool parseCommandLine(const string &line);

    TC45Learner();

    virtual PClassifier operator()(PExampleGenerator gen, const int &weight = 0);
};


class TC45Classifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  int domainVersion; //P domain version on which the tree was trained
  Tree tree;

  TC45Classifier(PDomain, Tree);
  virtual ~TC45Classifier();
  virtual TValue operator ()(const TExample &);
  PDistribution classDistribution(const TExample &);
};


#endif
