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

#ifndef __LWR_HPP
#define __LWR_HPP

#include "linreg.hpp"
WRAPPER(ExamplesDistanceConstructor)

class TLWRLearner : public TLearner {
public:
  __REGISTER_CLASS

  PExamplesDistanceConstructor distanceConstructor; //P constructor for object that will find the nearest neighbours
  PLinRegLearner linRegLearner; //P learner that performs local linear regression
  float k; //P number of neighbours
  bool rankWeight; //P use ranks instead of distances

  TLWRLearner();
  TLWRLearner(PExamplesDistanceConstructor, PLinRegLearner, const float &, bool);

  virtual PClassifier operator()(PExampleGenerator, const int & = 0);
};


class TLWRClassifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  PFindNearest findNearest; //P object that find nearest neighbours
  PLinRegLearner linRegLearner; //P learner the performs local linear regression
  float k; //P number of neighbours
  bool rankWeight; //P use ranks instead of distances
  int weightID; //P weights of examples stored in 'findNearest'

  TLWRClassifier();
  TLWRClassifier(PDomain, PFindNearest, PLinRegLearner, const float &, bool, const int &);

  TValue operator()(const TExample &ex);
};


#endif