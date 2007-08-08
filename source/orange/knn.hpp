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


#ifndef __KNN_HPP
#define __KNN_HPP

#include "classify.hpp"
#include "learn.hpp"

WRAPPER(FindNearestConstructor);
WRAPPER(FindNearest);

class ORANGE_API TkNNLearner : public TLearner {
public:
  __REGISTER_CLASS
  
  float k; //P number of neighbours (0 for sqrt of #examples)
  bool rankWeight; //P enable weighting by ranks
  PExamplesDistanceConstructor distanceConstructor; //P metrics

  TkNNLearner(const float &ak=0, PExamplesDistanceConstructor = PFindNearestConstructor());
  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};


class ORANGE_API TkNNClassifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  PFindNearest findNearest; //P 

  float k; //P number of neighbours (0 for sqrt of #examples)
  bool rankWeight; //P enable weighting by ranks
  int weightID; //P id of meta-attribute with weight
  int nExamples; //P the number of learning examples 

  TkNNClassifier(PDomain = PDomain(), const int &weightID =0, const float &ak=0, PFindNearest =PFindNearest(), const bool &rankWeight=true, const int &nEx=0);
  virtual PDistribution classDistribution(const TExample &);
};

#endif
