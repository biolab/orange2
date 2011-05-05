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

#ifndef __NEAREST_HPP
#define __NEAREST_HPP

#include "root.hpp"

class TExample;
class TExampleTable;
class TExamplesDistance;

WRAPPER(ExampleGenerator);
WRAPPER(ExamplesDistance);
WRAPPER(ExamplesDistanceConstructor);

class ORANGE_API TFindNearest : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  int distanceID; //P id of meta attribute where the distance should be stored (0 = no storing)
  bool includeSame; //P tells whether to include examples that are same as the reference example

  TFindNearest(const int anID = 0, const bool is = true);
  virtual PExampleGenerator operator()(const TExample &, const float &k = 0.0, bool needsClass = false) =0;
};

WRAPPER(FindNearest);

class ORANGE_API TFindNearest_BruteForce: public TFindNearest {
public:
  __REGISTER_CLASS

  PExamplesDistance distance; //P metrics
  PExampleGenerator examples; //P a list of stored examples
  int weightID; //P weight to use when counting examples

  TFindNearest_BruteForce(PExampleGenerator = PExampleGenerator(), const int &aweightID = 0, PExamplesDistance = PExamplesDistance(), const int anID = 0, const bool is = true);
  virtual PExampleGenerator operator()(const TExample &, const float &k = 0.0, bool needsClass = false);
};


class ORANGE_API TFindNearestConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  PExamplesDistanceConstructor distanceConstructor; //P metrics
  bool includeSame; //P tells whether to include examples that are same as the reference example
  
  TFindNearestConstructor(PExamplesDistanceConstructor = PExamplesDistanceConstructor(), const bool is = true);
  virtual PFindNearest operator()(PExampleGenerator, const int &weightID = 0, const int &distanceID = 0) =0;
};


class ORANGE_API TFindNearestConstructor_BruteForce : public TFindNearestConstructor {
public:
  __REGISTER_CLASS

  TFindNearestConstructor_BruteForce(PExamplesDistanceConstructor = PExamplesDistanceConstructor(), const bool is = true);
  virtual PFindNearest operator()(PExampleGenerator, const int &weightID = 0, const int &distanceID = 0);
};

#endif
