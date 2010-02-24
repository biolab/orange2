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


#ifndef __DISTANCE_HPP
#define __DISTANCE_HPP

#include "contingency.hpp"
#include "basstat.hpp"

class TExample;
WRAPPER(ExamplesDistance);
WRAPPER(ExamplesDistanceConstructor);
WRAPPER(ExampleGenerator);


class ORANGE_API TExamplesDistance : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS
  virtual float operator()(const TExample &, const TExample &) const=0;
};


class ORANGE_API TExamplesDistanceConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS
  bool ignoreClass; //P if true (default), class value is ignored when computing distances

  TExamplesDistanceConstructor(const bool & = true);
  virtual PExamplesDistance operator()(PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat()) const=0;
};



/* Hamming distance: 
      the number of different (incompatible) attribute values.
*/
class ORANGE_API TExamplesDistance_Hamming : public TExamplesDistance {
public:
  __REGISTER_CLASS

  bool ignoreClass; //P if true (default), class value is ignored when computing distances
  bool ignoreUnknowns; //P if true (default: false) unknown values are ignored in computation

  TExamplesDistance_Hamming(const bool &ic = true, const bool &iu = false);
  virtual float operator()(const TExample &, const TExample &) const;
};


class ORANGE_API TExamplesDistanceConstructor_Hamming : public TExamplesDistanceConstructor {
public:
  __REGISTER_CLASS

  bool ignoreClass; //P if true (default), class value is ignored when computing distances
  bool ignoreUnknowns; //P if true (default: false) unknown values are ignored in computation

  TExamplesDistanceConstructor_Hamming();
  virtual PExamplesDistance operator()(PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat()) const;
};



/* An abstract functional objects which returns 'normalized' distance between two examples.
   'ranges' stores
      1/attribute_range for continuous attributes
      1/number_of_values for ordinal attributes
      -1.0 for nominal attribute
      0 if attribute is to be ignored (this can happen for various reasons,
          such as continuous attribute with no known values)
   When computing "difs", it returns a vector that contains
      abs(ex1[i]-ex2[i]) * ranges[i] for continuous and ordinal attributes
      0 or 1 for nominal attributes
   Distance between two values can be greater than 1!
*/
class ORANGE_API TExamplesDistance_Normalized : public TExamplesDistance {
public:
  __REGISTER_ABSTRACT_CLASS

  PAttributedFloatList normalizers; //P normalizing factors for attributes
  PAttributedFloatList bases; //P lowest values for attributes
  PAttributedFloatList averages; //P average values for continuous attribute values
  PAttributedFloatList variances; //P variations for continuous attribute values
  int domainVersion; //P version of domain on which the ranges were computed
  bool normalize; //P tells whether to normalize distances between attributes
  bool ignoreUnknowns; //P if true (default: false) unknown values are ignored in computation


  TExamplesDistance_Normalized();
  TExamplesDistance_Normalized(const bool &ic, const bool &no, const bool &iu, PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat());

  void getDifs(const TExample &ex1, const TExample &ex2, vector<float> &difs) const;
  void getNormalized(const TExample &e1, vector<float> &normalized) const;
};


class ORANGE_API TExamplesDistanceConstructor_Normalized : public TExamplesDistanceConstructor {
public:
  __REGISTER_ABSTRACT_CLASS

  bool normalize; //P tells whether to normalize distances between attributes
  bool ignoreUnknowns; //P if true (default: false) unknown values are ignored in computation

  TExamplesDistanceConstructor_Normalized();
  TExamplesDistanceConstructor_Normalized(const bool &ic, const bool &norm, const bool &iu);
};

/* Maximal distance: 
      the largest distance between two corresponding attribute values
   Be careful about nominal attributes - they will often prevail since
   the distance between them is too easily 1
*/
class ORANGE_API TExamplesDistance_Maximal : public TExamplesDistance_Normalized {
public:
  __REGISTER_CLASS

  TExamplesDistance_Maximal();
  TExamplesDistance_Maximal(const bool &ignoreClass, const bool &normalize, const bool &ignoreUnknowns, PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat());
  virtual float operator()(const TExample &, const TExample &) const;
};


class ORANGE_API TExamplesDistanceConstructor_Maximal : public TExamplesDistanceConstructor_Normalized {
public:
  __REGISTER_CLASS

  virtual PExamplesDistance operator()(PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat()) const;
};



/* Manhattan distance:
      a sum of absolute differences between pairs attribute values
*/
class ORANGE_API TExamplesDistance_Manhattan : public TExamplesDistance_Normalized {
public:
  __REGISTER_CLASS

  TExamplesDistance_Manhattan();
  TExamplesDistance_Manhattan(const bool &ignoreClass, const bool &normalize, const bool &ignoreUnknowns, PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat());
  virtual float operator()(const TExample &, const TExample &) const;
};


class ORANGE_API TExamplesDistanceConstructor_Manhattan : public TExamplesDistanceConstructor_Normalized {
public:
  __REGISTER_CLASS

  TExamplesDistanceConstructor_Manhattan();
  virtual PExamplesDistance operator()(PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat()) const;
};


/* Euclidean distance:
     square root of sum of squared distances between corresponding attribute values
*/
class ORANGE_API TExamplesDistance_Euclidean : public TExamplesDistance_Normalized {
public:
  __REGISTER_CLASS

  PDomainDistributions distributions; //P distributions (of discrete attributes only)
  PAttributedFloatList bothSpecialDist; //P distances between discrete attributes if both values are unknown

  TExamplesDistance_Euclidean();
  TExamplesDistance_Euclidean(const bool &ignoreClass, const bool &normalize, const bool &ignoreUnknowns, PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat());
  virtual float operator()(const TExample &, const TExample &) const;
};


class ORANGE_API TExamplesDistanceConstructor_Euclidean : public TExamplesDistanceConstructor_Normalized {
public:
  __REGISTER_CLASS

  TExamplesDistanceConstructor_Euclidean();
  TExamplesDistanceConstructor_Euclidean(PExampleGenerator);
  virtual PExamplesDistance operator()(PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat()) const;
};


/* Relief distance */

class ORANGE_API TExamplesDistance_Relief : public TExamplesDistance {
public:
  __REGISTER_CLASS

  PDomainDistributions distributions; //P distributions of attributes' values
  PAttributedFloatList averages; //P average values of attributes
  PAttributedFloatList normalizations; //P ranges of attributes' values
  PAttributedFloatList bothSpecial; //P distance if both values of both attributes are undefined

  virtual float operator()(const TExample &, const TExample &) const;
  virtual float operator()(const int &attrNo, const TValue &v1, const TValue &v2) const;
};


class ORANGE_API TExamplesDistanceConstructor_Relief : public TExamplesDistanceConstructor {
public:
  __REGISTER_CLASS

  TExamplesDistanceConstructor_Relief();
  virtual PExamplesDistance operator()(PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat()) const;
};

WRAPPER(ExamplesDistance_Relief);

#endif

