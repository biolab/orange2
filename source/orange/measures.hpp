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


#ifndef __MEASURES_HPP
#define __MEASURES_HPP

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include <limits>
#include "root.hpp"
#include "estimateprob.hpp"
#include "cost.hpp"

using namespace std;


WRAPPER(Contingency);
WRAPPER(DiscDistribution);

float getEntropy(PContingency, bool unknownsToCommon = false);
float getEntropy(const vector<float> &);

float getGini(const vector<float> &);
float getGini(PContingency, const TDiscDistribution &caseWeights, const float &classGini=0.0);

WRAPPER(Contingency)
WRAPPER(DomainContingency)
WRAPPER(ExampleGenerator);

/* Attribute quality measures are here divided regarding the information they need.
   - class distribution before attribute is known and contingency matrix for the attribute
   - class distribution before attribute is known and contingency matrix for all attributes
   - complete example set
   Requirement is given in the 'needs' field
   Each measure should provide some of them.
   Corresponding methods in TMeasureAttribute provide the simplifications -- if ExampleGenerator
   is given but only contingency is needed, contingency is extracted and given to the simplest method.

   There is additional method for assess a quality of an attribute that is not in the domain
   of the example set but can be computed. In constructs an appropriate structure and calls one of
   the three methods. Concrete class should provide a more efficient method when possible.
*/

#define ATTRIBUTE_REJECTED numeric_limits<float>::min()

class TMeasureAttribute : public TOrange {
public:
  __REGISTER_CLASS

  enum {Contingency_Class, DomainContingency, Generator};
  int needs; //P describes what kind of data is needed for computation
  bool handlesDiscrete; //PR tells whether the measure can handle discrete attributes
  bool handlesContinuous; //PR tells whether the measure can handle continuous attributes

  TMeasureAttribute(const int &aneeds, const bool &handlesDiscrete, const bool &handlesContinuous = false);

  virtual float operator()(PContingency,  PDistribution classDistribution, PDistribution apriorClass=PDistribution());
  virtual float operator()(int attrNo,    PDomainContingency,              PDistribution apriorClass=PDistribution());
  virtual float operator()(int attrNo,    PExampleGenerator,               PDistribution apriorClass=PDistribution(), int weightID=0);
  virtual float operator()(PVariable var, PExampleGenerator,               PDistribution apriorClass=PDistribution(), int weightID=0);

  virtual float operator()(PDistribution) const;
  virtual float operator()(const TDiscDistribution &) const;
  virtual float operator()(const TContDistribution &) const;

  virtual bool checkClassType(const int &varType);
  virtual void checkClassTypeExc(const int &varType);
};


class TMeasureAttributeFromProbabilities : public TMeasureAttribute {
public: 
  __REGISTER_ABSTRACT_CLASS

  enum { IgnoreUnknowns, ReduceByUnknowns, UnknownsToCommon };

  PProbabilityEstimatorConstructor estimator; //P probability estimator (none by default)
  PConditionalProbabilityEstimatorConstructor conditionalEstimator; //P conditional probability estimator (none by default)

  int unknownsTreatment; //P treatment of unknown values

  TMeasureAttributeFromProbabilities(const bool &handlesDiscrete, const bool &handlesContinuous = false, const int &unkTreat = ReduceByUnknowns);

  virtual float operator()(PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution());
  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)=0; 
};

WRAPPER(MeasureAttribute);


class TMeasureAttribute_info : public TMeasureAttributeFromProbabilities {
public:
  __REGISTER_CLASS

  TMeasureAttribute_info(const int &unkTreat = ReduceByUnknowns);

  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
  virtual float operator()(const TDiscDistribution &) const;
};


class TMeasureAttribute_gainRatio : public TMeasureAttributeFromProbabilities {
public:
  __REGISTER_CLASS

  TMeasureAttribute_gainRatio(const int &unkTreat = ReduceByUnknowns);

  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
};


class TMeasureAttribute_gainRatioA : public TMeasureAttribute_gainRatio {
public:
  __REGISTER_CLASS

  virtual float operator()(const TDiscDistribution &) const;
};



class TMeasureAttribute_gini : public TMeasureAttributeFromProbabilities {
public:
  __REGISTER_CLASS

  TMeasureAttribute_gini(const int &unkTreat = ReduceByUnknowns);

  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
  virtual float operator()(const TDiscDistribution &) const;
};


WRAPPER(CostMatrix)

class TMeasureAttribute_cheapestClass : public TMeasureAttributeFromProbabilities {
public:
    __REGISTER_CLASS

    PCostMatrix cost; //P cost matrix

    TMeasureAttribute_cheapestClass(PCostMatrix costs=PCostMatrix());
    
    virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
  	float majorityCost(const TDiscDistribution &dval);
  	void majorityCost(const TDiscDistribution &dval, float &cost, TValue &cclass);
};


class TMeasureAttribute_MSE : public TMeasureAttribute {
public:
    __REGISTER_CLASS

    float m; //P m for m-estimate

    TMeasureAttribute_MSE();
    virtual float operator()(PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution());
};


class TMeasureAttribute_Tretis : public TMeasureAttribute {
public:
    __REGISTER_CLASS

    TMeasureAttribute_Tretis();
    virtual float operator()(PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution());
};

#endif
