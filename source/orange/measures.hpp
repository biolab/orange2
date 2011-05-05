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
WRAPPER(SymMatrix);

float getEntropy(PContingency, int unknownsTreatment);
float getEntropy(const vector<float> &);

float getGini(const vector<float> &, int unknownsTreatment);
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

class ORANGE_API TMeasureAttribute : public TOrange {
public:
  __REGISTER_CLASS

  CLASSCONSTANTS(Needs) enum {Contingency_Class, DomainContingency, Generator};
  CLASSCONSTANTS(UnknownsTreatment) enum {IgnoreUnknowns, ReduceByUnknowns, UnknownsToCommon, UnknownsAsValue};

  int needs; //P(&MeasureAttribute_Needs) describes what kind of data is needed for computation
  bool handlesDiscrete; //P tells whether the measure can handle discrete attributes
  bool handlesContinuous; //P tells whether the measure can handle continuous attributes
  bool computesThresholds; //P tells whether the measure can compute threshold functions/maxima for continuous attributes

  TMeasureAttribute(const int aneeds, const bool handlesDiscrete, const bool handlesContinuous = false, const bool computesThresholds = false);

  virtual float operator()(PContingency,  PDistribution classDistribution, PDistribution apriorClass=PDistribution());
  virtual float operator()(int attrNo,    PDomainContingency,              PDistribution apriorClass=PDistribution());

  // if the method implements one of these two but not both, it should implement the second
  virtual float operator()(int attrNo,    PExampleGenerator,               PDistribution apriorClass=PDistribution(), int weightID=0);
  virtual float operator()(PVariable var, PExampleGenerator,               PDistribution apriorClass=PDistribution(), int weightID=0);

  virtual float operator()(PDistribution) const;
  virtual float operator()(const TDiscDistribution &) const;
  virtual float operator()(const TContDistribution &) const;

  virtual void thresholdFunction(TFloatFloatList &res, PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution());
  virtual void thresholdFunction(TFloatFloatList &res, PVariable, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID = 0);

  virtual float bestThreshold(PDistribution &, float &score, PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution(), const float &minSubset = -1);
  virtual float bestThreshold(PDistribution &, float &score, PVariable, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID = 0, const float &minSubset = -1);

  virtual PIntList bestBinarization(PDistribution &, float &score, PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution(), const float &minSubset = -1);
  virtual PIntList bestBinarization(PDistribution &, float &score, PVariable, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID = 0, const float &minSubset = -1);

  virtual int bestValue(PDistribution &, float &score, PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution(), const float &minSubset = -1);
  virtual int bestValue(PDistribution &, float &score, PVariable, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID = 0, const float &minSubset = -1);

  virtual bool checkClassType(const int &varType);
  virtual void checkClassTypeExc(const int &varType);
};


class ORANGE_API TMeasureAttributeFromProbabilities : public TMeasureAttribute {
public:
  __REGISTER_ABSTRACT_CLASS

  PProbabilityEstimatorConstructor estimatorConstructor; //P probability estimator (none by default)
  PConditionalProbabilityEstimatorConstructor conditionalEstimatorConstructor; //P conditional probability estimator (none by default)

  int unknownsTreatment; //P(&MeasureAttribute_UnknownsTreatment) treatment of unknown values

  TMeasureAttributeFromProbabilities(const bool handlesDiscrete, const bool handlesContinuous = false, const int unkTreat = ReduceByUnknowns);

  virtual float operator()(PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution());
  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)=0;
};

WRAPPER(MeasureAttribute);


class ORANGE_API TMeasureAttribute_info : public TMeasureAttributeFromProbabilities {
public:
  __REGISTER_CLASS

  TMeasureAttribute_info(const int &unkTreat = ReduceByUnknowns);

  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
  virtual float operator()(const TDiscDistribution &) const;
};


class ORANGE_API TMeasureAttribute_gainRatio : public TMeasureAttributeFromProbabilities {
public:
  __REGISTER_CLASS

  TMeasureAttribute_gainRatio(const int &unkTreat = ReduceByUnknowns);

  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
};


class ORANGE_API TMeasureAttribute_gainRatioA : public TMeasureAttribute_gainRatio {
public:
  __REGISTER_CLASS

  virtual float operator()(const TDiscDistribution &) const;
};


class ORANGE_API TMeasureAttribute_gini : public TMeasureAttributeFromProbabilities {
public:
  __REGISTER_CLASS

  TMeasureAttribute_gini(const int &unkTreat = ReduceByUnknowns);

  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
  virtual float operator()(const TDiscDistribution &) const;
};


class ORANGE_API TMeasureAttribute_logOddsRatio : public TMeasureAttributeFromProbabilities {
public:
  __REGISTER_CLASS

  TMeasureAttribute_logOddsRatio();
  virtual float operator()(PContingency probabilities, const TDiscDistribution &);
};

class ORANGE_API TMeasureAttribute_relevance : public TMeasureAttributeFromProbabilities {
public:
  __REGISTER_CLASS

  TMeasureAttribute_relevance(const int &unkTreat = ReduceByUnknowns);

  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
  float valueRelevance(const TDiscDistribution &dval, const TDiscDistribution &classProbabilities);
};


class ORANGE_API TMeasureAttribute_chiSquare : public TMeasureAttributeFromProbabilities {
public:
  __REGISTER_CLASS

  bool computeProbabilities; //P

  TMeasureAttribute_chiSquare(const int &unkTreat = ReduceByUnknowns, const bool probs = false);

  virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
};


WRAPPER(CostMatrix)

class ORANGE_API TMeasureAttribute_cost: public TMeasureAttributeFromProbabilities {
public:
    __REGISTER_CLASS

    PCostMatrix cost; //P cost matrix

    TMeasureAttribute_cost(PCostMatrix costs=PCostMatrix());

    virtual float operator()(PContingency probabilities, const TDiscDistribution &classProbabilities);
  	float majorityCost(const TDiscDistribution &dval);
  	void majorityCost(const TDiscDistribution &dval, float &cost, TValue &cclass);
};


class ORANGE_API TMeasureAttribute_MSE : public TMeasureAttribute {
public:
    __REGISTER_CLASS

    float m; //P m for m-estimate
    int unknownsTreatment; //P(&MeasureAttribute_UnknownsTreatment) treatment of unknown values

    TMeasureAttribute_MSE(const int &unkTreat = ReduceByUnknowns);
    virtual float operator()(PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution());
};


PContingency prepareBinaryCheat(PDistribution classDistribution, PContingency origContingency,
                                PVariable &bvar,
                                TDiscDistribution *&dis0, TDiscDistribution *&dis1,
                                TContDistribution *&con0, TContDistribution *&con1);

#endif
