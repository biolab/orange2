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


#ifndef __DISTVARS_HPP
#define __DISTVARS_HPP

#ifdef _MSC_VER
 #pragma warning (disable : 4786 4114 4018 4267)
#endif

#include <map>
#include <vector>
#include "root.hpp"
#include "values.hpp"
#include "stladdon.hpp"
#include "orvector.hpp"

using namespace std;

class TValue;
WRAPPER(Variable)
WRAPPER(Distribution)
WRAPPER(DiscDistribution)
WRAPPER(ContDistribution)
WRAPPER(ExampleGenerator)
WRAPPER(RandomGenerator)

class TExample;


#define TDistributionList TOrangeVector<PDistribution> 
VWRAPPER(DistributionList)

class ORANGE_API TDistribution : public TSomeValue {
public:
  __REGISTER_ABSTRACT_CLASS

  PVariable variable; //P attribute descriptor (optional)
  float unknowns; //P number of unknown values
  float abs; //P sum of frequencies (not including unknown values!)
  float cases; //P number of cases; as abs, but doesn't change at *= and normalize()
  bool normalized; //P distribution is normalized

  bool supportsDiscrete; //PR distribution supports discrete interface
  bool supportsContinuous; //PR distribution supports continuous interface

  PRandomGenerator randomGenerator; //P random generator; initialized when needed, if not given earlier

  TDistribution();
  TDistribution(PVariable var);

  // Creates either TDiscDistribution or TContDistribution
  static TDistribution *create(PVariable);
  static TDistribution *fromGenerator(PExampleGenerator gen, const int &position, const int &weightID);
  static TDistribution *fromGenerator(PExampleGenerator gen, PVariable, const int &weightID);

  /* Derived classes must define those (if they make sense) */

  virtual TDistribution &operator += (const TDistribution &other);
  virtual TDistribution &operator -= (const TDistribution &other);
  virtual TDistribution &operator *= (const TDistribution &other);
  virtual TDistribution &operator *= (const float &);
  virtual void  normalize() = 0;
  virtual float highestProb() const =0;
  virtual bool noDeviation() const = 0;


  /* Those that have supportDiscrete == true must redefine those */

  virtual const float &atint(const int &i);
  virtual const float &atint(const int &i) const;
  virtual void  addint  (const int &v, const float &w = 1.0);
  virtual void  setint  (const int &v, const float &w);
  virtual int   highestProbIntIndex() const;
  virtual int   highestProbIntIndex(const long &) const;
  virtual int   highestProbIntIndex(const TExample &) const;
  virtual int   randomInt();
  virtual int   randomInt(const long &random);
  virtual float p(const int &) const;
  virtual int   noOfElements() const;


  /* Those that have supportContinuous == true must redefine those */

  virtual const float &atfloat(const float &f);
  virtual const float &atfloat(const float &f) const;
  virtual void  addfloat(const float &v, const float &w = 1.0);
  virtual void  setfloat(const float &v, const float &w);
  virtual float highestProbFloatIndex() const;
  virtual float randomFloat();
  virtual float randomFloat(const long &random);
  virtual float average() const;
  virtual float dev() const;
  virtual float var() const;
  virtual float percentile(const float &) const;
  virtual float error() const;
  virtual float p(const float &) const;

  /* The below methods may be redefined (they are not implemented in TDistribution) */

  virtual int   compare       (const TSomeValue &other) const;
  virtual bool  compatible    (const TSomeValue &other) const;
  virtual float compatibility (const TSomeValue &other) const;


  /* Those do not need to be redefined */

  TDistribution &operator +=(PDistribution);
  TDistribution &operator -=(PDistribution);
  TDistribution &operator *=(PDistribution);

  virtual const float &operator[](const TValue &val);
  virtual const float &operator[](const TValue &val) const;
  virtual void   add(const TValue &i, const float &p = 1.0);
  virtual void   set(const TValue &i, const float &p);
  virtual TValue highestProbValue() const;
  virtual TValue highestProbValue(const long &random) const;
  virtual TValue highestProbValue(const TExample &random) const;
  virtual TValue randomValue();
  virtual TValue randomValue(const long &random);
  virtual float p(const TValue &) const;

  virtual float operator -  (const TSomeValue &v) const;
  virtual float operator || (const TSomeValue &v) const;

  virtual int sumValues() const = 0;
};


/*  Distribution of discrete values. Class is (indirect) descendant of TSomeValue so that
    it can be used in TExamples in place of values.
    Frequencies of values are stored in vector<float>. Sum of frequencies is given in abs.
    If sum is 1.0, normalized is true.
    Normalize divides all values by abs.

    The variances field can be used to give variances of probabilities; DiscDistribution's
    operators do not manipulate it - it is up to you what you store in there and how
    you use it. */

#ifdef _MSC_VER
  template class ORANGE_API std::vector<float>;
#endif

class ORANGE_API TDiscDistribution : public TDistribution {
public:
  __REGISTER_CLASS
  VECTOR_INTERFACE(float, distribution)
  PFloatList variances; //P variances

  TDiscDistribution();
  TDiscDistribution(int values, float value=0);
  TDiscDistribution(const vector<float> &f);
  TDiscDistribution(const float *, const int &len);
  TDiscDistribution(PVariable);
  TDiscDistribution(PDistribution);
  TDiscDistribution(PDiscDistribution);
  TDiscDistribution(PExampleGenerator, const int &position, const int &weightID = 0);
  TDiscDistribution(PExampleGenerator, PVariable var, const int &weightID = 0);

  TDistribution &operator +=(const TDistribution &);
  TDistribution &operator -=(const TDistribution &);
  TDistribution &operator +=(PDistribution);
  TDistribution &operator -=(PDistribution);
  TDistribution &operator *=(const float &weight);
  TDistribution &operator *=(const TDistribution &);
  TDistribution &operator *=(PDistribution);
  TDistribution &operator /=(const TDistribution &);
  TDistribution &operator /=(PDistribution);

  TDistribution &adddist(const TDistribution &, const float &factor);
  TDistribution &adddist(PDistribution, const float &factor);
  TDistribution &mul(const TDistribution &, const float &weight);
  TDistribution &mul(PDistribution, const float &weight);

  virtual const float &atint   (const int &v);
  virtual const float &atint   (const int &v) const;
  virtual void  addint (const int &v, const float &w = 1.0);
  virtual void  setint (const int &v, const float &w);
  virtual float p(const int &) const;
  virtual int   noOfElements() const;

  virtual int   compare       (const TSomeValue &other) const;
  virtual bool  compatible    (const TSomeValue &other) const;
  virtual float compatibility (const TSomeValue &ot) const;

  virtual void  normalize();
  virtual int   highestProbIntIndex() const;
  virtual int   highestProbIntIndex(const long &) const;
  virtual int   highestProbIntIndex(const TExample &) const;
  virtual float highestProb() const;
  virtual int   randomInt();
  virtual int   randomInt(const long &random);
  virtual bool  noDeviation() const;

  virtual int sumValues() const;
};


/*  The variances field can be used to give variances of probabilities; ContDistribution's
    operators do not manipulate it - it is up to you what you store in there and how
    you use it. Each point in the vector corresponds to a point in the map.
    (This is so because you have not implemented - pointers to wrapped maps (ormap.hpp) */

class ORANGE_API TContDistribution : public TDistribution {
public:
  __REGISTER_CLASS
  #ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251)
  #endif
  MAP_INTERFACE_WOUT_OP(float, float, distribution, typedef)
  #ifdef _MSC_VER
    #pragma warning(pop)
  #endif

  PFloatList variances; //P variances
  float sum; //PR weighted sum of elements (i.e. N*average)
  float sum2; //PR weighted sum of squares of elements
  
  TContDistribution();
  TContDistribution(const map<float, float> &);
  TContDistribution(PVariable);
  TContDistribution(PExampleGenerator, const int &position, const int &weightID = 0);
  TContDistribution(PExampleGenerator, PVariable var, const int &weightID = 0);

  TDistribution &operator +=(const TDistribution &other);
  TDistribution &operator -=(const TDistribution &other);
  TDistribution &operator +=(PDistribution);
  TDistribution &operator -=(PDistribution);
  TDistribution &operator *=(const float &);

  virtual const float &atfloat (const float &v);
  virtual const float &atfloat (const float &v) const;
  virtual void  addfloat(const float &f, const float &w = 1.0);
  virtual void  setfloat(const float &v, const float &w);
  virtual float p(const float &) const;

  virtual float average() const;
  virtual float dev() const;
  virtual float var() const;
  virtual float error() const;
  virtual float percentile(const float &) const;
  
  virtual void  normalize();
  virtual float highestProbFloatIndex() const;
  virtual float highestProb() const;
  virtual float randomFloat();
  virtual float randomFloat(const long &random);
  virtual bool  noDeviation() const;

  virtual int sumValues() const;
};



class ORANGE_API TGaussianDistribution : public TDistribution {
public:
  __REGISTER_CLASS

  float mean; //P mu
  float sigma; //P sigma
  
  TGaussianDistribution(const float &mean = 0.0, const float &sigma = 1.0, const float &anabs = 1.0);
  TGaussianDistribution(PDistribution);

  virtual float average() const;
  virtual float dev() const;
  virtual float var() const;
  virtual float error() const;
  
  virtual void  normalize();
  virtual float highestProbFloatIndex() const;
  virtual float highestProb() const;
  virtual float randomFloat();
  virtual float randomFloat(const long &random);

  virtual float p(const float &) const;
  virtual bool  noDeviation() const;

  virtual int sumValues() const;
};


#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable: 4661)
  template class ORANGE_API TOrangeVector<PDistribution>;
  #pragma warning(pop)
#endif

// Distributions of attributes' and classes' values for the examples from the generator
class ORANGE_API TDomainDistributions : public TOrangeVector<PDistribution> {
public:
  __REGISTER_CLASS

  TDomainDistributions();
  TDomainDistributions(PExampleGenerator, const long weightID=0, bool skipDiscrete = false, bool skipContinuous = false);
  void normalize();
};

WRAPPER(DomainDistributions)

PDistribution getClassDistribution(PExampleGenerator, const long &weightID=0);

#define CAST_TO_DISCDISTRIBUTION(x) dynamic_cast<const TDiscDistribution &>((x).getReference())
#define CAST_TO_CONTDISTRIBUTION(x) dynamic_cast<const TContDistribution &>((x).getReference())

#endif
