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


#ifndef __CONTINGENCY_HPP
#define __CONTINGENCY_HPP

#include <vector>
#include <map>
using namespace std;

#include "distvars.hpp"

WRAPPER(Distribution)
WRAPPER(DiscDistribution)
WRAPPER(ContDistribution)
WRAPPER(ProbabilityEstimator)

typedef vector<PDistribution> TDistributionVector;
typedef map<float, PDistribution> TDistributionMap;

/*  Distribution of attribute's values for a domain.
    TContingency is a descendant of vector<TDiscDistribution> which holds a contingency table
    for the variable's values; each element of the vector<TDiscDistribution> corresponds to one of
    possible variable's values and each element of the 'subvector' corresponds to a class value and
    holds the number of examples which had the corresponding value of attribute and class. */
class TContingency : public TOrange {
public:
  __REGISTER_CLASS

  PVariable outerVariable; //P outer attribute
  PVariable innerVariable; //P (+variable) inner attribute
  int varType;             //PR outer attribute type

  union {
    TDistributionVector *discrete;
    TDistributionMap *continuous;
  };

  PDistribution outerDistribution; //P distribution of values of outer attributes
  PDistribution innerDistribution; //P distribution of values of inner attributes

  TContingency(PVariable variable=PVariable(), PVariable innervar=PVariable());
  TContingency(const TContingency &old);

  TContingency &operator=(const TContingency &old);
  ~TContingency();

  int traverse(visitproc visit, void *arg);
  int dropReferences();

  void add(const TValue &outvalue, const TValue &invalue, const float p=1);

  void add(PExampleGenerator, const int attrNo, const long weightID=0);
  void add(PExampleGenerator, PVariable var, const long weightID=0);

  PDistribution operator [](const int &i);
  const PDistribution operator [](const int &i) const;
  PDistribution operator [](const float &i);
  const PDistribution operator [](const float &i) const;
  PDistribution operator [](const TValue &i);
  PDistribution const operator [](const TValue &i) const;
  PDistribution operator [](const string &i);
  PDistribution const operator [](const string &i) const;

  PDistribution p(const int &i) const;
  PDistribution p(const float &i) const;
  PDistribution p(const TValue &i) const;
  PDistribution p(const string &i) const;

  void normalize();
};


class TContingencyClass : public TContingency {
public:
  __REGISTER_ABSTRACT_CLASS

  TContingencyClass(PVariable variable=PVariable(), PVariable innervar=PVariable());

  virtual void add_attrclass(const TValue &varValue, const TValue &classValue, const float &p) = 0;
  virtual float p_class(const TValue &varValue, const TValue &classValue) const;
  virtual float p_attr(const TValue &varValue, const TValue &classValue) const;
  virtual PDistribution p_classes(const TValue &varValue) const;
  virtual PDistribution p_attrs(const TValue &classValue) const;

  void constructFromGenerator(PVariable outer, PVariable inner, PExampleGenerator, const long &weightID, const int &attrNo, const bool &useValueFrom);

protected:
  virtual void add_gen(PExampleGenerator gen, const int &attrNo, const long &weightID) = 0;
  virtual void add_gen(PExampleGenerator gen, const long &weightID) = 0;
};


class TContingencyClassAttr : public TContingencyClass {
public:
  __REGISTER_CLASS

  TContingencyClassAttr(PVariable variable=PVariable(), PVariable innervar=PVariable());
  TContingencyClassAttr(PExampleGenerator gen, const int &attrNo, const long &weightID);
  TContingencyClassAttr(PExampleGenerator gen, PVariable var, const long &weightID);

  virtual void add_attrclass(const TValue &varValue, const TValue &classValue, const float &p);
  virtual float p_attr(const TValue &varValue, const TValue &classValue) const;
  virtual PDistribution p_attrs(const TValue &classValue) const;

protected:
  virtual void add_gen(PExampleGenerator gen, const int &attrNo, const long &weightID);
  virtual void add_gen(PExampleGenerator gen, const long &weightID);
};


class TContingencyAttrClass : public TContingencyClass {
public:
  __REGISTER_CLASS

  TContingencyAttrClass(PVariable variable=PVariable(), PVariable innervar=PVariable());
  TContingencyAttrClass(PExampleGenerator gen, PVariable var, const long &weightID);
  TContingencyAttrClass(PExampleGenerator gen, const int &attrNo, const long &weightID);

  virtual void add_attrclass(const TValue &varValue, const TValue &classValue, const float &p);
  virtual float p_class(const TValue &varValue, const TValue &classValue) const;
  virtual PDistribution p_classes(const TValue &varValue) const;

protected:
  virtual void add_gen(PExampleGenerator gen, const int &attrNo, const long &weightID);
  virtual void add_gen(PExampleGenerator gen, const long &weightID);
};


class TContingencyAttrAttr : public TContingency { 
public:
  __REGISTER_CLASS

  TContingencyAttrAttr(PVariable variable, PVariable innervar, PExampleGenerator, const long weightID=0);
  TContingencyAttrAttr(const int &var, const int &innervar, PExampleGenerator, const long weightID=0);

  void operator()(PExampleGenerator, const long =0);
};


WRAPPER(Contingency)
WRAPPER(ContingencyClass)
WRAPPER(ContingencyClassAttr)
WRAPPER(ContingencyAttrClass)
WRAPPER(TContingencyAttrAttr);

WRAPPER(ProbabilityEstimator);

/* Stores TContingency's for all attributes from the generator. Additional field holds frequencies of class values. */
class TDomainContingency : public TOrangeVector<PContingencyClass> {
public:
  __REGISTER_CLASS

  PDistribution classes; //P distribution of class values
  bool classIsOuter; //P tells whether the class is the outer variable
  
  TDomainContingency(bool acout=false); // this is the preferred constructor; use computeMatrix to fill the matrix
  TDomainContingency(PExampleGenerator, const long weightID=0, bool acout=false);  // obsolete; use ComputeDomainContingency instead
//  TDomainContingency(const TDomainContingency &, PProbabilityEstimator);

  virtual void computeMatrix(PExampleGenerator, const long &weightID, PDomain newDomain=PDomain());

  void normalize();
  PDomainDistributions getDistributions();
};


WRAPPER(DomainContingency)

class TComputeDomainContingency : public TOrange {
public:
  __REGISTER_CLASS

  bool classIsOuter; //P tells whether the class is the outer variable in contingencies

  TComputeDomainContingency(bool acout=false);

  virtual PDomainContingency operator()(PExampleGenerator, const long &weightID=0);
};

WRAPPER(TComputeDomainContingency)

#endif
