#ifndef __BASSTAT_HPP
#define __BASSTAT_HPP

#include "root.hpp"
using namespace std;

// Minimal, maximal, average value of attribute, and deviation
class ORANGE_API TBasicAttrStat : public TOrange {
public:
  __REGISTER_CLASS

  float sum; //P sum of values
  float sum2; //P sum of squares of values
  float n; //P number of examples for which the attribute is defined

  float min; //P the lowest value of the attribute
  float max; //P the highest value of the attribute
  float avg; //P the average value of the attribute
  float dev; //P the deviation of the value of the attribute
  PVariable variable; //P the attribute to which the data applies
  bool holdRecomputation; //P temporarily disables recomputation of avg and dev while adding values

  TBasicAttrStat(PVariable var, const bool &ahold=false);
  TBasicAttrStat(PExampleGenerator gen, PVariable var, const long &weightID = 0);

  void add(float f, float p=1);
  void recompute();

  void reset();
};

WRAPPER(BasicAttrStat);
WRAPPER(ExampleGenerator);

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable: 4661)
  template class ORANGE_API TOrangeVector<PBasicAttrStat>;
  #pragma warning(pop)
#endif


// Minimal, maximal, average value of attribute, and deviation for all attributes from the generator
class ORANGE_API TDomainBasicAttrStat : public TOrangeVector<PBasicAttrStat> {
public:
  __REGISTER_CLASS
  bool hasClassVar; //P has class var

  TDomainBasicAttrStat();
  TDomainBasicAttrStat(PExampleGenerator gen, const long &weightID=0);
  void purge();
};

WRAPPER(DomainBasicAttrStat);


class ORANGE_API TPearsonCorrelation : public TOrange {
public:
  __REGISTER_CLASS

  float r; //P correlation coefficient
  float t; //P t-statics significance
  int df; //P degrees of freedom
  float p; //P significance

  TPearsonCorrelation(PExampleGenerator gen, PVariable v1, PVariable v2, const long &weightID = 0);
};
  
#endif

