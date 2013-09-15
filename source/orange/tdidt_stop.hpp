#ifndef __TDIDT_STOP_HPP
#define __TDIDT_STOP_HPP

#include "root.hpp"

WRAPPER(ExampleGenerator)
WRAPPER(DomainContingency)

class ORANGE_API TTreeStopCriteria : public TOrange {
public:
  __REGISTER_CLASS
  virtual bool operator()(PExampleGenerator, const int &weightID = 0, PDomainContingency =PDomainContingency());
};

WRAPPER(TreeStopCriteria);


class ORANGE_API TTreeStopCriteria_common : public TTreeStopCriteria {
public:
  __REGISTER_CLASS
  float maxMajority; //P a maximal proportion of majority class for division to continue
  float minInstances; //P a minimal number of examples for division to continue

  TTreeStopCriteria_common(const TTreeStopCriteria_common &);
  TTreeStopCriteria_common(float aMaxMajor=1, float aMinExamples=0);

  virtual bool operator()(PExampleGenerator gen, const int &weightID = 0, PDomainContingency =PDomainContingency());
};

#endif
