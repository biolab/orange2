#ifndef __INDUCE_HPP
#define __INDUCE_HPP

#include "vars.hpp"
WRAPPER(ExampleGenerator)

/*  An abstract class with a pure virtual operator()(PExampleGenerator) which induces a single
    new feature from the given example set, binding the given attribute set */
class ORANGE_API TFeatureInducer : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PVariable operator()(PExampleGenerator, TVarList &boundSet, const string &name, float &quality, const int &weight=0) =0;
};

WRAPPER(FeatureInducer);

#endif

