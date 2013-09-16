#ifndef __TRANSDOMAIN_HPP
#define __TRANSDOMAIN_HPP

#include "root.hpp"
WRAPPER(ExampleGenerator)
WRAPPER(Domain)

class ORANGE_API TDomainTransformerConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  bool preservesOrder; //PR a flag telling that original attributes have corresponding attribute in the same order
  bool preservesAttributes; //PR a flag telling whether each original has a corresponding new attribute

  TDomainTransformerConstructor(const bool &po, const bool &pa);

  virtual PDomain operator()(PExampleGenerator, const long &weightID) =0;
};

WRAPPER(DomainTransformerConstructor)

#endif
