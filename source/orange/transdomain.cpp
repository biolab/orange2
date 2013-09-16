#include <vector>
using namespace std;

#include "values.hpp"
#include "transval.hpp"
#include "domain.hpp"

#include "transdomain.ppp"

TDomainTransformerConstructor::TDomainTransformerConstructor(const bool &po, const bool &pa)
: preservesOrder(po),
  preservesAttributes(pa)
{}
