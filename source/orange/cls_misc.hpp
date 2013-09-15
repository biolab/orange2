#ifndef __CLS_MISC_HPP
#define __CLS_MISC_HPP

#include "orange_api.hpp"

class TDomainDepot;

class ORANGE_API TPyDomainDepot {
public:
  PyObject_HEAD
  TDomainDepot *domainDepot;
};

#endif
