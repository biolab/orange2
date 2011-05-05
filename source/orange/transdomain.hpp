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
