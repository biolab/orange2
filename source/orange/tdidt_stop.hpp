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
  float minExamples; //P a minimal number of examples for division to continue

  TTreeStopCriteria_common(const TTreeStopCriteria_common &);
  TTreeStopCriteria_common(float aMaxMajor=1, float aMinExamples=0);

  virtual bool operator()(PExampleGenerator gen, const int &weightID = 0, PDomainContingency =PDomainContingency());
};

#endif
