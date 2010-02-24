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

