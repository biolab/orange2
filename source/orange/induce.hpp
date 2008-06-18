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

