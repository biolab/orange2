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


#ifndef __CLASSFROMVAR_HPP
#define __CLASSFROMVAR_HPP

#include "classify.hpp"
#include "stladdon.hpp"

WRAPPER(TransformValue);

class TClassifierFromVar : public TClassifier {
public:
  __REGISTER_CLASS

  PVariable whichVar; //P variable
  PTransformValue transformer; //P transformer
  PDistribution distributionForUnknown; //P distribution for unknown value

  TClassifierFromVar(PVariable classVar=PVariable(), PDistribution = PDistribution());
  TClassifierFromVar(PVariable classVar, PVariable whichVar, PDistribution = PDistribution());
  TClassifierFromVar(const TClassifierFromVar &);

  virtual TValue operator ()(const TExample &);
};


class TClassifierFromVarFD : public TClassifierFD {
public:
  __REGISTER_CLASS

  int position; //P position of the attribute in domain
  PTransformValue transformer; //P transformer
  PDistribution distributionForUnknown; //P distribution for unknown value

  TClassifierFromVarFD(PVariable classVar=PVariable(), PDomain =PDomain(), const int &position = ILLEGAL_INT, PDistribution = PDistribution(), PTransformValue = PTransformValue());
  TClassifierFromVarFD(const TClassifierFromVarFD &);

  virtual TValue operator ()(const TExample &);
};

#endif
