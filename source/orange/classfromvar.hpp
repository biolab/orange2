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

#ifndef __CLASSFROMVAR_HPP
#define __CLASSFROMVAR_HPP

#include "classify.hpp"
#include "stladdon.hpp"

WRAPPER(TransformValue);

class ORANGE_API TClassifierFromVar : public TClassifier {
public:
  __REGISTER_CLASS

  PVariable whichVar; //P(+variable) variable
  PTransformValue transformer; //P transformer
  PDistribution distributionForUnknown; //P distribution for unknown value
  bool transformUnknowns; //P if false (default), unknowns stay unknown or are changed into distribution if given

  TClassifierFromVar(PVariable classVar=PVariable(), PDistribution = PDistribution());
  TClassifierFromVar(PVariable classVar, PVariable whichVar, PDistribution = PDistribution());
  TClassifierFromVar(PVariable classVar, PVariable whichVar, PDistribution, PTransformValue);
  TClassifierFromVar(const TClassifierFromVar &);

  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &exam);
  virtual void predictionAndDistribution(const TExample &ex, TValue &val, PDistribution &classDist);

protected:
  int lastDomainVersion;
  PVariable lastWhichVar;
  int position;
};


class ORANGE_API TClassifierFromVarFD : public TClassifierFD {
public:
  __REGISTER_CLASS

  int position; //P position of the attribute in domain
  PTransformValue transformer; //P transformer
  PDistribution distributionForUnknown; //P distribution for unknown value
  bool transformUnknowns; //P if false (default is true), unknowns stay unknown or are changed into distribution if given

  TClassifierFromVarFD(PVariable classVar=PVariable(), PDomain =PDomain(), const int &position = ILLEGAL_INT, PDistribution = PDistribution(), PTransformValue = PTransformValue());
  TClassifierFromVarFD(const TClassifierFromVarFD &);

  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &exam);
  virtual void predictionAndDistribution(const TExample &ex, TValue &val, PDistribution &classDist);
};

#endif
