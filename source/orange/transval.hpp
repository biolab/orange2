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


#ifndef __TRANSVAL_HPP
#define __TRANSVAL_HPP

#include "root.hpp"
#include "orvector.hpp"

WRAPPER(TransformValue);
WRAPPER(Domain);
WRAPPER(ExampleGenerator);

/*  Transforms the value to another value. Transforming can be done 'in place' by replacing the old
    value with a new one (function 'transform'). Alternatively, operator () can be used to get
    the transformed value without replacing the original. Transformations can be chained. */
class TTransformValue : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  PTransformValue subTransform; //P transformation executed prior to this

  TTransformValue(TTransformValue *tr =0);
  TTransformValue(const TTransformValue &old);

  TValue operator()(const TValue &val);

  virtual void transform(TValue &val) =0;
};


class TMapIntValue : public TTransformValue {
public:
  __REGISTER_CLASS

  PIntList mapping; //P a lookup table

  TMapIntValue(PIntList = PIntList());
  TMapIntValue(const TIntList &);

  virtual void transform(TValue &val);
};


class TDiscrete2Continuous : public TTransformValue {
public:
  __REGISTER_CLASS

  int value; //P target value
  bool invert; //P give 1.0 to values not equal to the target

  TDiscrete2Continuous(const int =-1, bool = false);
  virtual void transform(TValue &);
};


class TNormalizeContinuous : public TTransformValue {
public:
  __REGISTER_CLASS

  float average; //P the average value
  float span; //P the value span

  TNormalizeContinuous(const float =0.0, const float =0.0);
  virtual void transform(TValue &);
};

class TEnumVariable;
WRAPPER(Variable)

PVariable discrete2continuous(TEnumVariable *evar, PVariable wevar, const int &val);
void discrete2continous(PVariable var, TVarList &vars);
PVariable normalizeContinuous(PVariable var, const float &avg, const float &span);
PVariable discreteClass2continous(PVariable classVar, const int &targetClass, bool invertClass);
PDomain regressionDomain(PDomain, const int &targetClass = -1, bool invert = false);
PDomain regressionDomain(PExampleGenerator, const int &targetClass = -1, bool invertClass = false, bool normalizeContinuous = true);

bool hasNonContinuousAttributes(PDomain, bool checkClass);

#endif

