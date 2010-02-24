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
class ORANGE_API TTransformValue : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  PTransformValue subTransform; //P transformation executed prior to this

  TTransformValue(TTransformValue *tr =0);
  TTransformValue(const TTransformValue &old);

  TValue operator()(const TValue &val);

  virtual void transform(TValue &val) =0;
};


class ORANGE_API TMapIntValue : public TTransformValue {
public:
  __REGISTER_CLASS

  PIntList mapping; //P a lookup table

  TMapIntValue(PIntList = PIntList());
  TMapIntValue(const TIntList &);

  virtual void transform(TValue &val);
};


class ORANGE_API TDiscrete2Continuous : public TTransformValue {
public:
  __REGISTER_CLASS

  int value; //P target value
  bool invert; //P give 1.0 to values not equal to the target
  bool zeroBased; //P if true (default) it gives values 0.0 and 1.0; else -1.0 and 1.0, 0.0 for undefined

  TDiscrete2Continuous(const int =-1, bool invert = false, bool zeroBased = true);
  virtual void transform(TValue &);
};


class ORANGE_API TOrdinal2Continuous : public TTransformValue {
public:
  __REGISTER_CLASS

  float factor; //P number of values

  TOrdinal2Continuous(const float & = 1.0);
  virtual void transform(TValue &);
};


class ORANGE_API TNormalizeContinuous : public TTransformValue {
public:
  __REGISTER_CLASS

  float average; //P the average value
  float span; //P the value span

  TNormalizeContinuous(const float =0.0, const float =0.0);
  virtual void transform(TValue &);
};

class TEnumVariable;
WRAPPER(Variable)


class ORANGE_API TDomainContinuizer : public TOrange {
public:
  __REGISTER_CLASS

  CLASSCONSTANTS(MultinomialTreatment) enum { LowestIsBase, FrequentIsBase, NValues, Ignore, IgnoreAllDiscrete, ReportError, AsOrdinal, AsNormalizedOrdinal};
  CLASSCONSTANTS(ContinuousTreatment) enum { Leave, NormalizeBySpan, NormalizeByVariance };
  CLASSCONSTANTS(ClassTreatment: LeaveUnlessTarget=3; ErrorIfCannotHandle=4; AsOrdinal=5)

  bool zeroBased; //P if true (default) it gives values 0.0 and 1.0; else -1.0 and 1.0, 0.0 for undefined
  int continuousTreatment; //P(&DomainContinuizer_MultinomialTreatment) 0-leave as they are, 1-divide by span, 1-normalize
  int multinomialTreatment; //P(&DomainContinuizer_ContinuousTreatment) 0-lowest value, 1-most frequent (or baseValue), 2-n binary, 3-ignore, 4-error, 5-convert as ordinal, 6-ordinal,normalized
  int classTreatment; //P(&DomainContinuizer_ClassTreatment) 3-leave as is unless target is given, 4-error if not continuous nor binary nor target given, 5-convert as ordinal (unless target given)

  TDomainContinuizer();

  PVariable discrete2continuous(TEnumVariable *evar, PVariable wevar, const int &val, bool inv = false) const;
  void discrete2continuous(PVariable var, TVarList &vars, const int &mostFrequent) const;
  PVariable continuous2normalized(PVariable var, const float &avg, const float &span) const;
  PVariable discreteClass2continous(PVariable classVar, const int &targetClass) const;
  PVariable ordinal2continuous(TEnumVariable *evar, PVariable wevar, const float &factor) const;

  PDomain operator()(PDomain, const int &targetClass = -1) const;
  PDomain operator()(PExampleGenerator, const int &weightID, const int &targetClass = -1) const;
};


WRAPPER(DomainContinuizer)

#endif

