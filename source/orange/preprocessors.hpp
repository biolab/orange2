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


#ifndef __PREPROCESSORS_HPP
#define __PREPROCESSORS_HPP

#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "ormap.hpp"

#include "filter.hpp"
#include "discretize.hpp"

WRAPPER(Filter);
class TExampleTable;


class TPreprocessor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight)=0;

protected:
  PExampleGenerator filterExamples(PFilter filter, PExampleGenerator generator);
};

WRAPPER(Preprocessor);

#define TVariableFilterMap TOrangeMap<PVariable, PValueFilter, true, true>
MWRAPPER(VariableFilterMap)


class TPreprocessor_drop : public TPreprocessor {
public:
  __REGISTER_CLASS

  PVariableFilterMap values; //P variable-filter pairs

  TPreprocessor_drop();
  TPreprocessor_drop(PVariableFilterMap);
  virtual PExampleGenerator operator()(PExampleGenerator, const int &weightID, int &newWeight);
};


class TPreprocessor_take : public TPreprocessor {
public:
  __REGISTER_CLASS

  PVariableFilterMap values; //P variable-filter pairs

  TPreprocessor_take();
  TPreprocessor_take(PVariableFilterMap);
  virtual PExampleGenerator operator()(PExampleGenerator, const int &weightID, int &newWeight);

  static PFilter constructFilter(PVariableFilterMap values, PDomain domain);
};


class TPreprocessor_ignore : public TPreprocessor {
public:
  __REGISTER_CLASS

  PVarList attributes; //P tells which attributes to remove

  TPreprocessor_ignore();
  TPreprocessor_ignore(PVarList);
  virtual PExampleGenerator operator()(PExampleGenerator, const int &weightID, int &newWeight);
};


class TPreprocessor_select : public TPreprocessor {
public:
  __REGISTER_CLASS

  PVarList attributes; //P tells which attributes to select

  TPreprocessor_select();
  TPreprocessor_select(PVarList);
  virtual PExampleGenerator operator()(PExampleGenerator, const int &weightID, int &newWeight);
};



class TPreprocessor_remove_duplicates : public TPreprocessor {
public:
  __REGISTER_CLASS
  virtual PExampleGenerator operator()(PExampleGenerator, const int &weightID, int &newWeight);
};


class TPreprocessor_skip_missing : public TPreprocessor {
public:
  __REGISTER_CLASS
  virtual PExampleGenerator operator()(PExampleGenerator, const int &weightID, int &newWeight);
};

class TPreprocessor_only_missing: public TPreprocessor {
public:
  __REGISTER_CLASS
  virtual PExampleGenerator operator()(PExampleGenerator, const int &weightID, int &newWeight);
};

class TPreprocessor_skip_missing_classes : public TPreprocessor {
public:
  __REGISTER_CLASS
  virtual PExampleGenerator operator()(PExampleGenerator, const int &weightID, int &newWeight);
};

class TPreprocessor_only_missing_classes : public TPreprocessor {
public:
  __REGISTER_CLASS
  virtual PExampleGenerator operator()(PExampleGenerator, const int &weightID, int &newWeight);
};


#define TVariableFloatMap TOrangeMap<PVariable, float, true, false>
MWRAPPER(VariableFloatMap)

class TPreprocessor_noise : public TPreprocessor {
public:
  __REGISTER_CLASS

  PVariableFloatMap probabilities; //P probabilities for change for individual attributes
  float defaultNoise; //P default noise level

  TPreprocessor_noise();
  TPreprocessor_noise(PVariableFloatMap, const float & = 0.0);
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);
};


class TPreprocessor_gaussian_noise : public TPreprocessor {
public:
  __REGISTER_CLASS

  PVariableFloatMap deviations; //P deviations individual values
  float defaultDeviation; //P default deviation

  TPreprocessor_gaussian_noise();
  TPreprocessor_gaussian_noise(PVariableFloatMap, const float & = 0.0);
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);
};


class TPreprocessor_missing : public TPreprocessor {
public:
  __REGISTER_CLASS

  PVariableFloatMap probabilities; //P probabilities for removal for individual values
  float defaultMissing; //P default proportion of missing values
  int specialType; //P special value type (1=DC, 2=DK)

  TPreprocessor_missing();
  TPreprocessor_missing(PVariableFloatMap, const float & = 0.0, const int &specialType = valueDK);
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);
};


class TPreprocessor_class_noise : public TPreprocessor {
public:
  __REGISTER_CLASS

  float classNoise; //P class noise level

  TPreprocessor_class_noise(const float & = 0.0);
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);
};


class TPreprocessor_class_gaussian_noise : public TPreprocessor {
public:
  __REGISTER_CLASS

  float classDeviation; //P class deviation

  TPreprocessor_class_gaussian_noise(const float & = 0.0);
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);
};


class TPreprocessor_class_missing : public TPreprocessor {
public:
  __REGISTER_CLASS

  float classMissing; //P proportion of missing class values
  int specialType; //P special value type (1=DC, 2=DK)

  TPreprocessor_class_missing(const float & = 0.0, const int & = valueDK);
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);

private:
  void addMissing(PExampleGenerator);
};


class TPreprocessor_cost_weight : public TPreprocessor {
public:
  __REGISTER_CLASS

  PFloatList classWeights; //P weights of examples of particular classes
  bool equalize; //P reweight examples to equalize class proportions

  TPreprocessor_cost_weight();
  TPreprocessor_cost_weight(PFloatList, const bool & = false);
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);
};


class TPreprocessor_censor_weight : public TPreprocessor {
public:
  __REGISTER_CLASS

  // Do not change the order!
  enum {km, nmr, linear};

  PVariable outcomeVar; //P outcome variable name
  TValue eventValue; //P event (fail) value
  int timeID; //P time variable meta ID
  int method; //P weighting method
  float maxTime; //P maximal time

  TPreprocessor_censor_weight();
  TPreprocessor_censor_weight(PVariable, const TValue & = TValue(), const int & = 0, const int & = km, const float & = 0.0);
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);
};


WRAPPER(Discretization);

class TPreprocessor_discretize : public TPreprocessor {
public:
  __REGISTER_CLASS

  PVarList attributes; //P attributes to be discretized (all, if not defined or empty)
  bool notClass; //P do not discretize the class attribute (default: true)
  PDiscretization method; //P discretization method

  PDomain discretizedDomain(PExampleGenerator, int &);

  TPreprocessor_discretize();
  TPreprocessor_discretize(PVarList, const bool & = true, PDiscretization = PDiscretization());
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);
};


class TPreprocessor_filter : public TPreprocessor {
public:
  __REGISTER_CLASS

  PFilter filter; //P filter

  TPreprocessor_filter(PFilter = PFilter());
  virtual PExampleGenerator operator()(PExampleGenerator generators, const int &weightID, int &newWeight);
};

#endif
