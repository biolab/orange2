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


#ifndef __CALLBACK_HPP
#define __CALLBACK_HPP

#include "Python.h"


PyObject *setCallbackFunction(PyObject *self, PyObject *func);

#include "filter.hpp"
class TFilter_Python : public TFilter {
public:
  __REGISTER_CLASS
  bool operator()(const TExample &ex);
};


#include "transval.hpp"
class TTransformValue_Python : public TTransformValue {
public:
  __REGISTER_CLASS
  void transform(TValue &);
};


#include "measures.hpp"
class TMeasureAttribute_Python : public TMeasureAttribute {
public:
  __REGISTER_CLASS
  TMeasureAttribute_Python();
  virtual float operator()(PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution());
  virtual float operator()(int attrNo, PDomainContingency, PDistribution apriorClass=PDistribution());

private:
  float callMeasure(PyObject *args);
};


#include "learn.hpp"
class TLearner_Python : public TLearner {
public:
  __REGISTER_CLASS
  virtual PClassifier operator()(PExampleGenerator, const int &);
};


#include "logfit.hpp"
class TLogisticFitter_Python : public TLogisticFitter {
public:
  __REGISTER_CLASS
  virtual PFloatList operator()(PExampleGenerator, const int &, PFloatList &, float &, int &, PVariable &, const bool &);
};


#include "classify.hpp"
class TClassifier_Python : public TClassifier {
public:
  __REGISTER_CLASS
  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &, TValue &, PDistribution &);
};


#include "tdidt.hpp"
#include "tdidt_split.hpp"
#include "tdidt_stop.hpp"

class TTreeSplitConstructor_Python : public TTreeSplitConstructor {
public:
  __REGISTER_CLASS
  virtual PClassifier operator ()(PStringList &, PDiscDistribution &, float &, int &, PExampleGenerator, const int & =0, PDomainContingency =PDomainContingency(), PDistribution = PDistribution(), const vector<bool> & = vector<bool>(), PClassifier nodeClassifier = PClassifier());
};


class TTreeStopCriteria_Python : public TTreeStopCriteria {
public:
  __REGISTER_CLASS
  virtual bool operator()(PExampleGenerator gen, const int & =0, PDomainContingency =PDomainContingency());
};


class TTreeDescender_Python : public TTreeDescender {
public:
  __REGISTER_CLASS
  virtual PTreeNode operator() (PTreeNode node, const TExample &, PDiscDistribution &);
};


class TTreeExampleSplitter_Python : public TTreeExampleSplitter {
public:
  __REGISTER_CLASS
  virtual PExampleGeneratorList operator() (PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights);
};

/*
Not verified yet:

#include "decomposition.hpp"
class TConstructIM_Python : public TConstructIM {
public:
  __ R E G I S T E R _ C L A S S
  virtual PIM operator() (PExampleGenerator, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID=0);
};
*/

// This is defined by Python but then redefined by STLPort
#undef LONGLONG_MAX
#undef ULONGLONG_MAX
#endif
