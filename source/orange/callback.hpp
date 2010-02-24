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


#ifndef __CALLBACK_HPP
#define __CALLBACK_HPP

#include "Python.h"
#include "garbage.hpp"

ORANGE_API PyObject *callCallback(PyObject *self, PyObject *args);
ORANGE_API PyObject *setCallbackFunction(PyObject *self, PyObject *func);

ORANGE_API PyObject *callbackReduce(PyObject *self, TOrangeType &basetype);

#include "filter.hpp"
class ORANGE_API TFilter_Python : public TFilter {
public:
  __REGISTER_CLASS
  bool operator()(const TExample &ex);
  PFilter deepCopy() const;
};

#include "transval.hpp"
class ORANGE_API TTransformValue_Python : public TTransformValue {
public:
  __REGISTER_CLASS
  void transform(TValue &);
};


#include "measures.hpp"
class ORANGE_API TMeasureAttribute_Python : public TMeasureAttribute {
public:
  __REGISTER_CLASS
  TMeasureAttribute_Python();
  virtual float operator()(int attrNo, PExampleGenerator gen, PDistribution apriorClass, int weightID);
  virtual float operator()(PVariable var, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID=0);
  virtual float operator()(PContingency, PDistribution classDistribution, PDistribution apriorClass=PDistribution());
  virtual float operator()(int attrNo, PDomainContingency, PDistribution apriorClass=PDistribution());
  virtual float operator()(PDistribution) const;
  virtual float operator ()(const TDiscDistribution &) const;
  virtual float operator ()(const TContDistribution &) const;

  virtual void thresholdFunction(TFloatFloatList &res, PVariable, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID = 0);

private:
  float callMeasure(PyObject *args);
};


#include "learn.hpp"
class ORANGE_API TLearner_Python : public TLearner {
public:
  __REGISTER_CLASS
  virtual PClassifier operator()(PExampleGenerator, const int &);
};


#include "logfit.hpp"
class ORANGE_API TLogRegFitter_Python : public TLogRegFitter {
public:
  __REGISTER_CLASS
  virtual PAttributedFloatList operator()(PExampleGenerator, const int &, PAttributedFloatList &, float &, int &, PVariable &);
};


#include "classify.hpp"
class ORANGE_API TClassifier_Python : public TClassifier {
public:
  __REGISTER_CLASS
  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &, TValue &, PDistribution &);
};


#include "tdidt.hpp"
#include "tdidt_split.hpp"
#include "tdidt_stop.hpp"

class ORANGE_API TTreeSplitConstructor_Python : public TTreeSplitConstructor {
public:
  __REGISTER_CLASS
  virtual PClassifier operator ()(PStringList &, PDiscDistribution &, float &, int &, PExampleGenerator, const int & =0, PDomainContingency =PDomainContingency(), PDistribution = PDistribution(), const vector<bool> & = vector<bool>(), PClassifier nodeClassifier = PClassifier());
};


class ORANGE_API TTreeStopCriteria_Python : public TTreeStopCriteria {
public:
  __REGISTER_CLASS
  virtual bool operator()(PExampleGenerator gen, const int & =0, PDomainContingency =PDomainContingency());
};


class ORANGE_API TTreeDescender_Python : public TTreeDescender {
public:
  __REGISTER_CLASS
  virtual PTreeNode operator() (PTreeNode node, const TExample &, PDiscDistribution &);
};


class ORANGE_API TTreeExampleSplitter_Python : public TTreeExampleSplitter {
public:
  __REGISTER_CLASS
  virtual PExampleGeneratorList operator() (PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights);
};


#include "progress.hpp"

class ORANGE_API TProgressCallback_Python : public TProgressCallback {
public:
  __REGISTER_CLASS
  virtual bool operator()(const float &, POrange);
};


#include "imputation.hpp"

class ORANGE_API TImputerConstruct_Python : public TImputerConstructor {
public:
  __REGISTER_CLASS
  virtual PImputer operator()(PExampleGenerator, const int &);
};


class ORANGE_API TImputer_Python : public TImputer {
public:
  __REGISTER_CLASS
  virtual TExample *operator()(TExample &example);
};

#include "rulelearner.hpp"

class ORANGE_API TRuleEvaluator_Python : public TRuleEvaluator {
public:
  __REGISTER_CLASS
  virtual float operator()(PRule, PExampleTable, const int &, const int &targetClass, PDistribution);
};

class ORANGE_API TEVDistGetter_Python : public TEVDistGetter {
public:
  __REGISTER_CLASS
  virtual PEVDist operator()(const PRule, const int &, const int &) const;
};

class ORANGE_API TRuleValidator_Python : public TRuleValidator {
public:
  __REGISTER_CLASS
  virtual bool operator()(PRule, PExampleTable, const int &, const int &targetClass, PDistribution) const;
};

class ORANGE_API TRuleCovererAndRemover_Python : public TRuleCovererAndRemover {
public:
  __REGISTER_CLASS
  virtual PExampleTable operator()(PRule, PExampleTable, const int &weightID, int &newWeight, const int &targetClass = -1) const;
};

class ORANGE_API TRuleStoppingCriteria_Python : public TRuleStoppingCriteria {
public:
  __REGISTER_CLASS
  virtual bool operator()(PRuleList, PRule, PExampleTable, const int &weightID) const;
};

class ORANGE_API TRuleDataStoppingCriteria_Python : public TRuleDataStoppingCriteria {
public:
  __REGISTER_CLASS
  virtual bool operator()(PExampleTable, const int &weightID, const int &targetClass) const;
};

class ORANGE_API TRuleFinder_Python : public TRuleFinder {
public:
  __REGISTER_CLASS
  virtual PRule operator()(PExampleTable, const int &weightID, const int &targetClass, PRuleList baseRules = PRuleList());
};

/*class ORANGE_API TRuleBeamFilter_Python : public TRuleBeamFilter {
public:
  __REG ISTER_CLASS
  virtual void operator()(PRuleList existingRules, PExampleTable, const int &weightID);
};*/

class ORANGE_API TRuleBeamRefiner_Python : public TRuleBeamRefiner {
public:
  __REGISTER_CLASS
  virtual PRuleList operator()(PRule rule, PExampleTable, const int &weightID, const int &targetClass = -1);
};

class ORANGE_API TRuleBeamInitializer_Python : public TRuleBeamInitializer {
public:
  __REGISTER_CLASS
  virtual PRuleList operator()(PExampleTable, const int &weightID, const int &targetClass, PRuleList baseRules, PRuleEvaluator, PDistribution prior, PRule &bestRule);
};

class ORANGE_API TRuleBeamCandidateSelector_Python : public TRuleBeamCandidateSelector {
public:
  __REGISTER_CLASS
  virtual PRuleList operator()(PRuleList &existingRules, PExampleTable, const int &weightID);
};

class ORANGE_API TRuleBeamFilter_Python : public TRuleBeamFilter {
public:
  __REGISTER_CLASS
  virtual void operator()(PRuleList &rules, PExampleTable, const int &weightID);
};

class ORANGE_API TRuleClassifierConstructor_Python : public TRuleClassifierConstructor {
public:
  __REGISTER_CLASS
  virtual PRuleClassifier operator ()(PRuleList, PExampleTable, const int &weightID);
};

#include "svm.hpp"

class ORANGE_API TKernelFunc_Python : public TKernelFunc{
public:
	__REGISTER_CLASS
	virtual float operator()(const TExample &, const TExample &);
};

#include "distance.hpp"

class ORANGE_API TExamplesDistanceConstructor_Python : public TExamplesDistanceConstructor {
public:
  __REGISTER_CLASS
  virtual PExamplesDistance operator()(PExampleGenerator, const int & = 0, PDomainDistributions = PDomainDistributions(), PDomainBasicAttrStat = PDomainBasicAttrStat()) const;
};

class ORANGE_API TExamplesDistance_Python : public TExamplesDistance {
public:
  __REGISTER_CLASS
  virtual float operator()(const TExample &, const TExample &) const;
};

/*
Not verified yet:

#include "decomposition.hpp"
class ORANGE_API TConstructIM_Python : public TConstructIM {
public:
  __ R E G I S T E R _ C L A S S
  virtual PIM operator() (PExampleGenerator, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID=0);
};
*/

// This is defined by Python but then redefined by STLPort
#undef LONGLONG_MAX
#undef ULONGLONG_MAX
#endif
