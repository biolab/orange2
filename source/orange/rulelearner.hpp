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

    Authors: Martin Mozina, Janez Demsar, Blaz Zupan, 1996--2005
    Contact: martin.mozina@fri.uni-lj.si
*/


#ifndef __RULES_HPP
#define __RULES_HPP

#include "domain.hpp"
#include "classify.hpp"
#include "learn.hpp"

WRAPPER(ProgressCallback)
WRAPPER(Rule)
WRAPPER(Discretization)
WRAPPER(EVCDist)

#define TRuleList TOrangeVector<PRule>
VWRAPPER(RuleList)
#define TEVCDistList TOrangeVector<PEVCDist>
VWRAPPER(EVCDistList)


WRAPPER(ExampleGenerator)
WRAPPER(ExampleTable)
WRAPPER(Filter)

class ORANGE_API TRule : public TOrange {
public:
  __REGISTER_CLASS

  PFilter filter; //P stored filter for this rule
  PClassifier classifier; //P classifies an example
  PLearner learner; //P learns a classifier from data
  PRule parentRule; //P

  PDistribution classDistribution; //P distribution of classes covered by the rule

  PExampleTable examples; //P covered examples
  int weightID; //P weight for the stored examples
  float quality; //P some measure of rule quality
  float complexity; //P
  float chi; //P 

  int *coveredExamples;
  int coveredExamplesLength;
  
  TRule();
  TRule(PFilter filter, PClassifier classifier, PLearner lr, PDistribution dist, PExampleTable ce = PExampleTable(), const int &w = 0, const float &qu = -1);
  TRule(const TRule &other, bool copyData = true);
  ~TRule();

  bool operator()(const TExample &); //P Returns 1 for accept, 0 for reject
  PExampleTable operator()(PExampleTable, const bool ref = true, const bool negate = false); //P filter examples
    
  void filterAndStore(PExampleTable, const int &weightID = 0, const int &targetClass = -1, const int *prevCovered = NULL, const int anExamples = -1); //P Selects examples from given data
                                                                          // stores them in coveredExamples, computes distribution
                                                                          // and sets classValue (if -1 then take majority)
  bool operator >(const TRule &) const; 
  bool operator <(const TRule &) const;
  bool operator >=(const TRule &) const;
  bool operator <=(const TRule &) const;
  bool operator ==(const TRule &) const;

  bool operator >(const PRule &r) const
  { return operator >(r.getReference()); }

  bool operator <(const PRule &r) const
  { return operator <(r.getReference()); }

  bool operator >=(const PRule &r) const
  { return operator >=(r.getReference()); }

  bool operator <=(const PRule &r) const
  { return operator <=(r.getReference()); }

  bool operator ==(const PRule &r) const
  { return operator ==(r.getReference()); }

  // need string representation of a rule? 
};



WRAPPER(RuleValidator)
class ORANGE_API TRuleValidator : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual bool operator()(PRule, PExampleTable, const int &, const int &targetClass, PDistribution apriori) const = 0;
};


class ORANGE_API TRuleValidator_LRS : public TRuleValidator {
public:
  __REGISTER_CLASS

  float alpha; //P
  float min_coverage; //P
  float max_rule_complexity; //P
  float min_quality; //P

  TRuleValidator_LRS(const float &alpha = 0.05, const float &min_coverage = 0.0, const float &max_rule_complexity = -1.0, const float &min_quality = -numeric_limits<float>::max());
  virtual bool operator()(PRule, PExampleTable, const int &, const int &targetClass, PDistribution ) const;
};


WRAPPER(RuleEvaluator)
class ORANGE_API TRuleEvaluator : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual float operator()(PRule, PExampleTable, const int &, const int &targetClass, PDistribution ) = 0;
};


class ORANGE_API TRuleEvaluator_Entropy : public TRuleEvaluator {
  __REGISTER_CLASS

  virtual float operator()(PRule, PExampleTable, const int &, const int &targetClass, PDistribution );
};

class ORANGE_API TRuleEvaluator_Laplace : public TRuleEvaluator {
  __REGISTER_CLASS

  virtual float operator()(PRule, PExampleTable, const int &, const int &targetClass, PDistribution );
};

class ORANGE_API TEVCDist : public TOrange {
public:
  __REGISTER_CLASS

  float mu; //P mu of Fisher-Tippett distribution
  float beta; //P beta of Fisher-Tippett distribution
  PFloatList percentiles; //P 10 values - 0 = 5th percentile, 1 = 15th percentile, 9 = 95th percentile

  TEVCDist();
  TEVCDist(const float &, const float &, PFloatList &);
  double getProb(const float & chi);
  float median();
};

WRAPPER(EVCDistGetter)
class ORANGE_API TEVCDistGetter: public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PEVCDist operator()(const PRule , const int &) const = 0;
};

class ORANGE_API TEVCDistGetter_Standard: public TEVCDistGetter {
public:
  __REGISTER_CLASS

  PEVCDistList dists; //P EVC distribution (sorted by rule length, 0 = for rules without conditions)
  TEVCDistGetter_Standard();
  TEVCDistGetter_Standard(PEVCDistList);
  virtual PEVCDist operator()(const PRule, const int &) const;
};

WRAPPER(ChiFunction)
class ORANGE_API TChiFunction: public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual float operator()(PRule rule, PExampleTable data, const int & weightID, const int & targetClass, PDistribution apriori, float & nonOptimistic_Chi) const = 0;
};

class ORANGE_API TChiFunction_2LOGLR: public TChiFunction {
public:
  __REGISTER_CLASS

  virtual float operator()(PRule rule, PExampleTable data, const int & weightID, const int & targetClass, PDistribution apriori, float & nonOptimistic_Chi) const;
};

class DiffFunc {
public:
  virtual double operator()(float) const = 0;
};

class LNLNChiSq: public DiffFunc {
public:
  PEVCDist evc;
  float chi;
  double extremeAlpha;

  LNLNChiSq(PEVCDist evc, const float & chi);
  double operator()(float chix) const;
};

class LRInv: public DiffFunc {
public:
  float pn, P, N, chiCorrected;

  LRInv(const float & pn, const float & P, const float & PN, const float & chiCorrected);
  double operator()(float p) const;
};


class ORANGE_API TRuleEvaluator_mEVC: public TRuleEvaluator {
public:
  __REGISTER_CLASS

  int m; //P Parameter m for m-estimate after EVC correction
  PChiFunction chiFunction; //P function for computing chi square significance
  PEVCDistGetter evcDistGetter; //P get EVC distribution for chi correction
  PVariable probVar;//P probability coverage variable (meta usually)
  PRuleValidator validator; //P rule validator for best rule
  int min_improved; //P minimal number of improved examples
  float min_improved_perc; //P minimal percentage of improved examples
  PRule bestRule; //P best rule found and evaluated given conditions (min_improved, validator)

  TRuleEvaluator_mEVC();
  TRuleEvaluator_mEVC(const int & m, PChiFunction, PEVCDistGetter, PVariable, PRuleValidator, const int & min_improved, const float & min_improved_perc);
  void reset();
  float chiAsimetryCorrector(const float &);
  float evaluateRule(PRule rule, PExampleTable examples, const int & weightID, const int &targetClass, PDistribution apriori, const int & rLength, const float & aprioriProb) const;
  float operator()(PRule, PExampleTable, const int &, const int &targetClass, PDistribution );
};


class ORANGE_API TRuleEvaluator_LRS : public TRuleEvaluator {
public:
  __REGISTER_CLASS

  PRuleList rules; //P
  bool storeRules; //P

  TRuleEvaluator_LRS(const bool & = false);
  virtual float operator()(PRule, PExampleTable, const int &, const int &targetClass, PDistribution );
};

WRAPPER(RuleFinder)
class ORANGE_API TRuleFinder : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  PRuleValidator validator; //P
  PRuleEvaluator evaluator; //P

  virtual PRule operator()(PExampleTable, const int & =0, const int &targetClass = -1, PRuleList baseRules = PRuleList()) = 0;
};


WRAPPER(RuleBeamInitializer)
class ORANGE_API TRuleBeamInitializer : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PRuleList operator()(PExampleTable, const int &weightID, const int &targetClass, PRuleList baseRules, PRuleEvaluator, PDistribution apriori, PRule &bestRule) = 0;
};


class ORANGE_API TRuleBeamInitializer_Default : public TRuleBeamInitializer {
public:
  __REGISTER_CLASS

  virtual PRuleList operator()(PExampleTable, const int &weightID, const int &targetClass, PRuleList baseRules, PRuleEvaluator, PDistribution apriori, PRule &bestRule);
};


WRAPPER(RuleBeamRefiner)
class ORANGE_API TRuleBeamRefiner : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PRuleList operator()(PRule rule, PExampleTable, const int &weightID, const int &targetClass = -1) = 0;
};


class ORANGE_API TRuleBeamRefiner_Selector : public TRuleBeamRefiner {
public:
  __REGISTER_CLASS

  PDiscretization discretization; //P discretization for continuous attributes
  
  virtual PRuleList operator()(PRule rule, PExampleTable, const int &weightID, const int &targetClass = -1);
};


WRAPPER(RuleBeamCandidateSelector)
class ORANGE_API TRuleBeamCandidateSelector : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PRuleList operator()(PRuleList &existingRules, PExampleTable, const int &weightID) = 0;
};


class ORANGE_API TRuleBeamCandidateSelector_TakeAll : public TRuleBeamCandidateSelector {
public:
  __REGISTER_CLASS

  virtual PRuleList operator()(PRuleList &existingRules, PExampleTable, const int &weightID);
};


WRAPPER(RuleBeamFilter)
class ORANGE_API TRuleBeamFilter : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual void operator()(PRuleList &existingRules, PExampleTable, const int &weightID) = 0;
};


class ORANGE_API TRuleBeamFilter_Width : public TRuleBeamFilter {
public:
  __REGISTER_CLASS

  int width; //P

  TRuleBeamFilter_Width(const int &w = 5);

  void operator()(PRuleList &rules, PExampleTable, const int &weightID);
};



class ORANGE_API TRuleBeamFinder : public TRuleFinder {
public:
  __REGISTER_CLASS

  PRuleBeamInitializer initializer; //P
  PRuleBeamRefiner refiner; //P
  PRuleBeamCandidateSelector candidateSelector; //P
  PRuleBeamFilter ruleFilter; //P
  PRuleValidator ruleStoppingValidator; //P
  
  PRule operator()(PExampleTable, const int & =0, const int &targetClass = -1, PRuleList baseRules = PRuleList());
};



WRAPPER(RuleDataStoppingCriteria)
class ORANGE_API TRuleDataStoppingCriteria : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual bool operator()(PExampleTable, const int &weightID, const int &targetClass) const = 0;
};


class ORANGE_API TRuleDataStoppingCriteria_NoPositives : public TRuleDataStoppingCriteria {
public:
  __REGISTER_CLASS

  virtual bool operator()(PExampleTable, const int &weightID, const int &targetClass) const;
};


WRAPPER(RuleStoppingCriteria)
class ORANGE_API TRuleStoppingCriteria : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual bool operator()(PRuleList, PRule, PExampleTable, const int &weightID) const = 0;
};

class ORANGE_API TRuleStoppingCriteria_NegativeDistribution : public TRuleStoppingCriteria {
public:
  __REGISTER_CLASS

  virtual bool operator()(PRuleList, PRule, PExampleTable, const int &weightID) const;
};

WRAPPER(RuleCovererAndRemover)
class ORANGE_API TRuleCovererAndRemover : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

   virtual PExampleTable operator()(PRule, PExampleTable, const int &weightID, int &newWeight, const int &targetClass) const = 0;
};


class ORANGE_API TRuleCovererAndRemover_Default : public TRuleCovererAndRemover {
public:
  __REGISTER_CLASS

  virtual PExampleTable operator()(PRule, PExampleTable, const int &weightID, int &newWeight, const int &targetClass) const;
};

WRAPPER(RuleClassifierConstructor)
WRAPPER(RuleClassifier)
class ORANGE_API TRuleClassifierConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PRuleClassifier operator()(PRuleList, PExampleTable, const int &weightID = 0) = 0;
};


class ORANGE_API TRuleClassifierConstructor_firstRule: public TRuleClassifierConstructor {
  __REGISTER_CLASS
  virtual PRuleClassifier operator()(PRuleList, PExampleTable, const int &weightID = 0);
};


WRAPPER(RuleLearner)
class ORANGE_API TRuleLearner : public TLearner {
public:
  __REGISTER_CLASS
  
  PRuleDataStoppingCriteria dataStopping; //P
  PRuleStoppingCriteria ruleStopping; //P
  PRuleCovererAndRemover coverAndRemove; //P
  PRuleFinder ruleFinder; //P
  PRuleClassifierConstructor classifierConstructor; //P classifier

  PProgressCallback progressCallback; //P progress callback function

  bool storeExamples; //P
  int targetClass; //P
  PRuleList baseRules; //P

  TRuleLearner(bool storeExamples = true, int targetClass = -1, PRuleList baseRules = PRuleList());

  PClassifier operator()(PExampleGenerator, const int & =0);
  PClassifier operator()(PExampleGenerator, const int &, const int &targetClass = -1, PRuleList baseRules = PRuleList());
};



class ORANGE_API TRuleClassifier : public TClassifier {
public:
  __REGISTER_ABSTRACT_CLASS

  PRuleList rules; //P
  PExampleTable examples; //P
  int weightID; //P

  TRuleClassifier();
  TRuleClassifier(PRuleList rules, PExampleTable examples, const int &weightID = 0);

  virtual PDistribution classDistribution(const TExample &ex) = 0;
};

// Zakaj moram se enkrat definirati konstruktor;
class ORANGE_API TRuleClassifier_firstRule : public TRuleClassifier {
public:
  __REGISTER_CLASS

  PDistribution prior; //P prior distribution

  TRuleClassifier_firstRule();
  TRuleClassifier_firstRule(PRuleList rules, PExampleTable examples, const int &weightID = 0);
  virtual PDistribution classDistribution(const TExample &ex);
};

// Logit rule based classifier;
class ORANGE_API TRuleClassifier_logit : public TRuleClassifier {
public:
  __REGISTER_CLASS

  PDistribution prior; //P prior distribution
  PDomain domain; //P Domain
  PFloatList ruleBetas; //P Rule betas
  PFloatList priorProbBetas; //P Prior probabilitiy betas

  float eval, **f, **p, **tempP, **tempF, *betas, *priorBetas, *minbetas;
  PFloatList wavgCov, wavgCovPrior, wsd, wsdPrior, wpriorProb, wavgProb;
  PIntList *ruleIndices, **coveredRules;
  int psize,fsize;
  bool useBestRuleOnly;
  float minBeta; //P minimum beta value of a rule, if lower, rule is set to have beta 0. 

  PClassifier priorClassifier;

  TRuleClassifier_logit();
  TRuleClassifier_logit(PRuleList rules, const float &minBeta, PExampleTable examples, const int &weightID = 0, const PClassifier &classifer = NULL, const PDistributionList &probList = NULL, const bool &useBestRuleOnly = false);
  ~TRuleClassifier_logit();

  void initialize(const PDistributionList &);
  void updateRuleBetas(float step);
  float cutOptimisticBetas(float step, float curr_eval);
  void correctPriorBetas(float step);
  void distortPriorBetas(float step);
  float findMax(int clIndex, int exIndex);
  float compPotEval(int ruleIndex, int classIndex, float newBeta, float **tempF, float **tempP, PFloatList &wavgProb, PFloatList &wpriorProb);
  int getClassIndex(PRule r);
  void addPriorClassifier(const TExample &, double *);
  virtual PDistribution classDistribution(const TExample &ex);
};

class ORANGE_API TRuleClassifier_logit_bestRule : public TRuleClassifier_logit {
public:
  __REGISTER_CLASS

  TRuleClassifier_logit_bestRule();
  TRuleClassifier_logit_bestRule(PRuleList rules, const float &minBeta, PExampleTable examples, const int &weightID = 0, const PClassifier &classifer = NULL, const PDistributionList &probList = NULL);
};

#endif
