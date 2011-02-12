"""

.. index:: rule induction

.. index:: 
   single: classification; rule induction

**************
Rule induction
**************

Orange implements several supervised rule induction algorithms
and rule-based classification methods. First, there is an implementation of the classic 
`CN2 induction algorithm <http://www.springerlink.com/content/k6q2v76736w5039r/>`_. 
The implementation of CN2 is modular, providing the oportunity to change, specialize
and improve the algorithm. The implementation is thus based on the rule induction 
framework that we describe below.

CN2 algorithm
=============

.. index:: 
   single: classification; CN2

Several variations of well-known CN2 rule learning algorithms are implemented.
All are implemented by wrapping the
:class:`Orange.classification.rules.RuleLearner` class. Each CN2 learner class
in this module changes some of RuleLearner's replaceable components to reflect
the required behavior.

Usage is consistent with typical learner usage in Orange:

`rules-cn2.py`_ (uses `titanic.tab`_)

.. literalinclude:: code/rules-cn2.py
    :lines: 7-

.. _rules-cn2.py: code/rules-cn2.py
.. _titanic.tab: code/titanic.tab

This is the resulting printout::
    
    IF sex=['female'] AND status=['first'] AND age=['child'] THEN survived=yes<0.000, 1.000>
    IF sex=['female'] AND status=['second'] AND age=['child'] THEN survived=yes<0.000, 13.000>
    IF sex=['male'] AND status=['second'] AND age=['child'] THEN survived=yes<0.000, 11.000>
    IF sex=['female'] AND status=['first'] THEN survived=yes<4.000, 140.000>
    IF status=['first'] AND age=['child'] THEN survived=yes<0.000, 5.000>
    IF sex=['male'] AND status=['second'] THEN survived=no<154.000, 14.000>
    IF status=['crew'] AND sex=['female'] THEN survived=yes<3.000, 20.000>
    IF status=['second'] THEN survived=yes<13.000, 80.000>
    IF status=['third'] AND sex=['male'] AND age=['adult'] THEN survived=no<387.000, 75.000>
    IF status=['crew'] THEN survived=no<670.000, 192.000>
    IF age=['child'] AND sex=['male'] THEN survived=no<35.000, 13.000>
    IF sex=['male'] THEN survived=no<118.000, 57.000>
    IF age=['child'] THEN survived=no<17.000, 14.000>
    IF TRUE THEN survived=no<89.000, 76.000>
    
.. autoclass:: Orange.classification.rules.CN2Learner
   :members:
   :show-inheritance:
   
.. autoclass:: Orange.classification.rules.CN2Classifier
   :members:
   :show-inheritance:
   
.. index:: unordered CN2

.. index:: 
   single: classification; unordered CN2

.. autoclass:: Orange.classification.rules.CN2UnorderedLearner
   :members:
   :show-inheritance:
   
.. autoclass:: Orange.classification.rules.CN2UnorderedClassifier
   :members:
   :show-inheritance:
   
.. index:: CN2-SD
.. index:: subgroup discovery

.. index:: 
   single: classification; CN2-SD
   
.. autoclass:: Orange.classification.rules.CN2SDUnorderedLearner
   :members:
   :show-inheritance:
   
.. autoclass:: Orange.classification.rules.CN2EVCUnorderedLearner
   :members:
   :show-inheritance:
   
References
----------

* Clark, Niblett. `The CN2 Induction Algorithm
  <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.9180>`_. Machine
  Learning 3(4):261--284, 1989.
* Clark, Boswell. `Rule Induction with CN2: Some Recent Improvements
  <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.1700>`_. In
  Machine Learning - EWSL-91. Proceedings of the European Working Session on
  Learning., pages 151--163, Porto, Portugal, March 1991.
* Lavrac, Kavsek, Flach, Todorovski: `Subgroup Discovery with CN2-SD
  <http://jmlr.csail.mit.edu/papers/volume5/lavrac04a/lavrac04a.pdf>`_. Journal
  of Machine Learning Research 5: 153-188, 2004.


Rule induction framework
========================

A general framework of classes supports the described CN2 implementation, and
can in fact be fine-tuned to specific needs by replacing individual components.
Here is a simple example, while a detailed architecture can be observed
in description of classes that follows it:

part of `rules-customized.py`_ (uses `titanic.tab`_)

.. literalinclude:: code/rules-customized.py
    :lines: 7-17

.. _rules-customized.py: code/rules-customized.py

In the example, the rule evaluation function was set to an m-estimate of
probability with m=50. The result is::

    IF sex=['male'] AND status=['second'] AND age=['adult'] THEN survived=no<154.000, 14.000>
    IF sex=['male'] AND status=['third'] AND age=['adult'] THEN survived=no<387.000, 75.000>
    IF sex=['female'] AND status=['first'] THEN survived=yes<4.000, 141.000>
    IF status=['crew'] AND sex=['male'] THEN survived=no<670.000, 192.000>
    IF status=['second'] THEN survived=yes<13.000, 104.000>
    IF status=['third'] AND sex=['male'] THEN survived=no<35.000, 13.000>
    IF status=['first'] AND age=['adult'] THEN survived=no<118.000, 57.000>
    IF status=['crew'] THEN survived=yes<3.000, 20.000>
    IF sex=['female'] THEN survived=no<106.000, 90.000>
    IF TRUE THEN survived=yes<0.000, 5.000>

Notice that we first need to set the ruleFinder component, because the default
components are not constructed when the learner is constructed, but only when
we run it on data. At that time, the algorithm checks which components are
necessary and sets defaults. Similarly, when the learner finishes, it destructs
all *default* components. Continuing with our example, assume that we wish to
set a different validation function and a different bean width. This is simply
written as:

part of `rules-customized.py`_ (uses `titanic.tab`_)

.. literalinclude:: code/rules-customized.py
    :lines: 19-23


.. py:class:: Orange.classification.rules.Rule(filter, classifier, lr, dist, ce, w = 0, qu = -1)
   
   Base class for presentation of a single induced rule.
   
   Parameters, that can be passed to the constructor, correspond to the first
   7 attributes. All attributes are:
   
   .. attribute:: filter
   
      contents of the rule; this is the basis of the Rule class. Must be of
      type :class:`Orange.core.Filter`; an instance of
      :class:`Orange.core.Filter_values` is set as a default.
   
   .. attribute:: classifier
      
      each rule can be used as a classical Orange like
      classifier. Must be of type :class:`Orange.classification.Classifier`.
      By default, an instance of :class:`Orange.core.DefaultClassifier` is used.
   
   .. attribute:: learner
      
      learner to be used for making a classifier. Must be of type
      :class:`Orange.core.learner`. By default,
      :class:`Orange.core.MajorityLearner` is used.
   
   .. attribute:: classDistribution
      
      distribution of class in data instances covered by this rule
      (:class:`Orange.core.Distribution`).
   
   .. attribute:: examples
      
      data instances covered by this rule (:class:`Orange.data.Table`).
   
   .. attribute:: weightID
   
      ID of the weight meta-attribute for the stored data instances (int).
   
   .. attribute:: quality
      
      quality of the rule. Rules with higher quality are better (float).
   
   .. attribute:: complexity
   
      complexity of the rule (float). Complexity is used for
      selecting between rules with equal quality, where rules with lower
      complexity are preferred. Typically, complexity corresponds to the
      number of selectors in rule (actually to number of conditions in filter),
      but, obviously, any other measure can be applied.
   
   .. method:: filterAndStore(instances, weightID=0, targetClass=-1)
   
      Filter passed data instances and store them in the attribute 'examples'.
      Also, compute classDistribution, set weight of stored examples and create
      a new classifier using 'learner' attribute.
      
      :param weightID: ID of the weight meta-attribute.
      :type weightID: int
      :param targetClass: index of target class; -1 for all.
      :type targetClass: int
   
   Objects of this class can be invoked:

   .. method:: __call__(instance, instances, weightID=0, targetClass=-1)
   
      There are two ways of invoking this method. One way is only passing the
      data instance; then the Rule object returns True if the given instance is
      covered by the rule's filter.
      
      :param instance: data instance.
      :type instance: :class:`Orange.data.Instance`
      
      Another way of invocation is passing a table of data instances,
      in which case a table of instances, covered by this rule, is returned.
      
      :param instances: a table of data instances.
      :type instances: :class:`Orange.data.Table`
      :param ref: TODO
      :type ref: bool
      :param negate: if set to True, the result is inverted: the resulting
          table contains instances *not* covered by the rule.
      :type negate: bool

.. py:class:: Orange.classification.rules.RuleLearner(storeInstances = true, targetClass = -1, baseRules = Orange.classification.rules.RuleList())
   
   Bases: :class:`Orange.core.Learner`
   
   A base rule induction learner. The algorithm follows separate-and-conquer
   strategy, which has its origins in the AQ family of algorithms
   (Fuernkranz J.; Separate-and-Conquer Rule Learning, Artificial Intelligence
   Review 13, 3-54, 1999). Basically, such algorithms search for the "best"
   possible rule in learning instancess, remove covered data from learning
   instances (separate) and repeat the process (conquer) on the remaining
   instances.
   
   The class' functionality can be best explained by showing its __call__
   function:
   
   .. parsed-literal::

      def \_\_call\_\_(self, instances, weightID=0):
          ruleList = Orange.classification.rules.RuleList()
          allInstances = Orange.data.Table(instances)
          while not self.\ **dataStopping**\ (instances, weightID, self.targetClass):
              newRule = self.\ **ruleFinder**\ (instances, weightID, self.targetClass,
                                        self.baseRules)
              if self.\ **ruleStopping**\ (ruleList, newRule, instances, weightID):
                  break
              instances, weightID = self.\ **coverAndRemove**\ (newRule, instances,
                                                      weightID, self.targetClass)
              ruleList.append(newRule)
          return Orange.classification.rules.RuleClassifier_FirstRule(
              rules=ruleList, instances=allInstances)
                
   The four customizable components here are the invoked dataStopping,
   ruleFinder, coverAndRemove and ruleStopping objects. By default, components
   of the original CN2 algorithm will be used, but this can be changed by
   modifying those attributes:
   
   .. attribute:: dataStopping
   
      an object of class
      :class:`Orange.classification.rules.RuleDataStoppingCriteria`
      that determines whether there will be any benefit from further learning
      (ie. if there is enough data to continue learning). The default
      implementation
      (:class:`Orange.classification.rules.RuleDataStoppingCriteria_NoPositives`)
      returns True if there are no more instances of given class. 
   
   .. attribute:: ruleStopping
      
      an object of class 
      :class:`Orange.classification.rules.RuleStoppingCriteria`
      that decides from the last rule learned if it is worthwhile to use the
      rule and learn more rules. By default, no rule stopping criteria is
      used (ruleStopping==None), thus accepting all rules.
       
   .. attribute:: coverAndRemove
       
      an object of class
      :class:`Orange.classification.rules.RuleCovererAndRemover`
      that removes instances covered by the rule and returns remaining
      instances. The default implementation
      (:class:`Orange.classification.rules.RuleCovererAndRemover_Default`)
      only removes the instances that belong to given target class, except if
      it is not given (ie. targetClass==-1).
    
   .. attribute:: ruleFinder
      
      an object of class
      :class:`Orange.classification.rules.RuleFinder` that learns a single
      rule from instances. Default implementation is
      :class:`Orange.classification.rules.RuleBeamFinder`.

   Constructor can be given the following parameters:
    
   :param storeInstances: if set to True, the rules will have data instances
       stored.
   :type storeInstances: bool
    
   :param targetClass: index of a specific class being learned; -1 for all.
   :type targetClass: int
   
   :param baseRules: Rules that we would like to use in ruleFinder to
       constrain the learning space. If not set, it will be set to a set
       containing only an empty rule.
   :type baseRules: :class:`Orange.classification.rules.RuleList`

Rule finders
------------

.. class:: Orange.classification.rules.RuleFinder

   Base class for all rule finders. These are used to learn a single rule from
   instances.
   
   Rule finders are invokable in the following manner:
   
   .. method:: __call__(table, weightID, targetClass, baseRules)
   
      Return a new rule, induced from instances in the given table.
      
      :param table: data instances to learn from.
      :type table: :class:`Orange.data.Table`
      
      :param weightID: ID of the weight meta-attribute for the stored data
          instances.
      :type weightID: int
      
      :param targetClass: index of a specific class being learned; -1 for all.
      :type targetClass: int 
      
      :param baseRules: Rules that we would like to use in ruleFinder to
          constrain the learning space. If not set, it will be set to a set
          containing only an empty rule.
      :type baseRules: :class:`Orange.classification.rules.RuleList`

.. class:: Orange.classification.rules.RuleBeamFinder
   
   Bases: :class:`Orange.classification.rules.RuleFinder`
   
   Beam search for the best rule. This is the default class used in RuleLearner
   to find the best rule. Pseudo code of the algorithm is shown here:

   .. parsed-literal::

      def \_\_call\_\_(self, table, weightID, targetClass, baseRules):
          prior = orange.Distribution(table.domain.classVar, table, weightID)
          rulesStar, bestRule = self.\ **initializer**\ (table, weightID, targetClass, baseRules, self.evaluator, prior)
          \# compute quality of rules in rulesStar and bestRule
          ...
          while len(rulesStar) \> 0:
              candidates, rulesStar = self.\ **candidateSelector**\ (rulesStar, table, weightID)
              for cand in candidates:
                  newRules = self.\ **refiner**\ (cand, table, weightID, targetClass)
                  for newRule in newRules:
                      if self.\ **ruleStoppingValidator**\ (newRule, table, weightID, targetClass, cand.classDistribution):
                          newRule.quality = self.\ **evaluator**\ (newRule, table, weightID, targetClass, prior)
                          rulesStar.append(newRule)
                          if self.\ **validator**\ (newRule, table, weightID, targetClass, prior) and
                              newRule.quality \> bestRule.quality:
                              bestRule = newRule
              rulesStar = self.\ **ruleFilter**\ (rulesStar, table, weightID)
          return bestRule

   Bolded in the pseudo-code are several exchangeable components, exposed as
   attributes. These are:

   .. attribute:: initializer
   
      an object of class
      :class:`Orange.classification.rules.RuleBeamInitializer`
      used to initialize rulesStar and for selecting the
      initial best rule. By default
      (:class:`Orange.classification.rules.RuleBeamInitializer_Default`),
      baseRules are returned as starting rulesSet and the best from baseRules
      is set as bestRule. If baseRules are not set, this class will return
      rulesStar with rule that covers all instances (has no selectors) and
      this rule will be also used as bestRule.
   
   .. attribute:: candidateSelector
   
      an object of class
      :class:`Orange.classification.rules.RuleBeamCandidateSelector`
      used to separate a subset from the current
      rulesStar and return it. These rules will be used in the next
      specification step. Default component (an instance of
      :class:`Orange.classification.rules.RuleBeamCandidateSelector_TakeAll`)
      takes all rules in rulesStar
    
   .. attribute:: refiner
   
      an object of class
      :class:`Orange.classification.rules.RuleBeamRefiner`
      used to refine given rule. New rule should cover a
      strict subset of examples covered by given rule. Default component
      (:class:`Orange.classification.rules.RuleBeamRefiner_Selector`) adds
      a conjunctive selector to selectors present in the rule.
    
   .. attribute:: ruleFilter
   
      an object of class
      :class:`Orange.classification.rules.RuleBeamFilter`
      used to filter rules to keep beam relatively small
      to contain search complexity. By default, it takes five best rules:
      :class:`Orange.classification.rules.RuleBeamFilter_Width`\ *(m=5)*\ .

   .. method:: __call__(data, weightID, targetClass, baseRules)

   Determines the next best rule to cover the remaining data instances.
   
   :param data: data instances.
   :type data: :class:`Orange.data.Table`
   
   :param weightID: index of the weight meta-attribute.
   :type weightID: int
   
   :param targetClass: index of the target class.
   :type targetClass: int
   
   :param baseRules: existing rules.
   :type baseRules: :class:`Orange.classification.rules.RuleList`

Rule evaluators
---------------

.. class:: Orange.classification.rules.RuleEvaluator

   Base class for rule evaluators that evaluate the quality of the rule based
   on covered data instances. All evaluators support being invoked in the
   following manner:
   
   .. method:: __call__(rule, instances, weightID, targetClass, prior)
   
      Calculates a non-negative rule quality.
      
      :param rule: rule to evaluate.
      :type rule: :class:`Orange.classification.rules.Rule`
      
      :param instances: a table of instances, covered by the rule.
      :type instances: :class:`Orange.data.Table`
      
      :param weightID: index of the weight meta-attribute.
      :type weightID: int
      
      :param targetClass: index of target class of this rule.
      :type targetClass: int
      
      :param prior: prior class distribution.
      :type prior: :class:`Orange.core.Distribution`

.. autoclass:: Orange.classification.rules.LaplaceEvaluator
   :members:
   :show-inheritance:

.. autoclass:: Orange.classification.rules.WRACCEvaluator
   :members:
   :show-inheritance:
   
.. class:: Orange.classification.rules.RuleEvaluator_Entropy

   Bases: :class:`Orange.classification.rules.RuleEvaluator`
    
.. class:: Orange.classification.rules.RuleEvaluator_LRS

   Bases: :class:`Orange.classification.rules.RuleEvaluator`

.. class:: Orange.classification.rules.RuleEvaluator_Laplace

   Bases: :class:`Orange.classification.rules.RuleEvaluator`

.. class:: Orange.classification.rules.RuleEvaluator_mEVC

   Bases: :class:`Orange.classification.rules.RuleEvaluator`
   
Instance covering and removal
-----------------------------

.. class:: RuleCovererAndRemover

   Base class for rule coverers and removers that, when invoked, remove
   instances covered by the rule and return remaining instances.

   .. method:: __call__(rule, instances, weights, targetClass)
   
      Calculates a non-negative rule quality.
      
      :param rule: rule to evaluate.
      :type rule: :class:`Orange.classification.rules.Rule`
      
      :param instances: a table of instances, covered by the rule.
      :type instances: :class:`Orange.data.Table`
      
      :param weights: index of the weight meta-attribute.
      :type weights: int
      
      :param targetClass: index of target class of this rule.
      :type targetClass: int

.. autoclass:: CovererAndRemover_MultWeights

.. autoclass:: CovererAndRemover_AddWeights
   
Miscellaneous functions
-----------------------

.. automethod:: Orange.classification.rules.ruleToString

..
    Undocumented are:
    Data-based Stopping Criteria
    ----------------------------
    Rule-based Stopping Criteria
    ----------------------------
    Rule-based Stopping Criteria
    ----------------------------

"""

from Orange.core import \
    AssociationClassifier, \
    AssociationLearner, \
    RuleClassifier, \
    RuleClassifier_firstRule, \
    RuleClassifier_logit, \
    RuleLearner, \
    Rule, \
    RuleBeamCandidateSelector, \
    RuleBeamCandidateSelector_TakeAll, \
    RuleBeamFilter, \
    RuleBeamFilter_Width, \
    RuleBeamInitializer, \
    RuleBeamInitializer_Default, \
    RuleBeamRefiner, \
    RuleBeamRefiner_Selector, \
    RuleClassifierConstructor, \
    RuleCovererAndRemover, \
    RuleCovererAndRemover_Default, \
    RuleDataStoppingCriteria, \
    RuleDataStoppingCriteria_NoPositives, \
    RuleEvaluator, \
    RuleEvaluator_Entropy, \
    RuleEvaluator_LRS, \
    RuleEvaluator_Laplace, \
    RuleEvaluator_mEVC, \
    RuleFinder, \
    RuleBeamFinder, \
    RuleList, \
    RuleStoppingCriteria, \
    RuleStoppingCriteria_NegativeDistribution, \
    RuleValidator, \
    RuleValidator_LRS

import Orange.core
import random
import math
from orngABCN2 import ABCN2


class LaplaceEvaluator(RuleEvaluator):
    """
    Laplace's rule of succession.
    """
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if not rule.classDistribution:
            return 0.
        sumDist = rule.classDistribution.cases
        if not sumDist or (targetClass>-1 and not rule.classDistribution[targetClass]):
            return 0.
        # get distribution
        if targetClass>-1:
            return (rule.classDistribution[targetClass]+1)/(sumDist+2)
        else:
            return (max(rule.classDistribution)+1)/(sumDist+len(data.domain.classVar.values))


class WRACCEvaluator(RuleEvaluator):
    """
    Weighted relative accuracy.
    """
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if not rule.classDistribution:
            return 0.
        sumDist = rule.classDistribution.cases
        if not sumDist or (targetClass>-1 and not rule.classDistribution[targetClass]):
            return 0.
        # get distribution
        if targetClass>-1:
            pRule = rule.classDistribution[targetClass]/apriori[targetClass]
            pTruePositive = rule.classDistribution[targetClass]/sumDist
            pClass = apriori[targetClass]/apriori.cases
        else:
            pRule = sumDist/apriori.cases
            pTruePositive = max(rule.classDistribution)/sumDist
            pClass = apriori[rule.classDistribution.modus()]/sum(apriori)
        if pTruePositive>pClass:
            return pRule*(pTruePositive-pClass)
        else: return (pTruePositive-pClass)/max(pRule,1e-6)


class MEstimateEvaluator(RuleEvaluator):
    """
    Rule evaluator using m-estimate of probability rule evaluation function.
    
    :param m: m-value for m-estimate
    :type m: int
    
    """
    def __init__(self, m=2):
        self.m = m
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if not rule.classDistribution:
            return 0.
        sumDist = rule.classDistribution.abs
        if self.m == 0 and not sumDist:
            return 0.
        # get distribution
        if targetClass>-1:
            p = rule.classDistribution[targetClass]+self.m*apriori[targetClass]/apriori.abs
            p = p / (rule.classDistribution.abs + self.m)
        else:
            p = max(rule.classDistribution)+self.m*apriori[rule.\
                classDistribution.modus()]/apriori.abs
            p = p / (rule.classDistribution.abs + self.m)      
        return p


class CN2Learner(RuleLearner):
    """
    Classical CN2 (see Clark and Niblett; 1988) induces a set of ordered
    rules, which means that classificator must try these rules in the same
    order as they were learned.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    Constructor can be given the following parameters:
    
    :param evaluator: an object that evaluates a rule from covered instances.
        By default, entropy is used as a measure. 
    :type evaluator: :class:`Orange.classification.rules.RuleEvaluator`
    :param beamWidth: width of the search beam.
    :type beamWidth: int
    :param alpha: significance level of the statistical test to determine
        whether rule is good enough to be returned by rulefinder. Likelihood
        ratio statistics is used that gives an estimate if rule is
        statistically better than the default rule.
    :type alpha: float

    """
    
    def __new__(cls, instances=None, weightID=0, **kwargs):
        self = RuleLearner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(**kwargs)
            return self.__call__(instances, weightID)
        else:
            return self
        
    def __init__(self, evaluator = RuleEvaluator_Entropy(), beamWidth = 5,
        alpha = 1.0, **kwds):
        self.__dict__.update(kwds)
        self.ruleFinder = RuleBeamFinder()
        self.ruleFinder.ruleFilter = RuleBeamFilter_Width(width = beamWidth)
        self.ruleFinder.evaluator = evaluator
        self.ruleFinder.validator = RuleValidator_LRS(alpha = alpha)
        
    def __call__(self, instances, weight=0):
        supervisedClassCheck(instances)
        
        cl = RuleLearner.__call__(self,instances,weight)
        rules = cl.rules
        return CN2Classifier(rules, instances, weight)


class CN2Classifier(RuleClassifier):
    """
    Classical CN2 (see Clark and Niblett; 1988) classifies a new instance
    using an ordered set of rules. Usually the learner
    (:class:`Orange.classification.rules.CN2Learner`) is used to construct the
    classifier.
    
    When constructing the classifier manually, the following parameters can
    be passed:
    
    :param rules: learned rules to be used for classification (mandatory).
    :type rules: :class:`Orange.classification.rules.RuleList`
    
    :param instances: data instances that were used for learning.
    :type instances: :class:`Orange.data.Table`
    
    :param weightID: ID of the weight meta-attribute.
    :type weightID: int

    """
    
    def __init__(self, rules=None, instances=None, weightID = 0, **argkw):
        self.rules = rules
        self.examples = instances
        self.weightID = weightID
        self.classVar = None if instances is None else instances.domain.classVar
        self.__dict__.update(argkw)
        if instances is not None:
            self.prior = Orange.core.Distribution(instances.domain.classVar,instances)

    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue):
        """
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        classifier = None
        for r in self.rules:
         #   r.filter.domain = instance.domain
            if r(instance) and r.classifier:
                classifier = r.classifier
                classifier.defaultDistribution = r.classDistribution
                break
        if not classifier:
            classifier = Orange.core.DefaultClassifier(instance.domain.classVar,\
                self.prior.modus())
            classifier.defaultDistribution = self.prior

        if result_type == Orange.classification.Classifier.GetValue:
          return classifier(instance)
        if result_type == Orange.classification.Classifier.GetProbabilities:
          return classifier.defaultDistribution
        return (classifier(instance),classifier.defaultDistribution)

    def __str__(self):
        retStr = ruleToString(self.rules[0])+" "+str(self.rules[0].\
            classDistribution)+"\n"
        for r in self.rules[1:]:
            retStr += "ELSE "+ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr


class CN2UnorderedLearner(RuleLearner):
    """
    CN2 unordered (see Clark and Boswell; 1991) induces a set of unordered
    rules - classification from rules does not assume ordering of rules.
    Learning rules is quite similar to learning in classical CN2, where
    the process of learning of rules is separated to learning rules for each
    class.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    Constructor can be given the following parameters:
    
    :param evaluator: an object that evaluates a rule from covered instances.
        By default, Laplace's rule of succession is used as a measure. 
    :type evaluator: :class:`Orange.classification.rules.RuleEvaluator`
    :param beamWidth: width of the search beam.
    :type beamWidth: int
    :param alpha: significance level of the statistical test to determine
        whether rule is good enough to be returned by rulefinder. Likelihood
        ratio statistics is used that gives an estimate if rule is
        statistically better than the default rule.
    :type alpha: float
    """
    def __new__(cls, instances=None, weightID=0, **kwargs):
        self = RuleLearner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(**kwargs)
            return self.__call__(instances, weightID)
        else:
            return self
            
    def __init__(self, evaluator = RuleEvaluator_Laplace(), beamWidth = 5,
        alpha = 1.0, **kwds):
        self.__dict__.update(kwds)
        self.ruleFinder = RuleBeamFinder()
        self.ruleFinder.ruleFilter = RuleBeamFilter_Width(width = beamWidth)
        self.ruleFinder.evaluator = evaluator
        self.ruleFinder.validator = RuleValidator_LRS(alpha = alpha)
        self.ruleFinder.ruleStoppingValidator = RuleValidator_LRS(alpha = 1.0)
        self.ruleStopping = RuleStopping_Apriori()
        self.dataStopping = RuleDataStoppingCriteria_NoPositives()
        
    def __call__(self, instances, weight=0):
        supervisedClassCheck(instances)
        
        rules = RuleList()
        self.ruleStopping.apriori = Orange.core.Distribution(instances.\
            domain.classVar,instances)
        progress=getattr(self,"progressCallback",None)
        if progress:
            progress.start = 0.0
            progress.end = 0.0
            distrib = Orange.core.Distribution(instances.domain.classVar,\
                instances, weight)
            distrib.normalize()
        for targetClass in instances.domain.classVar:
            if progress:
                progress.start = progress.end
                progress.end += distrib[targetClass]
            self.targetClass = targetClass
            cl = RuleLearner.__call__(self,instances,weight)
            for r in cl.rules:
                rules.append(r)
        if progress:
            progress(1.0,None)
        return CN2UnorderedClassifier(rules, instances, weight)


class CN2UnorderedClassifier(RuleClassifier):
    """
    CN2 unordered (see Clark and Boswell; 1991) classifies a new instance using
    a set of unordered rules. Usually the learner
    (:class:`Orange.classification.rules.CN2UnorderedLearner`) is used to
    construct the classifier.
    
    When constructing the classifier manually, the following parameters can
    be passed:
    
    :param rules: learned rules to be used for classification (mandatory).
    :type rules: :class:`Orange.classification.rules.RuleList`
    
    :param instances: data instances that were used for learning.
    :type instances: :class:`Orange.data.Table`
    
    :param weightID: ID of the weight meta-attribute.
    :type weightID: int

    """
    def __init__(self, rules = None, instances = None, weightID = 0, **argkw):
        self.rules = rules
        self.examples = instances
        self.weightID = weightID
        self.classVar = instances.domain.classVar if instances is not None else None
        self.__dict__.update(argkw)
        if instances is not None:
            self.prior = Orange.core.Distribution(instances.domain.classVar, instances)

    def __call__(self, instance, result_type=Orange.core.GetValue, retRules = False):
        """
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        def add(disc1, disc2, sumd):
            disc = Orange.core.DiscDistribution(disc1)
            sumdisc = sumd
            for i,d in enumerate(disc):
                disc[i]+=disc2[i]
                sumdisc += disc2[i]
            return disc, sumdisc

        # create empty distribution
        retDist = Orange.core.DiscDistribution(self.examples.domain.classVar)
        covRules = RuleList()
        # iterate through instances - add distributions
        sumdisc = 0.
        for r in self.rules:
            if r(instance) and r.classDistribution:
                retDist, sumdisc = add(retDist, r.classDistribution, sumdisc)
                covRules.append(r)
        if not sumdisc:
            retDist = self.prior
            sumdisc = self.prior.abs
            
        if sumdisc > 0.0:
            for c in self.examples.domain.classVar:
                retDist[c] /= sumdisc
        else:
            retDist.normalize()
        
        if retRules:
            if result_type == Orange.classification.Classifier.GetValue:
              return (retDist.modus(), covRules)
            if result_type == Orange.classification.Classifier.GetProbabilities:
              return (retDist, covRules)
            return (retDist.modus(),retDist,covRules)
        if result_type == Orange.classification.Classifier.GetValue:
          return retDist.modus()
        if result_type == Orange.classification.Classifier.GetProbabilities:
          return retDist
        return (retDist.modus(),retDist)

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr


class CN2SDUnorderedLearner(CN2UnorderedLearner):
    """
    CN2-SD (see Lavrac et al.; 2004) induces a set of unordered rules, which
    is the same as :class:`Orange.classification.rules.CN2UnorderedLearner`.
    The difference between classical CN2 unordered and CN2-SD is selection of
    specific evaluation function and covering function:
    :class:`Orange.classifier.rules.WRACCEvaluator` is used to implement
    weight-relative accuracy and 
    :class:`Orange.classifier.rules.CovererAndRemover_MultWeight` avoids
    excluding covered instances, multiplying their weight by the value of
    mult parameter instead.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    Constructor can be given the following parameters:
    
    :param evaluator: an object that evaluates a rule from covered instances.
        By default, weighted relative accuracy is used.
    :type evaluator: :class:`Orange.classification.rules.RuleEvaluator`
    :param beamWidth: width of the search beam.
    :type beamWidth: int
    :param alpha: significance level of the statistical test to determine
        whether rule is good enough to be returned by rulefinder. Likelihood
        ratio statistics is used that gives an estimate if rule is
        statistically better than the default rule.
    :type alpha: float
    :param mult: multiplicator for weights of covered instances.
    :type mult: float
    """
    def __new__(cls, instances=None, weightID=0, **kwargs):
        self = CN2UnorderedLearner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(**kwargs)
            return self.__call__(instances, weightID)
        else:
            return self
        
    def __init__(self, evaluator = WRACCEvaluator(), beamWidth = 5,
                alpha = 0.05, mult=0.7, **kwds):
        CN2UnorderedLearnerClass.__init__(self, evaluator = evaluator,
                                          beamWidth = beamWidth, alpha = alpha, **kwds)
        self.coverAndRemove = CovererAndRemover_MultWeights(mult=mult)

    def __call__(self, instances, weight=0):        
        supervisedClassCheck(instances)
        
        oldInstances = Orange.data.Table(instances)
        classifier = CN2UnorderedLearnerClass.__call__(self,instances,weight)
        for r in classifier.rules:
            r.filterAndStore(oldInstances,weight,r.classifier.defaultVal)
        return classifier


class CN2EVCUnorderedLearner(ABCN2):
    """
    CN2-SD (see Lavrac et al.; 2004) induces a set of unordered rules in a
    simmilar manner as
    :class:`Orange.classification.rules.CN2SDUnorderedLearner`. This
    implementation uses the EVC rule evaluation.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    Constructor can be given the following parameters:
    
    :param evaluator: an object that evaluates a rule from covered instances.
        By default, weighted relative accuracy is used.
    :type evaluator: :class:`Orange.classification.rules.RuleEvaluator`
    :param beamWidth: width of the search beam.
    :type beamWidth: int
    :param alpha: significance level of the statistical test to determine
        whether rule is good enough to be returned by rulefinder. Likelihood
        ratio statistics is used that gives an estimate if rule is
        statistically better than the default rule.
    :type alpha: float
    :param mult: multiplicator for weights of covered instances.
    :type mult: float
    """
    def __init__(self, width=5, nsampling=100, rule_sig=1.0, att_sig=1.0,\
        min_coverage = 1., max_rule_complexity = 5.):
        ABCN2.__init__(self, width=width, nsampling=nsampling,
            rule_sig=rule_sig, att_sig=att_sig, min_coverage=int(min_coverage),
            max_rule_complexity = int(max_rule_complexity))


class RuleStopping_Apriori(RuleStoppingCriteria):
    def __init__(self, apriori=None):
        self.apriori =  None
        
    def __call__(self,rules,rule,instances,data):
        if not self.apriori:
            return False
        if not type(rule.classifier) == Orange.core.DefaultClassifier:
            return False
        ruleAcc = rule.classDistribution[rule.classifier.defaultVal]/rule.classDistribution.abs
        aprioriAcc = self.apriori[rule.classifier.defaultVal]/self.apriori.abs
        if ruleAcc>aprioriAcc:
            return False
        return True


class RuleStopping_SetRules(RuleStoppingCriteria):
    def __init__(self,validator):
        self.ruleStopping = RuleStoppingCriteria_NegativeDistribution()
        self.validator = validator

    def __call__(self,rules,rule,instances,data):        
        ru_st = self.ruleStopping(rules,rule,instances,data)
        if not ru_st:
            self.validator.rules.append(rule)
        return bool(ru_st)


class LengthValidator(RuleValidator):
    """ prune rules with more conditions than self.length. """
    def __init__(self, length=-1):
        self.length = length
        
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if self.length >= 0:
            return len(rule.filter.conditions) <= self.length
        return True    


class NoDuplicatesValidator(RuleValidator):
    def __init__(self,alpha=.05,min_coverage=0,max_rule_length=0,rules=RuleList()):
        self.rules = rules
        self.validator = RuleValidator_LRS(alpha=alpha,\
            min_coverage=min_coverage,max_rule_length=max_rule_length)
        
    def __call__(self, rule, data, weightID, targetClass, apriori):
        if rule_in_set(rule,self.rules):
            return False
        return bool(self.validator(rule,data,weightID,targetClass,apriori))
                


class RuleClassifier_BestRule(RuleClassifier):
    def __init__(self, rules, instances, weightID = 0, **argkw):
        self.rules = rules
        self.examples = instances
        self.classVar = instances.domain.classVar
        self.__dict__.update(argkw)
        self.prior = Orange.core.Distribution(instances.domain.classVar, instances)

    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue):
        retDist = Orange.core.Distribution(instance.domain.classVar)
        bestRule = None
        for r in self.rules:
            if r(instance) and (not bestRule or r.quality>bestRule.quality):
                for v_i,v in enumerate(instance.domain.classVar):
                    retDist[v_i] = r.classDistribution[v_i]
                bestRule = r
        if not bestRule:
            retDist = self.prior
        else:
            bestRule.used += 1
        sumdist = sum(retDist)
        if sumdist > 0.0:
            for c in self.examples.domain.classVar:
                retDist[c] /= sumdisc
        else:
            retDist.normalize()
        # return classifier(instance, result_type=result_type)
        if result_type == Orange.classification.Classifier.GetValue:
          return retDist.modus()
        if result_type == Orange.classification.Classifier.GetProbabilities:
          return retDist
        return (retDist.modus(),retDist)

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr    


class CovererAndRemover_MultWeights(RuleCovererAndRemover):
    """
    Covering and removing of instances using weight multiplication.
    :param mult: weighting multiplication factor
    :type mult: float
    
    """
    
    def __init__(self, mult = 0.7):
        self.mult = mult
    def __call__(self, rule, instances, weights, targetClass):
        if not weights:
            weights = Orange.core.newmetaid()
            instances.addMetaAttribute(weights,1.)
            instances.domain.addmeta(weights, Orange.data.variable.\
                Continuous("weights-"+str(weights)), True)
        newWeightsID = Orange.core.newmetaid()
        instances.addMetaAttribute(newWeightsID,1.)
        instances.domain.addmeta(newWeightsID, Orange.data.variable.\
            Continuous("weights-"+str(newWeightsID)), True)
        for instance in instances:
            if rule(instance) and instance.getclass() == rule.classifier(\
                instance,Orange.classification.Classifier.GetValue):
                instance[newWeightsID]=instance[weights]*self.mult
            else:
                instance[newWeightsID]=instance[weights]
        return (instances,newWeightsID)


class CovererAndRemover_AddWeights(RuleCovererAndRemover):
    """
    Covering and removing of instances using weight addition.
    
    """
    
    def __call__(self, rule, instances, weights, targetClass):
        if not weights:
            weights = Orange.core.newmetaid()
            instances.addMetaAttribute(weights,1.)
            instances.domain.addmeta(weights, Orange.data.variable.\
                Continuous("weights-"+str(weights)), True)
        try:
            coverage = instances.domain.getmeta("Coverage")
        except:
            coverage = Orange.data.variable.Continuous("Coverage")
            instances.domain.addmeta(Orange.core.newmetaid(),coverage, True)
            instances.addMetaAttribute(coverage,0.0)
        newWeightsID = Orange.core.newmetaid()
        instances.addMetaAttribute(newWeightsID,1.)
        instances.domain.addmeta(newWeightsID, Orange.data.variable.\
            Continuous("weights-"+str(newWeightsID)), True)
        for instance in instances:
            if rule(instance) and instance.getclass() == rule.classifier(instance,\
                    Orange.classification.Classifier.GetValue):
                try:
                    instance[coverage]+=1.0
                except:
                    instance[coverage]=1.0
                instance[newWeightsID]=1.0/(instance[coverage]+1)
            else:
                instance[newWeightsID]=instance[weights]
        return (instances,newWeightsID)


class CovererAndRemover_Prob(RuleCovererAndRemover):
    """ This class impements probabilistic covering. """
    def __init__(self, probAttribute=None, sigAttribute=None):
        self.indices = None
        self.probAttribute = probAttribute
        self.bestRule = []

    def initialize(self, instances, weightID, targetClass, apriori):
        self.bestRule = [None]*len(instances)
        self.probAttribute = Orange.core.newmetaid()
        instances.addMetaAttribute(self.probAttribute,-1.e-6)
        instances.domain.addmeta(self.probAttribute, \
            Orange.data.variable.Continuous("Probs"))
        for instance in instances:
##            if targetClass<0 or (instance.getclass() == targetClass):
            instance[self.probAttribute] = apriori[targetClass]/apriori.abs
        return instances

    def getBestRules(self, currentRules, instances, weightID):
        bestRules = RuleList()
        for r in currentRules:
            if hasattr(r.learner, "argumentRule") and not orngCN2.rule_in_set(r,bestRules):
                bestRules.append(r)
        for r_i,r in enumerate(self.bestRule):
            if r and not rule_in_set(r,bestRules) and instances[r_i].\
                getclass()==r.classifier.defaultValue:
                bestRules.append(r)
        return bestRules

    def remainingInstancesP(self, instances, targetClass):
        pSum, pAll = 0.0, 0.0
        for ex in instances:
            if ex.getclass() == targetClass:
                pSum += ex[self.probAttribute]
                pAll += 1.0
        return pSum/pAll

    def __call__(self, rule, instances, weights, targetClass):
        if targetClass<0:
            for instance_i, instance in enumerate(instances):
                if rule(instance) and rule.quality>instance[self.probAttribute]-0.01:
                    instance[self.probAttribute] = rule.quality+0.01
                    self.bestRule[instance_i]=rule
        else:
            for instance_i, instance in enumerate(instances): #rule.classifier.defaultVal == instance.getclass() and
                if rule(instance) and rule.quality>instance[self.probAttribute]:
                    instance[self.probAttribute] = rule.quality+0.001
                    self.bestRule[instance_i]=rule
##                if rule.classifier.defaultVal == instance.getclass():
##                    print instance[self.probAttribute]
        # compute factor
        return (instances,weights)


def ruleToString(rule, showDistribution = True):
    """
    Write a string presentation of rule in human readable format.
    
    :param rule: rule to pretty-print.
    :type rule: :class:`Orange.classification.rules.Rule`
    
    :param showDistribution: determines whether presentation should also
        contain the distribution of covered instances
    :type showDistribution: bool
    
    """
    def selectSign(oper):
        if oper == Orange.core.ValueFilter_continuous.Less:
            return "<"
        elif oper == Orange.core.ValueFilter_continuous.LessEqual:
            return "<="
        elif oper == Orange.core.ValueFilter_continuous.Greater:
            return ">"
        elif oper == Orange.core.ValueFilter_continuous.GreaterEqual:
            return ">="
        else: return "="

    if not rule:
        return "None"
    conds = rule.filter.conditions
    domain = rule.filter.domain
    
    ret = "IF "
    if len(conds)==0:
        ret = ret + "TRUE"

    for i,c in enumerate(conds):
        if i > 0:
            ret += " AND "
        if type(c) == Orange.core.ValueFilter_discrete:
            ret += domain[c.position].name + "=" + str([domain[c.position].\
                values[int(v)] for v in c.values])
        elif type(c) == Orange.core.ValueFilter_continuous:
            ret += domain[c.position].name + selectSign(c.oper) + str(c.ref)
    if rule.classifier and type(rule.classifier) == Orange.core.DefaultClassifier\
            and rule.classifier.defaultVal:
        ret = ret + " THEN "+domain.classVar.name+"="+\
        str(rule.classifier.defaultValue)
        if showDistribution:
            ret += str(rule.classDistribution)
    elif rule.classifier and type(rule.classifier) == Orange.core.DefaultClassifier\
            and type(domain.classVar) == Orange.core.EnumVariable:
        ret = ret + " THEN "+domain.classVar.name+"="+\
        str(rule.classDistribution.modus())
        if showDistribution:
            ret += str(rule.classDistribution)
    return ret        

def supervisedClassCheck(instances):
    if not instances.domain.classVar:
        raise Exception("Class variable is required!")
    if instances.domain.classVar.varType == Orange.core.VarTypes.Continuous:
        raise Exception("CN2 requires a discrete class!")
    

def rule_in_set(rule,rules):
    for r in rules:
        if rules_equal(rule,r):
            return True
    return False

def rules_equal(rule1,rule2):
    if not len(rule1.filter.conditions)==len(rule2.filter.conditions):
        return False
    for c1 in rule1.filter.conditions:
        found=False # find the same condition in the other rule
        for c2 in rule2.filter.conditions:
            try:
                if not c1.position == c2.position: continue # same feature?
                if not type(c1) == type(c2): continue # same type of condition
                if type(c1) == Orange.core.ValueFilter_discrete:
                    if not type(c1.values[0]) == type(c2.values[0]): continue
                    if not c1.values[0] == c2.values[0]: continue # same value?
                if type(c1) == Orange.core.ValueFilter_continuous:
                    if not c1.oper == c2.oper: continue # same operator?
                    if not c1.ref == c2.ref: continue #same threshold?
                found=True
                break
            except:
                pass
        if not found:
            return False
    return True

# Miscellaneous - utility functions
def avg(l):
    if len(l)==0:
        return 0.
    return sum(l)/len(l)

def var(l):
    if len(l)<2:
        return 0.
    av = avg(l)
    vars=[math.pow(li-av,2) for li in l]
    return sum(vars)/(len(l)-1)

def median(l):
    if len(l)==0:
        return 0.    
    l.sort()
    le = len(l)
    if le%2 == 1:
        return l[(le-1)/2]
    else:
        return (l[le/2-1]+l[le/2])/2

def perc(l,p):
    l.sort()
    return l[int(math.floor(p*len(l)))]

def createRandomDataSet(data):
    newData = Orange.data.Table(data)
    # shuffle data
    cl_num = newData.toNumeric("C")
    random.shuffle(cl_num[0][:,0])
    clData = Orange.data.Table(Orange.data.Domain([newData.domain.classVar]),cl_num[0])
    for d_i,d in enumerate(newData):
        d[newData.domain.classVar] = clData[d_i][newData.domain.classVar]
    return newData

# estimated fisher tippett parameters for a set of values given in vals list (+ deciles)
def compParameters(vals,oldMi=0.5,oldBeta=1.1):                    
    # compute percentiles
    vals.sort()
    N = len(vals)
    percs = [avg(vals[int(float(N)*i/10):int(float(N)*(i+1)/10)]) for i in range(10)]            
    if N<10:
        return oldMi, oldBeta, percs
    beta = math.sqrt(6*var(vals)/math.pow(math.pi,2))
    beta = min(2.0,max(oldBeta, beta))
    mi = max(oldMi, avg(vals) - 0.57721*beta)
    return mi, beta, percs

def computeDists(data, weight=0, targetClass=0, N=100, learner=None):
    """ Compute distributions of likelihood ratio statistics of extreme (best) rules."""
    if not learner:
        learner = createLearner()

    #########################
    ## Learner preparation ##
    #########################
    oldStopper = learner.ruleFinder.ruleStoppingValidator
    evaluator = learner.ruleFinder.evaluator
    learner.ruleFinder.evaluator = RuleEvaluator_LRS()
    learner.ruleFinder.evaluator.storeRules = True
    learner.ruleFinder.ruleStoppingValidator = RuleValidator_LRS(alpha=1.0)
    learner.ruleFinder.ruleStoppingValidator.max_rule_complexity = 0  

    # loop through N (sampling repetitions)
    maxVals = []
    for d_i in range(N):
        # create data set (remove and randomize)
        tempData = createRandomDataSet(data)
        learner.ruleFinder.evaluator.rules = RuleList()
        # Next, learn a rule
        bestRule = learner.ruleFinder(tempData,weight,targetClass,RuleList())
        maxVals.append(bestRule.quality)
    extremeDists=[compParameters(maxVals,1.0,1.0)]

    #####################
    ## Restore learner ##
    #####################
    learner.ruleFinder.evaluator = evaluator
    learner.ruleFinder.ruleStoppingValidator = oldStopper
    return extremeDists

def createEVDistList(evdList):
    l = Orange.core.EVDistList()
    for el in evdList:
        l.append(Orange.core.EVDist(mu=el[0],beta=el[1],percentiles=el[2]))
    return l

def add_sub_rules(rules, instances, weight, learner, dists):
    apriori = Orange.core.Distribution(instances.domain.classVar,instances,weight)
    newRules = RuleList()
    for r in rules:
        newRules.append(r)

    # loop through rules
    for r in rules:
        tmpList = RuleList()
        tmpRle = r.clone()
        tmpRle.filter.conditions = []
        tmpRle.parentRule = None
        tmpRle.filterAndStore(instances,weight,r.classifier.defaultVal)
        tmpList.append(tmpRle)
        while tmpList and len(tmpList[0].filter.conditions) <= len(r.filter.conditions):
            tmpList2 = RuleList()
            for tmpRule in tmpList:
                # evaluate tmpRule
                oldREP = learner.ruleFinder.evaluator.returnExpectedProb
                learner.ruleFinder.evaluator.returnExpectedProb = False
                learner.ruleFinder.evaluator.evDistGetter.dists = createEVDistList(\
                        dists[int(r.classifier.defaultVal)])
                tmpRule.quality = learner.ruleFinder.evaluator(tmpRule,
                        instances,weight,r.classifier.defaultVal,apriori)
                learner.ruleFinder.evaluator.returnExpectedProb = oldREP
                # if rule not in rules already, add it to the list
                if not True in [rules_equal(ri,tmpRule) for ri in newRules] and\
                        len(tmpRule.filter.conditions)>0 and tmpRule.quality >\
                            apriori[r.classifier.defaultVal]/apriori.abs:
                    newRules.append(tmpRule)
                # create new tmpRules, set parent Rule, append them to tmpList2
                if not True in [rules_equal(ri,tmpRule) for ri in newRules]:
                    for c in r.filter.conditions:
                        tmpRule2 = tmpRule.clone()
                        tmpRule2.parentRule = tmpRule
                        tmpRule2.filter.conditions.append(c)
                        tmpRule2.filterAndStore(instances,weight,r.classifier.defaultVal)
                        if tmpRule2.classDistribution.abs < tmpRule.classDistribution.abs:
                            tmpList2.append(tmpRule2)
            tmpList = tmpList2
    for cl in instances.domain.classVar:
        tmpRle = Rule()
        tmpRle.filter = Orange.core.Filter_values(domain = instances.domain)
        tmpRle.parentRule = None
        tmpRle.filterAndStore(instances,weight,int(cl))
        tmpRle.quality = tmpRle.classDistribution[int(cl)]/tmpRle.classDistribution.abs
        newRules.append(tmpRle)
    return newRules

