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



# Main ABCN2 class
class ABCN2(Orange.core.RuleLearner):
    """COPIED&PASTED FROM orngABCN2 -- REFACTOR AND DOCUMENT ASAP!
    This is implementation of ABCN2 + EVC as evaluation + LRC classification.
    """
    
    def __init__(self, argumentID=0, width=5, m=2, opt_reduction=2, nsampling=100, max_rule_complexity=5,
                 rule_sig=1.0, att_sig=1.0, postpruning=None, min_quality=0., min_coverage=1, min_improved=1, min_improved_perc=0.0,
                 learn_for_class = None, learn_one_rule = False, evd=None, evd_arguments=None, prune_arguments=False, analyse_argument=-1,
                 alternative_learner = None, min_cl_sig = 0.5, min_beta = 0.0, set_prefix_rules = False, add_sub_rules = False,
                 **kwds):
        """
        Parameters:
            General rule learning:
                width               ... beam width (default 5)
                learn_for_class     ... learner rules for one class? otherwise None
                learn_one_rule      ... learn one rule only ?
                analyse_argument    ... learner only analyses argument with this index; if set to -1, then it learns normally
                
            Evaluator related:
                m                   ... m-estimate to be corrected with EVC (default 2)
                opt_reduction       ... types of EVC correction; 0=no correction, 1=pessimistic, 2=normal (default 2)
                nsampling           ... number of samples in estimating extreme value distribution (for EVC) (default 100)
                evd                 ... pre given extreme value distributions
                evd_arguments       ... pre given extreme value distributions for arguments

            Rule Validation:
                rule_sig            ... minimal rule significance (default 1.0)
                att_sig             ... minimal attribute significance in rule (default 1.0)
                max_rule_complexity ... maximum number of conditions in rule (default 5)
                min_coverage        ... minimal number of covered examples (default 5)

            Probabilistic covering:
                min_improved        ... minimal number of examples improved in probabilistic covering (default 1)
                min_improved_perc   ... minimal percentage of covered examples that need to be improved (default 0.0)

            Classifier (LCR) related:
                add_sub_rules       ... add sub rules ? (default False)
                min_cl_sig          ... minimal significance of beta in classifier (default 0.5)
                min_beta            ... minimal beta value (default 0.0)
                set_prefix_rules    ... should ordered prefix rules be added? (default False)
                alternative_learner ... use rule-learner as a correction method for other machine learning methods. (default None)

        """

        
        # argument ID which is passed to abcn2 learner
        self.argumentID = argumentID
        # learn for specific class only?        
        self.learn_for_class = learn_for_class
        # only analysing a specific argument or learning all at once
        self.analyse_argument = analyse_argument
        # should we learn only one rule?
        self.learn_one_rule = learn_one_rule
        self.postpruning = postpruning
        # rule finder
        self.ruleFinder = Orange.core.RuleBeamFinder()
        self.ruleFilter = Orange.core.RuleBeamFilter_Width(width=width)
        self.ruleFilter_arguments = ABBeamFilter(width=width)
        if max_rule_complexity - 1 < 0:
            max_rule_complexity = 10
        self.ruleFinder.ruleStoppingValidator = Orange.core.RuleValidator_LRS(alpha = 1.0, min_quality = 0., max_rule_complexity = max_rule_complexity - 1, min_coverage=min_coverage)
        self.refiner = Orange.core.RuleBeamRefiner_Selector()
        self.refiner_arguments = SelectorAdder(discretizer = Orange.core.EntropyDiscretization(forceAttribute = 1,
                                                                                           maxNumberOfIntervals = 2))
        self.prune_arguments = prune_arguments
        # evc evaluator
        evdGet = Orange.core.EVDistGetter_Standard()
        self.ruleFinder.evaluator = Orange.core.RuleEvaluator_mEVC(m=m, evDistGetter = evdGet, min_improved = min_improved, min_improved_perc = min_improved_perc)
        self.ruleFinder.evaluator.returnExpectedProb = True
        self.ruleFinder.evaluator.optimismReduction = opt_reduction
        self.ruleFinder.evaluator.ruleAlpha = rule_sig
        self.ruleFinder.evaluator.attributeAlpha = att_sig
        self.ruleFinder.evaluator.validator = Orange.core.RuleValidator_LRS(alpha = 1.0, min_quality = min_quality, min_coverage=min_coverage, max_rule_complexity = max_rule_complexity - 1)

        # learn stopping criteria
        self.ruleStopping = None
        self.dataStopping = Orange.core.RuleDataStoppingCriteria_NoPositives()
        # evd fitting
        self.evd_creator = EVDFitter(self,n=nsampling)
        self.evd = evd
        self.evd_arguments = evd_arguments
        # classifier
        self.add_sub_rules = add_sub_rules
        self.classifier = PILAR(alternative_learner = alternative_learner, min_cl_sig = min_cl_sig, min_beta = min_beta, set_prefix_rules = set_prefix_rules)
        # arbitrary parameters
        self.__dict__.update(kwds)


    def __call__(self, examples, weightID=0):
        # initialize progress bar
        progress=getattr(self,"progressCallback",None)
        if progress:
            progress.start = 0.0
            progress.end = 0.0
            distrib = Orange.core.Distribution(examples.domain.classVar, examples, weightID)
            distrib.normalize()
        
        # we begin with an empty set of rules
        all_rules = Orange.core.RuleList()

        # th en, iterate through all classes and learn rule for each class separately
        for cl_i,cl in enumerate(examples.domain.classVar):
            if progress:
                step = distrib[cl] / 2.
                progress.start = progress.end
                progress.end += step
                
            if self.learn_for_class and not self.learn_for_class in [cl,cl_i]:
                continue

            # rules for this class only
            rules, arg_rules = Orange.core.RuleList(), Orange.core.RuleList()

            # create dichotomous class
            dich_data = self.create_dich_class(examples, cl)

            # preparation of the learner (covering, evd, etc.)
            self.prepare_settings(dich_data, weightID, cl_i, progress)

            # learn argumented rules first ...
            self.turn_ABML_mode(dich_data, weightID, cl_i)
            # first specialize all unspecialized arguments
            # dich_data = self.specialise_arguments(dich_data, weightID)
            # comment: specialisation of arguments is within learning of an argumented rule;
            #          this is now different from the published algorithm
            if progress:
                progress.start = progress.end
                progress.end += step
            
            aes = self.get_argumented_examples(dich_data)
            aes = self.sort_arguments(aes, dich_data)
            while aes:
                if self.analyse_argument > -1 and not dich_data[self.analyse_argument] == aes[0]:
                    aes = aes[1:]
                    continue
                ae = aes[0]
                rule = self.learn_argumented_rule(ae, dich_data, weightID) # target class is always first class (0)
                if not progress:
                    print "learned rule", Orange.classification.rules.ruleToString(rule)
                if rule:
                    arg_rules.append(rule)
                    aes = filter(lambda x: not rule(x), aes)
                else:
                    aes = aes[1:]
            if not progress:
                print " arguments finished ... "                    
                   
            # remove all examples covered by rules
##            for rule in rules:
##                dich_data = self.remove_covered_examples(rule, dich_data, weightID)
##            if progress:
##                progress(self.remaining_probability(dich_data),None)

            # learn normal rules on remaining examples
            if self.analyse_argument == -1:
                self.turn_normal_mode(dich_data, weightID, cl_i)
                while dich_data:
                    # learn a rule
                    rule = self.learn_normal_rule(dich_data, weightID, self.apriori)
                    if not rule:
                        break
                    if not progress:
                        print "rule learned: ", Orange.classification.rules.ruleToString(rule), rule.quality
                    dich_data = self.remove_covered_examples(rule, dich_data, weightID)
                    if progress:
                        progress(self.remaining_probability(dich_data),None)
                    rules.append(rule)
                    if self.learn_one_rule:
                        break

            for r in arg_rules:
                dich_data = self.remove_covered_examples(r, dich_data, weightID)
                rules.append(r)

            # prune unnecessary rules
            rules = self.prune_unnecessary_rules(rules, dich_data, weightID)

            if self.add_sub_rules:
                rules = self.add_sub_rules_call(rules, dich_data, weightID)

            # restore domain and class in rules, add them to all_rules
            for r in rules:
                all_rules.append(self.change_domain(r, cl, examples, weightID))

            if progress:
                progress(1.0,None)
        # create a classifier from all rules        
        return self.create_classifier(all_rules, examples, weightID)

    def learn_argumented_rule(self, ae, examples, weightID):
        # prepare roots of rules from arguments
        positive_args = self.init_pos_args(ae, examples, weightID)
        if not positive_args: # something wrong
            raise "There is a problem with argumented example %s"%str(ae)
            return None
        negative_args = self.init_neg_args(ae, examples, weightID)

        # set negative arguments in refiner
        self.ruleFinder.refiner.notAllowedSelectors = negative_args
        self.ruleFinder.refiner.example = ae
        # set arguments to filter
        self.ruleFinder.ruleFilter.setArguments(examples.domain,positive_args)

        # learn a rule
        self.ruleFinder.evaluator.bestRule = None
        self.ruleFinder.evaluator.returnBestFuture = True
        self.ruleFinder(examples,weightID,0,positive_args)
##        self.ruleFinder.evaluator.bestRule.quality = 0.8
        
        # return best rule
        return self.ruleFinder.evaluator.bestRule
        
    def prepare_settings(self, examples, weightID, cl_i, progress):
        # apriori distribution
        self.apriori = Orange.core.Distribution(examples.domain.classVar,examples,weightID)
        
        # prepare covering mechanism
        self.coverAndRemove = CovererAndRemover_Prob(examples, weightID, 0, self.apriori)
        self.ruleFinder.evaluator.probVar = examples.domain.getmeta(self.coverAndRemove.probAttribute)

        # compute extreme distributions
        # TODO: why evd and evd_this????
        if self.ruleFinder.evaluator.optimismReduction > 0 and not self.evd:
            self.evd_this = self.evd_creator.computeEVD(examples, weightID, target_class=0, progress = progress)
        if self.evd:
            self.evd_this = self.evd[cl_i]

    def turn_ABML_mode(self, examples, weightID, cl_i):
        # evaluator
        if self.ruleFinder.evaluator.optimismReduction > 0 and self.argumentID:
            if self.evd_arguments:
                self.ruleFinder.evaluator.evDistGetter.dists = self.evd_arguments[cl_i]
            else:
                self.ruleFinder.evaluator.evDistGetter.dists = self.evd_this # self.evd_creator.computeEVD_example(examples, weightID, target_class=0)
        # rule refiner
        self.ruleFinder.refiner = self.refiner_arguments
        self.ruleFinder.refiner.argumentID = self.argumentID
        self.ruleFinder.ruleFilter = self.ruleFilter_arguments

    def create_dich_class(self, examples, cl):
        """ create dichotomous class. """
        (newDomain, targetVal) = createDichotomousClass(examples.domain, examples.domain.classVar, str(cl), negate=0)
        newDomainmetas = newDomain.getmetas()
        newDomain.addmeta(Orange.core.newmetaid(), examples.domain.classVar) # old class as meta
        dichData = examples.select(newDomain)
        if self.argumentID:
            for d in dichData: # remove arguments given to other classes
                if not d.getclass() == targetVal:
                    d[self.argumentID] = "?"
        return dichData

    def get_argumented_examples(self, examples):
        if not self.argumentID:
            return None
        
        # get argumentated examples
        return ArgumentFilter_hasSpecial()(examples, self.argumentID, targetClass = 0)

    def sort_arguments(self, arg_examples, examples):
        if not self.argumentID:
            return None
        evaluateAndSortArguments(examples, self.argumentID)
        if len(arg_examples)>0:
            # sort examples by their arguments quality (using first argument as it has already been sorted)
            sorted = arg_examples.native()
            sorted.sort(lambda x,y: -cmp(x[self.argumentID].value.positiveArguments[0].quality,
                                         y[self.argumentID].value.positiveArguments[0].quality))
            return Orange.core.ExampleTable(examples.domain, sorted)
        else:
            return None

    def turn_normal_mode(self, examples, weightID, cl_i):
        # evaluator
        if self.ruleFinder.evaluator.optimismReduction > 0:
            if self.evd:
                self.ruleFinder.evaluator.evDistGetter.dists = self.evd[cl_i]
            else:
                self.ruleFinder.evaluator.evDistGetter.dists = self.evd_this # self.evd_creator.computeEVD(examples, weightID, target_class=0)
        # rule refiner
        self.ruleFinder.refiner = self.refiner
        self.ruleFinder.ruleFilter = self.ruleFilter
        
    def learn_normal_rule(self, examples, weightID, apriori):
        if hasattr(self.ruleFinder.evaluator, "bestRule"):
            self.ruleFinder.evaluator.bestRule = None
        rule = self.ruleFinder(examples,weightID,0,Orange.core.RuleList())
        if hasattr(self.ruleFinder.evaluator, "bestRule") and self.ruleFinder.evaluator.returnExpectedProb:
            rule = self.ruleFinder.evaluator.bestRule
            self.ruleFinder.evaluator.bestRule = None
        if self.postpruning:
            rule = self.postpruning(rule,examples,weightID,0, aprior)
        return rule

    def remove_covered_examples(self, rule, examples, weightID):
        nexamples, nweight = self.coverAndRemove(rule,examples,weightID,0)
        return nexamples


    def prune_unnecessary_rules(self, rules, examples, weightID):
        return self.coverAndRemove.getBestRules(rules,examples,weightID)

    def change_domain(self, rule, cl, examples, weightID):
        rule.examples = rule.examples.select(examples.domain)
        rule.classDistribution = Orange.core.Distribution(rule.examples.domain.classVar,rule.examples,weightID) # adapt distribution
        rule.classifier = Orange.core.DefaultClassifier(cl) # adapt classifier
        rule.filter = Orange.core.Filter_values(domain = examples.domain,
                                        conditions = rule.filter.conditions)
        if hasattr(rule, "learner") and hasattr(rule.learner, "arg_example"):
            rule.learner.arg_example = Orange.core.Example(examples.domain, rule.learner.arg_example)
        return rule

    def create_classifier(self, rules, examples, weightID):
        return self.classifier(rules, examples, weightID)

    def add_sub_rules_call(self, rules, examples, weightID):
        apriori = Orange.core.Distribution(examples.domain.classVar,examples,weightID)
        newRules = Orange.core.RuleList()
        for r in rules:
            newRules.append(r)

        # loop through rules
        for r in rules:
            tmpList = Orange.core.RuleList()
            tmpRle = r.clone()
            tmpRle.filter.conditions = r.filter.conditions[:r.requiredConditions] # do not split argument
            tmpRle.parentRule = None
            tmpRle.filterAndStore(examples,weightID,r.classifier.defaultVal)
            tmpRle.complexity = 0
            tmpList.append(tmpRle)
            while tmpList and len(tmpList[0].filter.conditions) <= len(r.filter.conditions):
                tmpList2 = Orange.core.RuleList()
                for tmpRule in tmpList:
                    # evaluate tmpRule
                    oldREP = self.ruleFinder.evaluator.returnExpectedProb
                    self.ruleFinder.evaluator.returnExpectedProb = False
                    tmpRule.quality = self.ruleFinder.evaluator(tmpRule,examples,weightID,r.classifier.defaultVal,apriori)
                    self.ruleFinder.evaluator.returnExpectedProb = oldREP
                    # if rule not in rules already, add it to the list
                    if not True in [Orange.classification.rules.rules_equal(ri,tmpRule) for ri in newRules] and len(tmpRule.filter.conditions)>0 and tmpRule.quality > apriori[r.classifier.defaultVal]/apriori.abs:
                        newRules.append(tmpRule)
                    # create new tmpRules, set parent Rule, append them to tmpList2
                    if not True in [Orange.classification.rules.rules_equal(ri,tmpRule) for ri in newRules]:
                        for c in r.filter.conditions:
                            tmpRule2 = tmpRule.clone()
                            tmpRule2.parentRule = tmpRule
                            tmpRule2.filter.conditions.append(c)
                            tmpRule2.filterAndStore(examples,weightID,r.classifier.defaultVal)
                            tmpRule2.complexity += 1
                            if tmpRule2.classDistribution.abs < tmpRule.classDistribution.abs:
                                tmpList2.append(tmpRule2)
                tmpList = tmpList2
        return newRules


    def init_pos_args(self, ae, examples, weightID):
        pos_args = Orange.core.RuleList()
        # prepare arguments
        for p in ae[self.argumentID].value.positiveArguments:
            new_arg = Orange.core.Rule(filter=ArgFilter(argumentID = self.argumentID,
                                                   filter = self.newFilter_values(p.filter)),
                                                   complexity = 0)
            new_arg.valuesFilter = new_arg.filter.filter
            pos_args.append(new_arg)


        if hasattr(self.ruleFinder.evaluator, "returnExpectedProb"):
            old_exp = self.ruleFinder.evaluator.returnExpectedProb
            self.ruleFinder.evaluator.returnExpectedProb = False
            
        # argument pruning (all or just unfinished arguments)
        # if pruning is chosen, then prune arguments if possible
        for p in pos_args:
            p.filterAndStore(examples, weightID, 0)
            # pruning on: we check on all conditions and take only best
            if self.prune_arguments:
                allowed_conditions = [c for c in p.filter.conditions]
                pruned_conditions = self.prune_arg_conditions(ae, allowed_conditions, examples, weightID)
                p.filter.conditions = pruned_conditions
            else: # prune only unspecified conditions
                spec_conditions = [c for c in p.filter.conditions if not c.unspecialized_condition]
                unspec_conditions = [c for c in p.filter.conditions if c.unspecialized_condition]
                # let rule cover now all examples filtered by specified conditions
                p.filter.conditions = spec_conditions
                p.filterAndStore(examples, weightID, 0)
                pruned_conditions = self.prune_arg_conditions(ae, unspec_conditions, p.examples, p.weightID)
                p.filter.conditions.extend(pruned_conditions)
                p.filter.filter.conditions.extend(pruned_conditions)
                # if argument does not contain all unspecialized reasons, add those reasons with minimum values
                at_oper_pairs = [(c.position, c.oper) for c in p.filter.conditions if type(c) == Orange.core.ValueFilter_continuous]
                for u in unspec_conditions:
                    if not (u.position, u.oper) in at_oper_pairs:
                        # find minimum value
                        u.ref = min([float(e[u.position])-10. for e in p.examples])
                        p.filter.conditions.append(u)
                        p.filter.filter.conditions.append(u)
                

        # set parameters to arguments
        for p_i,p in enumerate(pos_args):
            p.filterAndStore(examples,weightID,0)
            p.filter.domain = examples.domain
            if not p.learner:
                p.learner = DefaultLearner(defaultValue=ae.getclass())
            p.classifier = p.learner(p.examples, p.weightID)
            p.baseDist = p.classDistribution
            p.requiredConditions = len(p.filter.conditions)
            p.learner.setattr("arg_length", len(p.filter.conditions))
            p.learner.setattr("arg_example", ae)
            p.complexity = len(p.filter.conditions)
            
        if hasattr(self.ruleFinder.evaluator, "returnExpectedProb"):
            self.ruleFinder.evaluator.returnExpectedProb = old_exp

        return pos_args

    def newFilter_values(self, filter):
        newFilter = Orange.core.Filter_values()
        newFilter.conditions = filter.conditions[:]
        newFilter.domain = filter.domain
        newFilter.negate = filter.negate
        newFilter.conjunction = filter.conjunction
        return newFilter

    def init_neg_args(self, ae, examples, weightID):
        return ae[self.argumentID].value.negativeArguments

    def remaining_probability(self, examples):
        return self.coverAndRemove.covered_percentage(examples)

    def prune_arg_conditions(self, crit_example, allowed_conditions, examples, weightID):
        if not allowed_conditions:
            return []
        cn2_learner = Orange.classification.rules.CN2UnorderedLearner()
        cn2_learner.ruleFinder = Orange.core.RuleBeamFinder()
        cn2_learner.ruleFinder.refiner = SelectorArgConditions(crit_example, allowed_conditions)
        cn2_learner.ruleFinder.evaluator = Orange.classification.rules.MEstimate(self.ruleFinder.evaluator.m)
        rule = cn2_learner.ruleFinder(examples,weightID,0,Orange.core.RuleList())
        return rule.filter.conditions


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









################################################################################
################################################################################
##  This has been copyed&pasted from orngABCN2.py and not yet appropriately   ##
##  refactored and documented.                                                ##
################################################################################
################################################################################


""" This module implements argument based rule learning.
The main learner class is ABCN2. The first few classes are some variants of ABCN2 with reasonable settings.  """


import operator
import random
import numpy
import math

from orngABML import *

# Default learner - returns     #
# default classifier with pre-  #
# defined output  class         #
class DefaultLearner(Orange.core.Learner):
    def __init__(self,defaultValue = None):
        self.defaultValue = defaultValue
    def __call__(self,examples,weightID=0):
        return Orange.core.DefaultClassifier(self.defaultValue,defaultDistribution = Orange.core.Distribution(examples.domain.classVar,examples,weightID))

class ABCN2Ordered(ABCN2):
    """ Rules learned by ABCN2 are ordered and used as a decision list. """
    def __init__(self, argumentID=0, **kwds):
        ABCN2.__init__(self, argumentID=argumentID, **kwds)
        self.classifier.set_prefix_rules = True
        self.classifier.optimize_betas = False

class ABCN2M(ABCN2):
    """ Argument based rule learning with m-estimate as evaluation function. """
    def __init__(self, argumentID=0, **kwds):
        ABCN2.__init__(self, argumentID=argumentID, **kwds)
        self.opt_reduction = 0
    

# *********************** #
# Argument based covering #
# *********************** #

class ABBeamFilter(Orange.core.RuleBeamFilter):
    """ ABBeamFilter: Filters beam;
        - leaves first N rules (by quality)
        - leaves first N rules that have only of arguments in condition part 
    """
    def __init__(self,width=5):
        self.width=width
        self.pArgs=None

    def __call__(self,rulesStar,examples,weightID):
        newStar=Orange.core.RuleList()
        rulesStar.sort(lambda x,y: -cmp(x.quality,y.quality))
        argsNum=0
        for r_i,r in enumerate(rulesStar):
            if r_i<self.width: # either is one of best "width" rules
                newStar.append(r)
            elif self.onlyPositives(r):
                if argsNum<self.width:
                    newStar.append(r)
                    argsNum+=1
        return newStar                

    def setArguments(self,domain,positiveArguments):
        self.pArgs = positiveArguments
        self.domain = domain
        self.argTab = [0]*len(self.domain.attributes)
        for arg in self.pArgs:
            for cond in arg.filter.conditions:
                self.argTab[cond.position]=1
        
    def onlyPositives(self,rule):
        if not self.pArgs:
            return False

        ruleTab=[0]*len(self.domain.attributes)
        for cond in rule.filter.conditions:
            ruleTab[cond.position]=1
        return map(operator.or_,ruleTab,self.argTab)==self.argTab


class ruleCoversArguments:
    """ Class determines if rule covers one out of a set of arguments. """
    def __init__(self, arguments):
        self.arguments = arguments
        self.indices = []
        for a in self.arguments:
            indNA = getattr(a.filter,"indices",None)
            if not indNA:
                a.filter.setattr("indices", ruleCoversArguments.filterIndices(a.filter))
            self.indices.append(a.filter.indices)

    def __call__(self, rule):
        if not self.indices:
            return False
        if not getattr(rule.filter,"indices",None):
            rule.filter.indices = ruleCoversArguments.filterIndices(rule.filter)
        for index in self.indices:
            if map(operator.or_,rule.filter.indices,index) == rule.filter.indices:
                return True
        return False

    def filterIndices(filter):
        if not filter.domain:
            return []
        ind = [0]*len(filter.domain.attributes)
        for c in filter.conditions:
            ind[c.position]=operator.or_(ind[c.position],
                                         ruleCoversArguments.conditionIndex(c))
        return ind
    filterIndices = staticmethod(filterIndices)

    def conditionIndex(c):
        if type(c) == Orange.core.ValueFilter_continuous:
            if (c.oper == Orange.core.ValueFilter_continuous.GreaterEqual or
                c.oper == Orange.core.ValueFilter_continuous.Greater):
                return 5# 0101
            elif (c.oper == Orange.core.ValueFilter_continuous.LessEqual or
                  c.oper == Orange.core.ValueFilter_continuous.Less):
                return 3 # 0011
            else:
                return c.oper
        else:
            return 1 # 0001
    conditionIndex = staticmethod(conditionIndex)        

    def oneSelectorToCover(ruleIndices, argIndices):
        at, type = -1, 0
        for r_i, ind in enumerate(ruleIndices):
            if not argIndices[r_i]:
                continue
            if at>-1 and not ind == argIndices[r_i]: # need two changes
                return (-1,0)
            if not ind == argIndices[r_i]:
                if argIndices[r_i] in [1,3,5]:
                    at,type=r_i,argIndices[r_i]
                if argIndices[r_i]==6:
                    if ind==3:
                        at,type=r_i,5
                    if ind==5:
                        at,type=r_i,3
        return at,type
    oneSelectorToCover = staticmethod(oneSelectorToCover)                 

class SelectorAdder(Orange.core.RuleBeamRefiner):
    """ Selector adder, this function is a refiner function:
       - refined rules are not consistent with any of negative arguments. """
    def __init__(self, example=None, notAllowedSelectors=[], argumentID = None,
                 discretizer = Orange.core.EntropyDiscretization(forceAttribute=True)):
        # required values - needed values of attributes
        self.example = example
        self.argumentID = argumentID
        self.notAllowedSelectors = notAllowedSelectors
        self.discretizer = discretizer
        
    def __call__(self, oldRule, data, weightID, targetClass=-1):
        inNotAllowedSelectors = ruleCoversArguments(self.notAllowedSelectors)
        newRules = Orange.core.RuleList()

        # get positive indices (selectors already in the rule)
        indices = getattr(oldRule.filter,"indices",None)
        if not indices:
            indices = ruleCoversArguments.filterIndices(oldRule.filter)
            oldRule.filter.setattr("indices",indices)

        # get negative indices (selectors that should not be in the rule)
        negativeIndices = [0]*len(data.domain.attributes)
        for nA in self.notAllowedSelectors:
            #print indices, nA.filter.indices
            at_i,type_na = ruleCoversArguments.oneSelectorToCover(indices, nA.filter.indices)
            if at_i>-1:
                negativeIndices[at_i] = operator.or_(negativeIndices[at_i],type_na)

        #iterate through indices = attributes 
        for i,ind in enumerate(indices):
            if not self.example[i] or self.example[i].isSpecial():
                continue
            if ind == 1: 
                continue
            if data.domain[i].varType == Orange.core.VarTypes.Discrete and not negativeIndices[i]==1: # DISCRETE attribute
                if self.example:
                    values = [self.example[i]]
                else:
                    values = data.domain[i].values
                for v in values:
                    tempRule = oldRule.clone()
                    tempRule.filter.conditions.append(Orange.core.ValueFilter_discrete(position = i,
                                                                                  values = [Orange.core.Value(data.domain[i],v)],
                                                                                  acceptSpecial=0))
                    tempRule.complexity += 1
                    tempRule.filter.indices[i] = 1 # 1 stands for discrete attribute (see ruleCoversArguments.conditionIndex)
                    tempRule.filterAndStore(oldRule.examples, oldRule.weightID, targetClass)
                    if len(tempRule.examples)<len(oldRule.examples):
                        newRules.append(tempRule)
            elif data.domain[i].varType == Orange.core.VarTypes.Continuous and not negativeIndices[i]==7: # CONTINUOUS attribute
                try:
                    at = data.domain[i]
                    at_d = self.discretizer(at,oldRule.examples)
                except:
                    continue # discretization failed !
                # If discretization makes sense? then:
                if len(at_d.values)>1:
                    for p in at_d.getValueFrom.transformer.points:
                        #LESS
                        if not negativeIndices[i]==3:
                            tempRule = self.getTempRule(oldRule,i,Orange.core.ValueFilter_continuous.LessEqual,p,targetClass,3)
                            if len(tempRule.examples)<len(oldRule.examples) and self.example[i]<=p:# and not inNotAllowedSelectors(tempRule):
                                newRules.append(tempRule)
                        #GREATER
                        if not negativeIndices[i]==5:
                            tempRule = self.getTempRule(oldRule,i,Orange.core.ValueFilter_continuous.Greater,p,targetClass,5)
                            if len(tempRule.examples)<len(oldRule.examples) and self.example[i]>p:# and not inNotAllowedSelectors(tempRule):
                                newRules.append(tempRule)
        for r in newRules:
            r.parentRule = oldRule
            r.valuesFilter = r.filter.filter
        return newRules

    def getTempRule(self,oldRule,pos,oper,ref,targetClass,atIndex):
        tempRule = oldRule.clone()

        tempRule.filter.conditions.append(Orange.core.ValueFilter_continuous(position=pos,
                                                                        oper=oper,
                                                                        ref=ref,
                                                                        acceptSpecial=0))
        tempRule.complexity += 1
        tempRule.filter.indices[pos] = operator.or_(tempRule.filter.indices[pos],atIndex) # from ruleCoversArguments.conditionIndex
        tempRule.filterAndStore(oldRule.examples,tempRule.weightID,targetClass)
        return tempRule

    def setCondition(self, oldRule, targetClass, ci, condition):
        tempRule = oldRule.clone()
        tempRule.filter.conditions[ci] = condition
        tempRule.filter.conditions[ci].setattr("specialized",1)
        tempRule.filterAndStore(oldRule.examples,oldRule.weightID,targetClass)
        return tempRule


# This filter is the ugliest code ever! Problem is with Orange, I had some problems with inheriting deepCopy
# I should take another look at it.
class ArgFilter(Orange.core.Filter):
    """ This class implements AB-covering principle. """
    def __init__(self, argumentID=None, filter = Orange.core.Filter_values()):
        self.filter = filter
        self.indices = getattr(filter,"indices",[])
        if not self.indices and len(filter.conditions)>0:
            self.indices = ruleCoversArguments.filterIndices(filter)
        self.argumentID = argumentID
        self.debug = 0
        self.domain = self.filter.domain
        self.conditions = filter.conditions
        
    def condIn(self,cond): # is condition in the filter?
        condInd = ruleCoversArguments.conditionIndex(cond)
        if operator.or_(condInd,self.indices[cond.position]) == self.indices[cond.position]:
            return True
        return False
    
    def __call__(self,example):
##        print "in", self.filter(example), self.filter.conditions[0](example)
##        print self.filter.conditions[1].values
        if self.filter(example):
            try:
                if example[self.argumentID].value and len(example[self.argumentID].value.positiveArguments)>0: # example has positive arguments
                    # conditions should cover at least one of the positive arguments
                    oneArgCovered = False
                    for pA in example[self.argumentID].value.positiveArguments:
                        argCovered = [self.condIn(c) for c in pA.filter.conditions]
                        oneArgCovered = oneArgCovered or len(argCovered) == sum(argCovered) #argCovered
                        if oneArgCovered:
                            break
                    if not oneArgCovered:
                        return False
                if example[self.argumentID].value and len(example[self.argumentID].value.negativeArguments)>0: # example has negative arguments
                    # condition should not cover neither of negative arguments
                    for pN in example[self.argumentID].value.negativeArguments:
                        argCovered = [self.condIn(c) for c in pN.filter.conditions]
                        if len(argCovered)==sum(argCovered):
                            return False
            except:
                return True
            return True
        else:
            return False

    def __setattr__(self,name,obj):
        self.__dict__[name]=obj
        self.filter.setattr(name,obj)

    def deepCopy(self):
        newFilter = ArgFilter(argumentID=self.argumentID)
        newFilter.filter = Orange.core.Filter_values() #self.filter.deepCopy()
        newFilter.filter.conditions = self.filter.conditions[:]
        newFilter.domain = self.filter.domain
        newFilter.negate = self.filter.negate
        newFilter.conjunction = self.filter.conjunction
        newFilter.domain = self.filter.domain
        newFilter.conditions = newFilter.filter.conditions
        newFilter.indices = self.indices[:]
        if getattr(self,"candidateValues",None):
            newFilter.candidateValues = self.candidateValues[:]
        return newFilter


class SelectorArgConditions(Orange.core.RuleBeamRefiner):
    """ Selector adder, this function is a refiner function:
       - refined rules are not consistent with any of negative arguments. """
    def __init__(self, example, allowed_selectors):
        # required values - needed values of attributes
        self.example = example
        self.allowed_selectors = allowed_selectors

    def __call__(self, oldRule, data, weightID, targetClass=-1):
        if len(oldRule.filter.conditions) >= len(self.allowed_selectors):
            return Orange.core.RuleList()
        newRules = Orange.core.RuleList()
        for c in self.allowed_selectors:
            # normal condition
            if not c.unspecialized_condition:
                tempRule = oldRule.clone()
                tempRule.filter.conditions.append(c)
                tempRule.filterAndStore(oldRule.examples, oldRule.weightID, targetClass)
                if len(tempRule.examples)<len(oldRule.examples):
                    newRules.append(tempRule)
            # unspecified condition
            else:
                # find all possible example values
                vals = {}
                for e in oldRule.examples:
                    if not e[c.position].isSpecial():
                        vals[str(e[c.position])] = 1
                values = vals.keys()
                # for each value make a condition
                for v in values:
                    tempRule = oldRule.clone()
                    tempRule.filter.conditions.append(Orange.core.ValueFilter_continuous(position=c.position,
                                                                                    oper=c.oper,
                                                                                    ref=float(v),
                                                                                    acceptSpecial=0))
                    if tempRule(self.example):
                        tempRule.filterAndStore(oldRule.examples, oldRule.weightID, targetClass)
                        if len(tempRule.examples)<len(oldRule.examples):
                            newRules.append(tempRule)
##        print " NEW RULES "
##        for r in newRules:
##            print Orange.classification.rules.ruleToString(r)
        for r in newRules:
            r.parentRule = oldRule
##            print Orange.classification.rules.ruleToString(r)
        return newRules


# ********************** #
# Probabilistic covering #
# ********************** #

class CovererAndRemover_Prob(Orange.core.RuleCovererAndRemover):
    """ This class impements probabilistic covering. """

    def __init__(self, examples, weightID, targetClass, apriori):
        self.bestRule = [None]*len(examples)
        self.probAttribute = Orange.core.newmetaid()
        self.aprioriProb = apriori[targetClass]/apriori.abs
        examples.addMetaAttribute(self.probAttribute, self.aprioriProb)
        examples.domain.addmeta(self.probAttribute, Orange.core.FloatVariable("Probs"))

    def getBestRules(self, currentRules, examples, weightID):
        bestRules = Orange.core.RuleList()
##        for r in currentRules:
##            if hasattr(r.learner, "argumentRule") and not Orange.classification.rules.rule_in_set(r,bestRules):
##                bestRules.append(r)
        for r_i,r in enumerate(self.bestRule):
            if r and not Orange.classification.rules.rule_in_set(r,bestRules) and int(examples[r_i].getclass())==int(r.classifier.defaultValue):
                bestRules.append(r)
        return bestRules

    def __call__(self, rule, examples, weights, targetClass):
        if hasattr(rule, "learner") and hasattr(rule.learner, "arg_example"):
            example = rule.learner.arg_example
        else:
            example = None
        for ei, e in enumerate(examples):
##            if e == example:
##                e[self.probAttribute] = 1.0
##                self.bestRule[ei]=rule
            if example and not (hasattr(self.bestRule[ei], "learner") and hasattr(self.bestRule[ei].learner, "arg_example")):
                can_be_worst = True
            else:
                can_be_worst = False
            if can_be_worst and rule(e) and rule.quality>(e[self.probAttribute]-0.01):
                e[self.probAttribute] = rule.quality+0.001 # 0.001 is added to avoid numerical errors
                self.bestRule[ei]=rule
            elif rule(e) and rule.quality>e[self.probAttribute]:
                e[self.probAttribute] = rule.quality+0.001 # 0.001 is added to avoid numerical errors
                self.bestRule[ei]=rule
        return (examples,weights)

    def covered_percentage(self, examples):
        p = 0.0
        for ei, e in enumerate(examples):
            p += (e[self.probAttribute] - self.aprioriProb)/(1.0-self.aprioriProb)
        return p/len(examples)


# **************************************** #
# Estimation of extreme value distribution #
# **************************************** #

# Miscellaneous - utility functions
def avg(l):
    return sum(l)/len(l) if l else 0.

def var(l):
    if len(l)<2:
        return 0.
    av = avg(l)
    return sum([math.pow(li-av,2) for li in l])/(len(l)-1)

def perc(l,p):
    l.sort()
    return l[int(math.floor(p*len(l)))]

class EVDFitter:
    """ Randomizes a dataset and fits an extreme value distribution onto it. """

    def __init__(self, learner, n=200, randomseed=100):
        self.learner = learner
        self.n = n
        self.randomseed = randomseed
        
    def createRandomDataSet(self, data):
        newData = Orange.core.ExampleTable(data)
        # shuffle data
        cl_num = newData.toNumpy("C")
        random.shuffle(cl_num[0][:,0])
        clData = Orange.core.ExampleTable(Orange.core.Domain([newData.domain.classVar]),cl_num[0])
        for d_i,d in enumerate(newData):
            d[newData.domain.classVar] = clData[d_i][newData.domain.classVar]
        return newData

    def createEVDistList(self, evdList):
        l = Orange.core.EVDistList()
        for el in evdList:
            l.append(Orange.core.EVDist(mu=el[0],beta=el[1],percentiles=el[2]))
        return l

    # estimated fisher tippett parameters for a set of values given in vals list (+ deciles)
    def compParameters(self, vals, oldMi=0.5,oldBeta=1.1):                    
        # compute percentiles
        vals.sort()
        N = len(vals)
        percs = [avg(vals[int(float(N)*i/10):int(float(N)*(i+1)/10)]) for i in range(10)]            
        if N<10:
            return oldMi, oldBeta, percs
        beta = min(2.0, max(oldBeta, math.sqrt(6*var(vals)/math.pow(math.pi,2))))
        mi = max(oldMi,percs[-1]+beta*math.log(-math.log(0.95)))
        return mi, beta, percs

    def prepare_learner(self):
        self.oldStopper = self.learner.ruleFinder.ruleStoppingValidator
        self.evaluator = self.learner.ruleFinder.evaluator
        self.refiner = self.learner.ruleFinder.refiner
        self.validator = self.learner.ruleFinder.validator
        self.ruleFilter = self.learner.ruleFinder.ruleFilter
        self.learner.ruleFinder.validator = None
        self.learner.ruleFinder.evaluator = Orange.core.RuleEvaluator_LRS()
        self.learner.ruleFinder.evaluator.storeRules = True
        self.learner.ruleFinder.ruleStoppingValidator = Orange.core.RuleValidator_LRS(alpha=1.0)
        self.learner.ruleFinder.ruleStoppingValidator.max_rule_complexity = 0
        self.learner.ruleFinder.refiner = Orange.core.RuleBeamRefiner_Selector()
        self.learner.ruleFinder.ruleFilter = Orange.core.RuleBeamFilter_Width(width = 1)


    def restore_learner(self):
        self.learner.ruleFinder.evaluator = self.evaluator
        self.learner.ruleFinder.ruleStoppingValidator = self.oldStopper
        self.learner.ruleFinder.refiner = self.refiner
        self.learner.ruleFinder.validator = self.validator
        self.learner.ruleFinder.ruleFilter = self.ruleFilter

    def computeEVD(self, data, weightID=0, target_class=0, progress=None):
        # initialize random seed to make experiments repeatable
        random.seed(self.randomseed)

        # prepare learned for distribution computation        
        self.prepare_learner()

        # loop through N (sampling repetitions)
        extremeDists=[(0, 1, [])]
        self.learner.ruleFinder.ruleStoppingValidator.max_rule_complexity = self.oldStopper.max_rule_complexity
        maxVals = [[] for l in range(self.oldStopper.max_rule_complexity)]
        for d_i in range(self.n):
            if not progress:
                print d_i,
            else:
                progress(float(d_i)/self.n, None)                
            # create data set (remove and randomize)
            tempData = self.createRandomDataSet(data)
            self.learner.ruleFinder.evaluator.rules = Orange.core.RuleList()
            # Next, learn a rule
            self.learner.ruleFinder(tempData,weightID,target_class, Orange.core.RuleList())
            for l in range(self.oldStopper.max_rule_complexity):
                qs = [r.quality for r in self.learner.ruleFinder.evaluator.rules if r.complexity == l+1]
                if qs:
                    maxVals[l].append(max(qs))
                else:
                    maxVals[l].append(0)

        mu, beta = 1.0, 1.0
        for mi,m in enumerate(maxVals):
            mu, beta, perc = self.compParameters(m,mu,beta)
            extremeDists.append((mu, beta, perc))
            extremeDists.extend([(0,1,[])]*(mi))

        self.restore_learner()
        return self.createEVDistList(extremeDists)

# ************************* #
# Rule based classification #
# ************************* #

class CrossValidation:
    def __init__(self, folds=5, randomGenerator = 150):
        self.folds = folds
        self.randomGenerator = randomGenerator

    def __call__(self, learner, examples, weight):
        res = orngTest.crossValidation([learner], (examples, weight), folds = self.folds, randomGenerator = self.randomGenerator)
        return self.get_prob_from_res(res, examples)

    def get_prob_from_res(self, res, examples):
        probDist = Orange.core.DistributionList()
        for tex in res.results:
            d = Orange.core.Distribution(examples.domain.classVar)
            for di in range(len(d)):
                d[di] = tex.probabilities[0][di]
            probDist.append(d)
        return probDist

class PILAR:
    """ PILAR (Probabilistic improvement of learning algorithms with rules) """
    def __init__(self, alternative_learner = None, min_cl_sig = 0.5, min_beta = 0.0, set_prefix_rules = False, optimize_betas = True):
        self.alternative_learner = alternative_learner
        self.min_cl_sig = min_cl_sig
        self.min_beta = min_beta
        self.set_prefix_rules = set_prefix_rules
        self.optimize_betas = optimize_betas
        self.selected_evaluation = CrossValidation(folds=5)

    def __call__(self, rules, examples, weight=0):
        rules = self.add_null_rule(rules, examples, weight)
        if self.alternative_learner:
            probDist = self.selected_evaluation(self.alternative_learner, examples, weight)
            classifier = self.alternative_learner(examples,weight)
##            probDist = Orange.core.DistributionList()
##            for e in examples:
##                probDist.append(classifier(e,Orange.core.GetProbabilities))
            cl = Orange.core.RuleClassifier_logit(rules, self.min_cl_sig, self.min_beta, examples, weight, self.set_prefix_rules, self.optimize_betas, classifier, probDist)
        else:
            cl = Orange.core.RuleClassifier_logit(rules, self.min_cl_sig, self.min_beta, examples, weight, self.set_prefix_rules, self.optimize_betas)

##        print "result"
        for ri,r in enumerate(cl.rules):
            cl.rules[ri].setattr("beta",cl.ruleBetas[ri])
##            if cl.ruleBetas[ri] > 0:
##                print Orange.classification.rules.ruleToString(r), r.quality, cl.ruleBetas[ri]
        cl.all_rules = cl.rules
        cl.rules = self.sortRules(cl.rules)
        cl.ruleBetas = [r.beta for r in cl.rules]
        cl.setattr("data", examples)
        return cl

    def add_null_rule(self, rules, examples, weight):
        for cl in examples.domain.classVar:
            tmpRle = Orange.core.Rule()
            tmpRle.filter = Orange.core.Filter_values(domain = examples.domain)
            tmpRle.parentRule = None
            tmpRle.filterAndStore(examples,weight,int(cl))
            tmpRle.quality = tmpRle.classDistribution[int(cl)]/tmpRle.classDistribution.abs
            rules.append(tmpRle)
        return rules
        
    def sortRules(self, rules):
        newRules = Orange.core.RuleList()
        foundRule = True
        while foundRule:
            foundRule = False
            bestRule = None
            for r in rules:
                if r in newRules:
                    continue
                if r.beta < 0.01 and r.beta > -0.01:
                    continue
                if not bestRule:
                    bestRule = r
                    foundRule = True
                    continue
                if len(r.filter.conditions) < len(bestRule.filter.conditions):
                    bestRule = r
                    foundRule = True
                    continue
                if len(r.filter.conditions) ==  len(bestRule.filter.conditions) and r.beta > bestRule.beta:
                    bestRule = r
                    foundRule = True
                    continue
            if bestRule:
                newRules.append(bestRule)
        return newRules     


class CN2UnorderedClassifier(Orange.core.RuleClassifier):
    """ Classification from rules as in CN2. """
    def __init__(self, rules, examples, weightID = 0, **argkw):
        self.rules = rules
        self.examples = examples
        self.weightID = weightID
        self.prior = Orange.core.Distribution(examples.domain.classVar, examples, weightID)
        self.__dict__.update(argkw)

    def __call__(self, example, result_type=Orange.core.GetValue, retRules = False):
        # iterate through the set of induced rules: self.rules and sum their distributions 
        ret_dist = self.sum_distributions([r for r in self.rules if r(example)])
        # normalize
        a = sum(ret_dist)
        for ri, r in enumerate(ret_dist):
            ret_dist[ri] = ret_dist[ri]/a
##        ret_dist.normalize()
        # return value
        if result_type == Orange.core.GetValue:
          return ret_dist.modus()
        if result_type == Orange.core.GetProbabilities:
          return ret_dist
        return (ret_dist.modus(),ret_dist)

    def sum_distributions(self, rules):
        if not rules:
            return self.prior
        empty_disc = Orange.core.Distribution(rules[0].examples.domain.classVar)
        for r in rules:
            for i,d in enumerate(r.classDistribution):
                empty_disc[i] = empty_disc[i] + d
        return empty_disc

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += Orange.classification.rules.ruleToString(r)+" "+str(r.classDistribution)+"\n"
        return retStr


class RuleClassifier_bestRule(Orange.core.RuleClassifier):
    """ A very simple classifier, it takes the best rule of each class and normalizes probabilities. """
    def __init__(self, rules, examples, weightID = 0, **argkw):
        self.rules = rules
        self.examples = examples
        self.apriori = Orange.core.Distribution(examples.domain.classVar,examples,weightID)
        self.aprioriProb = [a/self.apriori.abs for a in self.apriori]
        self.weightID = weightID
        self.__dict__.update(argkw)
        self.defaultClassIndex = -1

    def __call__(self, example, result_type=Orange.core.GetValue, retRules = False):
        example = Orange.core.Example(self.examples.domain,example)
        tempDist = Orange.core.Distribution(example.domain.classVar)
        bestRules = [None]*len(example.domain.classVar.values)

        for r in self.rules:
            if r(example) and not self.defaultClassIndex == int(r.classifier.defaultVal) and \
               (not bestRules[int(r.classifier.defaultVal)] or r.quality>tempDist[r.classifier.defaultVal]):
                tempDist[r.classifier.defaultVal] = r.quality
                bestRules[int(r.classifier.defaultVal)] = r
        for b in bestRules:
            if b:
                used = getattr(b,"used",0.0)
                b.setattr("used",used+1)
        nonCovPriorSum = sum([tempDist[i] == 0. and self.aprioriProb[i] or 0. for i in range(len(self.aprioriProb))])
        if tempDist.abs < 1.:
            residue = 1. - tempDist.abs
            for a_i,a in enumerate(self.aprioriProb):
                if tempDist[a_i] == 0.:
                    tempDist[a_i]=self.aprioriProb[a_i]*residue/nonCovPriorSum
            finalDist = tempDist #Orange.core.Distribution(example.domain.classVar)
        else:
            tempDist.normalize() # prior probability
            tmpExamples = Orange.core.ExampleTable(self.examples)
            for r in bestRules:
                if r:
                    tmpExamples = r.filter(tmpExamples)
            tmpDist = Orange.core.Distribution(tmpExamples.domain.classVar,tmpExamples,self.weightID)
            tmpDist.normalize()
            probs = [0.]*len(self.examples.domain.classVar.values)
            for i in range(len(self.examples.domain.classVar.values)):
                probs[i] = tmpDist[i]+tempDist[i]*2
            finalDist = Orange.core.Distribution(self.examples.domain.classVar)
            for cl_i,cl in enumerate(self.examples.domain.classVar):
                finalDist[cl] = probs[cl_i]
            finalDist.normalize()
                
        if retRules: # Do you want to return rules with classification?
            if result_type == Orange.core.GetValue:
              return (finalDist.modus(),bestRules)
            if result_type == Orange.core.GetProbabilities:
              return (finalDist, bestRules)
            return (finalDist.modus(),finalDist, bestRules)
        if result_type == Orange.core.GetValue:
          return finalDist.modus()
        if result_type == Orange.core.GetProbabilities:
          return finalDist
        return (finalDist.modus(),finalDist)