"""

.. index:: rule induction

.. index:: 
   single: classification; rule induction

**************************
Rule induction (``rules``)
**************************

This module implements supervised rule induction algorithms
and rule-based classification methods, specifically the 
`CN2 induction algorithm <http://www.springerlink.com/content/k6q2v76736w5039r/>`_
in multiple variants, including an argument-based learning one. 
The implementation is modular, based on the rule induction 
framework that is described below, providing the opportunity to change, specialize
and improve the algorithm.

CN2 algorithm
=============

.. index:: 
   single: classification; CN2

Several variations of well-known CN2 rule learning algorithms are implemented.
All are implemented by wrapping the
:class:`~Orange.classification.rules.RuleLearner` class. Each CN2 learner class
in this module changes some of RuleLearner's replaceable components to reflect
the required behavior.

Usage is consistent with typical learner usage in Orange:

:download:`rules-cn2.py <code/rules-cn2.py>` (uses :download:`titanic.tab <code/titanic.tab>`)

.. literalinclude:: code/rules-cn2.py
    :lines: 7-

The result::
    
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
   :exclude-members: baseRules, beamWidth, coverAndRemove, dataStopping,
      ruleFinder, ruleStopping, storeInstances, targetClass, weightID
   
.. autoclass:: Orange.classification.rules.CN2Classifier
   :members:
   :show-inheritance:
   :exclude-members: beamWidth, resultType
   
.. index:: unordered CN2

.. index:: 
   single: classification; unordered CN2

.. autoclass:: Orange.classification.rules.CN2UnorderedLearner
   :members:
   :show-inheritance:
   :exclude-members: baseRules, beamWidth, coverAndRemove, dataStopping,
      ruleFinder, ruleStopping, storeInstances, targetClass, weightID
   
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
   :exclude-members: baseRules, beamWidth, coverAndRemove, dataStopping,
      ruleFinder, ruleStopping, storeInstances, targetClass, weightID
   
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
  Learning, pp 151--163, Porto, Portugal, March 1991.
* Lavrac, Kavsek, Flach, Todorovski: `Subgroup Discovery with CN2-SD
  <http://jmlr.csail.mit.edu/papers/volume5/lavrac04a/lavrac04a.pdf>`_. Journal
  of Machine Learning Research 5: 153-188, 2004.


Argument based CN2
==================

Orange also supports argument-based CN2 learning.

.. autoclass:: Orange.classification.rules.ABCN2
   :members:
   :show-inheritance:
   :exclude-members: baseRules, beamWidth, coverAndRemove, dataStopping,
      ruleFinder, ruleStopping, storeInstances, targetClass, weightID,
      argument_id
   
   This class has many more undocumented methods; see the source code for
   reference.
   
.. autoclass:: Orange.classification.rules.ABCN2Ordered
   :members:
   :show-inheritance:
   
.. autoclass:: Orange.classification.rules.ABCN2M
   :members:
   :show-inheritance:
   :exclude-members: baseRules, beamWidth, coverAndRemove, dataStopping,
      ruleFinder, ruleStopping, storeInstances, targetClass, weightID

Thismodule has many more undocumented argument-based learning related classed;
see the source code for reference.

References
----------

* Bratko, Mozina, Zabkar. `Argument-Based Machine Learning
  <http://www.springerlink.com/content/f41g17t1259006k4/>`_. Lecture Notes in
  Computer Science: vol. 4203/2006, 11-17, 2006.


Rule induction framework
========================

A general framework of classes supports the described CN2 implementation, and
can in fact be fine-tuned to specific needs by replacing individual components.
Here is a simple example, while a detailed architecture can be observed
in description of classes that follows it:

part of :download:`rules-customized.py <code/rules-customized.py>` (uses :download:`titanic.tab <code/titanic.tab>`)

.. literalinclude:: code/rules-customized.py
    :lines: 7-17

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

Notice that it is first necessary to set the :obj:`rule_finder` component,
because the default components are not constructed when the learner is
constructed, but only when we run it on data. At that time, the algorithm
checks which components are necessary and sets defaults. Similarly, when the
learner finishes, it destructs all *default* components. Continuing with our
example, assume that we wish to set a different validation function and a
different bean width. This is simply written as:

part of :download:`rules-customized.py <code/rules-customized.py>` (uses :download:`titanic.tab <code/titanic.tab>`)

.. literalinclude:: code/rules-customized.py
    :lines: 19-23

.. py:class:: Orange.classification.rules.Rule(filter, classifier, lr, dist, ce, w = 0, qu = -1)
   
   Representation of a single induced rule.
   
   Parameters, that can be passed to the constructor, correspond to the first
   7 attributes. All attributes are:
   
   .. attribute:: filter
   
      contents of the rule; this is the basis of the Rule class. Must be of
      type :class:`Orange.core.Filter`; an instance of
      :class:`Orange.core.Filter_values` is set as a default.
   
   .. attribute:: classifier
      
      each rule can be used as a classical Orange like
      classifier. Must be of type :class:`Orange.classification.Classifier`.
      By default, an instance of :class:`Orange.classification.ConstantClassifier` is used.
   
   .. attribute:: learner
      
      learner to be used for making a classifier. Must be of type
      :class:`Orange.classification.Learner`. By default,
      :class:`Orange.classification.majority.MajorityLearner` is used.
   
   .. attribute:: class_distribution
      
      distribution of class in data instances covered by this rule
      (:class:`Orange.statistics.distribution.Distribution`).
   
   .. attribute:: examples
      
      data instances covered by this rule (:class:`Orange.data.Table`).
   
   .. attribute:: weight_id
   
      ID of the weight meta-attribute for the stored data instances (int).
   
   .. attribute:: quality
      
      quality of the rule. Rules with higher quality are better (float).
   
   .. attribute:: complexity
   
      complexity of the rule (float). Complexity is used for
      selecting between rules with equal quality, where rules with lower
      complexity are preferred. Typically, complexity corresponds to the
      number of selectors in rule (actually to number of conditions in filter),
      but, obviously, any other measure can be applied.
   
   .. method:: filter_and_store(instances, weight_id=0, target_class=-1)
   
      Filter passed data instances and store them in the attribute 'examples'.
      Also, compute class_distribution, set weight of stored examples and create
      a new classifier using 'learner' attribute.
      
      :param weight_id: ID of the weight meta-attribute.
      :type weight_id: int
      :param target_class: index of target class; -1 for all.
      :type target_class: int
   
   Objects of this class can be invoked:

   .. method:: __call__(instance, instances, weight_id=0, target_class=-1)
   
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

.. py:class:: Orange.classification.rules.RuleLearner(store_instances = true, target_class = -1, base_rules = Orange.classification.rules.RuleList())
   
   Bases: :class:`Orange.classification.Learner`
   
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

      def \_\_call\_\_(self, instances, weight_id=0):
          rule_list = Orange.classification.rules.RuleList()
          all_instances = Orange.data.Table(instances)
          while not self.\ **data_stopping**\ (instances, weight_id, self.target_class):
              new_rule = self.\ **rule_finder**\ (instances, weight_id, self.target_class,
                                        self.base_rules)
              if self.\ **rule_stopping**\ (rule_list, new_rule, instances, weight_id):
                  break
              instances, weight_id = self.\ **cover_and_remove**\ (new_rule, instances,
                                                      weight_id, self.target_class)
              rule_list.append(new_rule)
          return Orange.classification.rules.RuleClassifier_FirstRule(
              rules=rule_list, instances=all_instances)
                
   The four customizable components here are the invoked :obj:`data_stopping`,
   :obj:`rule_finder`, :obj:`cover_and_remove` and :obj:`rule_stopping`
   objects. By default, components of the original CN2 algorithm are be used,
   but this can be changed by modifying those attributes:
   
   .. attribute:: data_stopping
   
      an object of class
      :class:`~Orange.classification.rules.RuleDataStoppingCriteria`
      that determines whether there will be any benefit from further learning
      (ie. if there is enough data to continue learning). The default
      implementation
      (:class:`~Orange.classification.rules.RuleDataStoppingCriteria_NoPositives`)
      returns True if there are no more instances of given class. 
   
   .. attribute:: rule_stopping
      
      an object of class 
      :class:`~Orange.classification.rules.RuleStoppingCriteria`
      that decides from the last rule learned if it is worthwhile to use the
      rule and learn more rules. By default, no rule stopping criteria is
      used (:obj:`rule_stopping` == :obj:`None`), thus accepting all
      rules.
       
   .. attribute:: cover_and_remove
       
      an object of
      :class:`RuleCovererAndRemover` that removes
      instances covered by the rule and returns remaining instances. The
      default implementation
      (:class:`RuleCovererAndRemover_Default`)
      only removes the instances that belong to given target class, except if
      it is not given (ie. :obj:`target_class` == -1).
    
   .. attribute:: rule_finder
      
      an object of class
      :class:`~Orange.classification.rules.RuleFinder` that learns a single
      rule from instances. Default implementation is
      :class:`~Orange.classification.rules.RuleBeamFinder`.

   :param store_instances: if set to True, the rules will have data instances
       stored.
   :type store_instances: bool
    
   :param target_class: index of a specific class being learned; -1 for all.
   :type target_class: int
   
   :param base_rules: Rules that we would like to use in :obj:`rule_finder` to
       constrain the learning space. If not set, it will be set to a set
       containing only an empty rule.
   :type base_rules: :class:`~Orange.classification.rules.RuleList`

Rule finders
------------

.. class:: Orange.classification.rules.RuleFinder

   Base class for all rule finders. These are used to learn a single rule from
   instances.
   
   Rule finders are invokable in the following manner:
   
   .. method:: __call__(table, weight_id, target_class, base_rules)
   
      Return a new rule, induced from instances in the given table.
      
      :param table: data instances to learn from.
      :type table: :class:`Orange.data.Table`
      
      :param weight_id: ID of the weight meta-attribute for the stored data
          instances.
      :type weight_id: int
      
      :param target_class: index of a specific class being learned; -1 for all.
      :type target_class: int 
      
      :param base_rules: Rules that we would like to use in :obj:`rule_finder`
          to constrain the learning space. If not set, it will be set to a set
          containing only an empty rule.
      :type base_rules: :class:`~Orange.classification.rules.RuleList`

.. class:: Orange.classification.rules.RuleBeamFinder
   
   Bases: :class:`~Orange.classification.rules.RuleFinder`
   
   Beam search for the best rule. This is the default class used in RuleLearner
   to find the best rule. Pseudo code of the algorithm is shown here:

   .. parsed-literal::

      def \_\_call\_\_(self, table, weight_id, target_class, base_rules):
          prior = Orange.statistics.distribution.Distribution(table.domain.class_var, table, weight_id)
          rules_star, best_rule = self.\ **initializer**\ (table, weight_id, target_class, base_rules, self.evaluator, prior)
          \# compute quality of rules in rules_star and best_rule
          ...
          while len(rules_star) \> 0:
              candidates, rules_star = self.\ **candidate_selector**\ (rules_star, table, weight_id)
              for cand in candidates:
                  new_rules = self.\ **refiner**\ (cand, table, weight_id, target_class)
                  for new_rule in new_rules:
                      if self.\ **rule_stopping_validator**\ (new_rule, table, weight_id, target_class, cand.class_distribution):
                          new_rule.quality = self.\ **evaluator**\ (new_rule, table, weight_id, target_class, prior)
                          rules_star.append(new_rule)
                          if self.\ **validator**\ (new_rule, table, weight_id, target_class, prior) and
                              new_rule.quality \> best_rule.quality:
                              best_rule = new_rule
              rules_star = self.\ **rule_filter**\ (rules_star, table, weight_id)
          return best_rule

   Bolded in the pseudo-code are several exchangeable components, exposed as
   attributes. These are:

   .. attribute:: initializer
   
      an object of class
      :class:`~Orange.classification.rules.RuleBeamInitializer`
      used to initialize :obj:`rules_star` and for selecting the
      initial best rule. By default
      (:class:`~Orange.classification.rules.RuleBeamInitializer_Default`),
      :obj:`base_rules` are returned as starting :obj:`rulesSet` and the best
      from :obj:`base_rules` is set as :obj:`best_rule`. If :obj:`base_rules`
      are not set, this class will return :obj:`rules_star` with rule that
      covers all instances (has no selectors) and this rule will be also used
      as :obj:`best_rule`.
   
   .. attribute:: candidate_selector
   
      an object of class
      :class:`~Orange.classification.rules.RuleBeamCandidateSelector`
      used to separate a subset from the current
      :obj:`rules_star` and return it. These rules will be used in the next
      specification step. Default component (an instance of
      :class:`~Orange.classification.rules.RuleBeamCandidateSelector_TakeAll`)
      takes all rules in :obj:`rules_star`.
    
   .. attribute:: refiner
   
      an object of class
      :class:`~Orange.classification.rules.RuleBeamRefiner`
      used to refine given rule. New rule should cover a
      strict subset of examples covered by given rule. Default component
      (:class:`~Orange.classification.rules.RuleBeamRefiner_Selector`) adds
      a conjunctive selector to selectors present in the rule.
    
   .. attribute:: rule_filter
   
      an object of class
      :class:`~Orange.classification.rules.RuleBeamFilter`
      used to filter rules to keep beam relatively small
      to contain search complexity. By default, it takes five best rules:
      :class:`~Orange.classification.rules.RuleBeamFilter_Width`\ *(m=5)*\ .

   .. method:: __call__(data, weight_id, target_class, base_rules)

   Determines the next best rule to cover the remaining data instances.
   
   :param data: data instances.
   :type data: :class:`Orange.data.Table`
   
   :param weight_id: index of the weight meta-attribute.
   :type weight_id: int
   
   :param target_class: index of the target class.
   :type target_class: int
   
   :param base_rules: existing rules.
   :type base_rules: :class:`~Orange.classification.rules.RuleList`

Rule evaluators
---------------

.. class:: Orange.classification.rules.RuleEvaluator

   Base class for rule evaluators that evaluate the quality of the rule based
   on covered data instances. All evaluators support being invoked in the
   following manner:
   
   .. method:: __call__(rule, instances, weight_id, target_class, prior)
   
      Calculates a non-negative rule quality.
      
      :param rule: rule to evaluate.
      :type rule: :class:`~Orange.classification.rules.Rule`
      
      :param instances: a table of instances, covered by the rule.
      :type instances: :class:`Orange.data.Table`
      
      :param weight_id: index of the weight meta-attribute.
      :type weight_id: int
      
      :param target_class: index of target class of this rule.
      :type target_class: int
      
      :param prior: prior class distribution.
      :type prior: :class:`Orange.statistics.distribution.Distribution`

.. autoclass:: Orange.classification.rules.LaplaceEvaluator
   :members:
   :show-inheritance:
   :exclude-members: targetClass, weightID

.. autoclass:: Orange.classification.rules.WRACCEvaluator
   :members:
   :show-inheritance:
   :exclude-members: targetClass, weightID
   
.. class:: Orange.classification.rules.RuleEvaluator_Entropy

   Bases: :class:`~Orange.classification.rules.RuleEvaluator`
    
.. class:: Orange.classification.rules.RuleEvaluator_LRS

   Bases: :class:`~Orange.classification.rules.RuleEvaluator`

.. class:: Orange.classification.rules.RuleEvaluator_Laplace

   Bases: :class:`~Orange.classification.rules.RuleEvaluator`

.. class:: Orange.classification.rules.RuleEvaluator_mEVC

   Bases: :class:`~Orange.classification.rules.RuleEvaluator`
   
Instance covering and removal
-----------------------------

.. class:: RuleCovererAndRemover

   Base class for rule coverers and removers that, when invoked, remove
   instances covered by the rule and return remaining instances.

   .. method:: __call__(rule, instances, weights, target_class)
   
      Calculates a non-negative rule quality.
      
      :param rule: rule to evaluate.
      :type rule: :class:`~Orange.classification.rules.Rule`
      
      :param instances: a table of instances, covered by the rule.
      :type instances: :class:`Orange.data.Table`
      
      :param weights: index of the weight meta-attribute.
      :type weights: int
      
      :param target_class: index of target class of this rule.
      :type target_class: int

.. autoclass:: CovererAndRemover_MultWeights

.. autoclass:: CovererAndRemover_AddWeights
   
Miscellaneous functions
-----------------------

.. automethod:: Orange.classification.rules.rule_to_string

..
    Undocumented are:
    Data-based Stopping Criteria
    ----------------------------
    Rule-based Stopping Criteria
    ----------------------------
    Rule-based Stopping Criteria
    ----------------------------

"""

import random
import math
import operator
import numpy

import Orange
import Orange.core
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
from Orange.misc import deprecated_keywords
from Orange.misc import deprecated_members


class ConvertClass:
    """ Converting class variables into dichotomous class variable. """
    def __init__(self, classAtt, classValue, newClassAtt):
        self.classAtt = classAtt
        self.classValue = classValue
        self.newClassAtt = newClassAtt

    def __call__(self, example, returnWhat):
        if example[self.classAtt] == self.classValue:
            return Orange.data.Value(self.newClassAtt, self.classValue + "_")
        else:
            return Orange.data.Value(self.newClassAtt, "not " + self.classValue)


def create_dichotomous_class(domain, att, value, negate, removeAtt=None):
    # create new variable
    newClass = Orange.feature.Discrete(att.name + "_", values=[str(value) + "_", "not " + str(value)])
    positive = Orange.data.Value(newClass, str(value) + "_")
    negative = Orange.data.Value(newClass, "not " + str(value))
    newClass.getValueFrom = ConvertClass(att, str(value), newClass)

    att = [a for a in domain.attributes]
    newDomain = Orange.data.Domain(att + [newClass])
    newDomain.addmetas(domain.getmetas())
    if negate == 1:
        return (newDomain, negative)
    else:
        return (newDomain, positive)


class LaplaceEvaluator(RuleEvaluator):
    """
    Laplace's rule of succession.
    """
    def __call__(self, rule, data, weight_id, target_class, apriori):
        if not rule.class_distribution:
            return 0.
        sumDist = rule.class_distribution.cases
        if not sumDist or (target_class > -1 and not rule.class_distribution[target_class]):
            return 0.
        # get distribution
        if target_class > -1:
            return (rule.class_distribution[target_class] + 1) / (sumDist + 2)
        else:
            return (max(rule.class_distribution) + 1) / (sumDist + len(data.domain.class_var.values))

LaplaceEvaluator = deprecated_members({"weightID": "weight_id",
                                       "targetClass": "target_class"})(LaplaceEvaluator)


class WRACCEvaluator(RuleEvaluator):
    """
    Weighted relative accuracy.
    """
    def __call__(self, rule, data, weight_id, target_class, apriori):
        if not rule.class_distribution:
            return 0.
        sumDist = rule.class_distribution.cases
        if not sumDist or (target_class > -1 and not rule.class_distribution[target_class]):
            return 0.
        # get distribution
        if target_class > -1:
            pRule = rule.class_distribution[target_class] / apriori[target_class]
            pTruePositive = rule.class_distribution[target_class] / sumDist
            pClass = apriori[target_class] / apriori.cases
        else:
            pRule = sumDist / apriori.cases
            pTruePositive = max(rule.class_distribution) / sumDist
            pClass = apriori[rule.class_distribution.modus()] / sum(apriori)
        if pTruePositive > pClass:
            return pRule * (pTruePositive - pClass)
        else: return (pTruePositive - pClass) / max(pRule, 1e-6)

WRACCEvaluator = deprecated_members({"weightID": "weight_id",
                                     "targetClass": "target_class"})(WRACCEvaluator)


class MEstimateEvaluator(RuleEvaluator):
    """
    Rule evaluator using m-estimate of probability rule evaluation function.
    
    :param m: m-value for m-estimate
    :type m: int
    
    """
    def __init__(self, m=2):
        self.m = m
    def __call__(self, rule, data, weight_id, target_class, apriori):
        if not rule.class_distribution:
            return 0.
        sumDist = rule.class_distribution.abs
        if self.m == 0 and not sumDist:
            return 0.
        # get distribution
        if target_class > -1:
            p = rule.class_distribution[target_class] + self.m * apriori[target_class] / apriori.abs
            p = p / (rule.class_distribution.abs + self.m)
        else:
            p = max(rule.class_distribution) + self.m * apriori[rule.\
                class_distribution.modus()] / apriori.abs
            p = p / (rule.class_distribution.abs + self.m)
        return p

MEstimateEvaluator = deprecated_members({"weightID": "weight_id",
                                         "targetClass": "target_class"})(MEstimateEvaluator)


class CN2Learner(RuleLearner):
    """
    Classical CN2 (see Clark and Niblett; 1988) induces a set of ordered
    rules, which means that classificator must try these rules in the same
    order as they were learned.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    :param evaluator: an object that evaluates a rule from covered instances.
        By default, entropy is used as a measure. 
    :type evaluator: :class:`~Orange.classification.rules.RuleEvaluator`
    :param beam_width: width of the search beam.
    :type beam_width: int
    :param alpha: significance level of the likelihood ratio statistics to
        determine whether rule is better than the default rule.
    :type alpha: float

    """

    def __new__(cls, instances=None, weight_id=0, **kwargs):
        self = RuleLearner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(**kwargs)
            return self.__call__(instances, weight_id)
        else:
            return self

    def __init__(self, evaluator=RuleEvaluator_Entropy(), beam_width=5,
        alpha=1.0, **kwds):
        self.__dict__.update(kwds)
        self.rule_finder = RuleBeamFinder()
        self.rule_finder.ruleFilter = RuleBeamFilter_Width(width=beam_width)
        self.rule_finder.evaluator = evaluator
        self.rule_finder.validator = RuleValidator_LRS(alpha=alpha)

    def __call__(self, instances, weight=0):
        supervisedClassCheck(instances)

        cl = RuleLearner.__call__(self, instances, weight)
        rules = cl.rules
        return CN2Classifier(rules, instances, weight)

CN2Learner = deprecated_members({"beamWidth": "beam_width",
                     "ruleFinder": "rule_finder",
                     "ruleStopping": "rule_stopping",
                     "dataStopping": "data_stopping",
                     "coverAndRemove": "cover_and_remove",
                     "storeInstances": "store_instances",
                     "targetClass": "target_class",
                     "baseRules": "base_rules",
                     "weightID": "weight_id"})(CN2Learner)


class CN2Classifier(RuleClassifier):
    """
    Classical CN2 (see Clark and Niblett; 1988) classifies a new instance
    using an ordered set of rules. Usually the learner
    (:class:`~Orange.classification.rules.CN2Learner`) is used to construct the
    classifier.
    
    :param rules: learned rules to be used for classification (mandatory).
    :type rules: :class:`~Orange.classification.rules.RuleList`
    
    :param instances: data instances that were used for learning.
    :type instances: :class:`Orange.data.Table`
    
    :param weight_id: ID of the weight meta-attribute.
    :type weight_id: int

    """

    @deprecated_keywords({"examples": "instances"})
    def __init__(self, rules=None, instances=None, weight_id=0, **argkw):
        self.rules = rules
        self.examples = instances
        self.weight_id = weight_id
        self.class_var = None if instances is None else instances.domain.class_var
        self.__dict__.update(argkw)
        if instances is not None:
            self.prior = Orange.statistics.distribution.Distribution(instances.domain.class_var, instances)

    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue):
        """
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.distribution.Distribution` or a tuple with both
        """
        classifier = None
        for r in self.rules:
         #   r.filter.domain = instance.domain
            if r(instance) and r.classifier:
                classifier = r.classifier
                classifier.defaultDistribution = r.class_distribution
                break
        if not classifier:
            classifier = Orange.classification.ConstantClassifier(instance.domain.class_var, \
                self.prior.modus())
            classifier.defaultDistribution = self.prior

        classifier.defaultDistribution.normalize()
        if result_type == Orange.classification.Classifier.GetValue:
          return classifier(instance)
        if result_type == Orange.classification.Classifier.GetProbabilities:
          return classifier.default_distribution
        return (classifier(instance), classifier.default_distribution)

    def __str__(self):
        ret_str = rule_to_string(self.rules[0]) + " " + str(self.rules[0].\
            class_distribution) + "\n"
        for r in self.rules[1:]:
            ret_str += "ELSE " + rule_to_string(r) + " " + str(r.class_distribution) + "\n"
        return ret_str

CN2Classifier = deprecated_members({"resultType": "result_type",
                                    "beamWidth": "beam_width"})(CN2Classifier)


class CN2UnorderedLearner(RuleLearner):
    """
    CN2 unordered (see Clark and Boswell; 1991) induces a set of unordered
    rules - classification from rules does not assume ordering of rules.
    Learning rules is quite similar to learning in classical CN2, where
    the process of learning of rules is separated to learning rules for each
    class.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    :param evaluator: an object that evaluates a rule from covered instances.
        By default, Laplace's rule of succession is used as a measure. 
    :type evaluator: :class:`~Orange.classification.rules.RuleEvaluator`
    :param beam_width: width of the search beam.
    :type beam_width: int
    :param alpha: significance level of the likelihood ratio statistics to
        determine whether rule is better than the default rule.
    :type alpha: float
    """
    def __new__(cls, instances=None, weight_id=0, **kwargs):
        self = RuleLearner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(**kwargs)
            return self.__call__(instances, weight_id)
        else:
            return self

    def __init__(self, evaluator=RuleEvaluator_Laplace(), beam_width=5,
        alpha=1.0, **kwds):
        self.__dict__.update(kwds)
        self.rule_finder = RuleBeamFinder()
        self.rule_finder.ruleFilter = RuleBeamFilter_Width(width=beam_width)
        self.rule_finder.evaluator = evaluator
        self.rule_finder.validator = RuleValidator_LRS(alpha=alpha)
        self.rule_finder.rule_stoppingValidator = RuleValidator_LRS(alpha=1.0)
        self.rule_stopping = RuleStopping_Apriori()
        self.data_stopping = RuleDataStoppingCriteria_NoPositives()

    @deprecated_keywords({"weight": "weight_id"})
    def __call__(self, instances, weight_id=0):
        supervisedClassCheck(instances)

        rules = RuleList()
        self.rule_stopping.apriori = Orange.statistics.distribution.Distribution(
            instances.domain.class_var, instances)
        progress = getattr(self, "progressCallback", None)
        if progress:
            progress.start = 0.0
            progress.end = 0.0
            distrib = Orange.statistics.distribution.Distribution(
                instances.domain.class_var, instances, weight_id)
            distrib.normalize()
        for target_class in instances.domain.class_var:
            if progress:
                progress.start = progress.end
                progress.end += distrib[target_class]
            self.target_class = target_class
            cl = RuleLearner.__call__(self, instances, weight_id)
            for r in cl.rules:
                rules.append(r)
        if progress:
            progress(1.0, None)
        return CN2UnorderedClassifier(rules, instances, weight_id)

CN2UnorderedLearner = deprecated_members({"beamWidth": "beam_width",
                     "ruleFinder": "rule_finder",
                     "ruleStopping": "rule_stopping",
                     "dataStopping": "data_stopping",
                     "coverAndRemove": "cover_and_remove",
                     "storeInstances": "store_instances",
                     "targetClass": "target_class",
                     "baseRules": "base_rules",
                     "weightID": "weight_id"})(CN2UnorderedLearner)


class CN2UnorderedClassifier(RuleClassifier):
    """
    CN2 unordered (see Clark and Boswell; 1991) classifies a new instance using
    a set of unordered rules. Usually the learner
    (:class:`~Orange.classification.rules.CN2UnorderedLearner`) is used to
    construct the classifier.
    
    :param rules: learned rules to be used for classification (mandatory).
    :type rules: :class:`~Orange.classification.rules.RuleList`
    
    :param instances: data instances that were used for learning.
    :type instances: :class:`Orange.data.Table`
    
    :param weight_id: ID of the weight meta-attribute.
    :type weight_id: int

    """

    @deprecated_keywords({"examples": "instances"})
    def __init__(self, rules=None, instances=None, weight_id=0, **argkw):
        self.rules = rules
        self.examples = instances
        self.weight_id = weight_id
        self.class_var = instances.domain.class_var if instances is not None else None
        self.__dict__.update(argkw)
        if instances is not None:
            self.prior = Orange.statistics.distribution.Distribution(
                                instances.domain.class_var, instances)

    @deprecated_keywords({"retRules": "ret_rules"})
    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue, ret_rules=False):
        """
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.distribution.Distribution` or a tuple with both
        """
        def add(disc1, disc2, sumd):
            disc = Orange.statistics.distribution.Discrete(disc1)
            sumdisc = sumd
            for i, d in enumerate(disc):
                disc[i] += disc2[i]
                sumdisc += disc2[i]
            return disc, sumdisc

        # create empty distribution
        retDist = Orange.statistics.distribution.Discrete(self.examples.domain.class_var)
        covRules = RuleList()
        # iterate through instances - add distributions
        sumdisc = 0.
        for r in self.rules:
            if r(instance) and r.class_distribution:
                retDist, sumdisc = add(retDist, r.class_distribution, sumdisc)
                covRules.append(r)
        if not sumdisc:
            retDist = self.prior
            sumdisc = self.prior.abs

        if sumdisc > 0.0:
            for c in self.examples.domain.class_var:
                retDist[c] /= sumdisc
        else:
            retDist.normalize()

        if ret_rules:
            if result_type == Orange.classification.Classifier.GetValue:
              return (retDist.modus(), covRules)
            if result_type == Orange.classification.Classifier.GetProbabilities:
              return (retDist, covRules)
            return (retDist.modus(), retDist, covRules)
        if result_type == Orange.classification.Classifier.GetValue:
          return retDist.modus()
        if result_type == Orange.classification.Classifier.GetProbabilities:
          return retDist
        return (retDist.modus(), retDist)

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += rule_to_string(r) + " " + str(r.class_distribution) + "\n"
        return retStr


class CN2SDUnorderedLearner(CN2UnorderedLearner):
    """
    CN2-SD (see Lavrac et al.; 2004) induces a set of unordered rules, which
    is the same as :class:`~Orange.classification.rules.CN2UnorderedLearner`.
    The difference between classical CN2 unordered and CN2-SD is selection of
    specific evaluation function and covering function:
    :class:`WRACCEvaluator` is used to implement
    weight-relative accuracy and 
    :class:`CovererAndRemover_MultWeights` avoids
    excluding covered instances, multiplying their weight by the value of
    mult parameter instead.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    :param evaluator: an object that evaluates a rule from covered instances.
        By default, weighted relative accuracy is used.
    :type evaluator: :class:`~Orange.classification.rules.RuleEvaluator`
    
    :param beam_width: width of the search beam.
    :type beam_width: int
    
    :param alpha: significance level of the likelihood ratio statistics to
        determine whether rule is better than the default rule.
    :type alpha: float
    
    :param mult: multiplicator for weights of covered instances.
    :type mult: float
    """
    def __new__(cls, instances=None, weight_id=0, **kwargs):
        self = CN2UnorderedLearner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(**kwargs)
            return self.__call__(instances, weight_id)
        else:
            return self

    def __init__(self, evaluator=WRACCEvaluator(), beam_width=5,
                alpha=0.05, mult=0.7, **kwds):
        CN2UnorderedLearner.__init__(self, evaluator=evaluator,
                                          beam_width=beam_width, alpha=alpha, **kwds)
        self.cover_and_remove = CovererAndRemover_MultWeights(mult=mult)

    def __call__(self, instances, weight=0):
        supervisedClassCheck(instances)

        oldInstances = Orange.data.Table(instances)
        classifier = CN2UnorderedLearner.__call__(self, instances, weight)
        for r in classifier.rules:
            r.filterAndStore(oldInstances, weight, r.classifier.default_val)
        return classifier


class ABCN2(RuleLearner):
    """
    This is an implementation of argument-based CN2 using EVC as evaluation
    and LRC classification.
    
    Rule learning parameters that can be passed to constructor:
    
    :param width: beam width (default 5).
    :type width: int
    :param learn_for_class: class for which to learn; None (default) if all
       classes are to be learnt.
    :param learn_one_rule: decides whether to rule one rule only (default
       False).
    :type learn_one_rule: boolean
    :param analyse_argument: index of argument to analyse; -1 to learn normally
       (default)
    :type analyse_argument: int
    :param debug: sets debug mode - prints some info during execution; False (default)
    :type debug: boolean
    
    The following evaluator related arguments are supported:
    
    :param m: m for m-estimate to be corrected with EVC (default 2).
    :type m: int
    :param opt_reduction: type of EVC correction: 0=no correction,
       1=pessimistic, 2=normal (default 2).
    :type opt_reduction: int
    :param nsampling: number of samples in estimating extreme value
       distribution for EVC (default 100).
    :type nsampling: int
    :param evd: pre-given extreme value distributions.
    :param evd_arguments: pre-given extreme value distributions for arguments.
    
    Those parameters control rule validation:
    
    :param rule_sig: minimal rule significance (default 1.0).
    :type rule_sig: float
    :param att_sig: minimal attribute significance in rule (default 1.0).
    :type att_sig: float
    :param max_rule_complexity: maximum number of conditions in rule (default 5).
    :type max_rule_complexity: int
    :param min_coverage: minimal number of covered instances (default 5).
    :type min_coverage: int
    
    Probabilistic covering can be controlled using:
    
    :param min_improved: minimal number of instances improved in probabilistic covering (default 1).
    :type min_improved: int
    :param min_improved_perc: minimal percentage of covered instances that need to be improved (default 0.0).
    :type min_improved_perc: float
    
    Finally, LRC (classifier) related parameters are:
    
    :param add_sub_rules: decides whether to add sub-rules.
    :type add_sub_rules: boolean
    :param min_cl_sig: minimal significance of beta in classifier (default 0.5).
    :type min_cl_sig: float
    :param min_beta: minimal beta value (default 0.0).
    :type min_beta: float
    :param set_prefix_rules: decides whether ordered prefix rules should be
       added (default False).
    :type set_prefix_rules: boolean
    :param alternative_learner: use rule-learner as a correction method for
       other machine learning methods (default None).

    """

    def __init__(self, argument_id=0, width=5, m=2, opt_reduction=2, nsampling=100, max_rule_complexity=5,
                 rule_sig=1.0, att_sig=1.0, postpruning=None, min_quality=0., min_coverage=1, min_improved=1, min_improved_perc=0.0,
                 learn_for_class=None, learn_one_rule=False, evd=None, evd_arguments=None, prune_arguments=False, analyse_argument= -1,
                 alternative_learner=None, min_cl_sig=0.5, min_beta=0.0, set_prefix_rules=False, add_sub_rules=False, debug=False,
                 **kwds):

        # argument ID which is passed to abcn2 learner
        self.argument_id = argument_id
        # learn for specific class only?        
        self.learn_for_class = learn_for_class
        # only analysing a specific argument or learning all at once
        self.analyse_argument = analyse_argument
        # should we learn only one rule?
        self.learn_one_rule = learn_one_rule
        self.postpruning = postpruning
        # rule finder
        self.rule_finder = RuleBeamFinder()
        self.ruleFilter = RuleBeamFilter_Width(width=width)
        self.ruleFilter_arguments = ABBeamFilter(width=width)
        if max_rule_complexity - 1 < 0:
            max_rule_complexity = 10
        self.rule_finder.rule_stoppingValidator = RuleValidator_LRS(alpha=1.0, min_quality=0., max_rule_complexity=max_rule_complexity - 1, min_coverage=min_coverage)
        self.refiner = RuleBeamRefiner_Selector()
        self.refiner_arguments = SelectorAdder(discretizer=Orange.feature.discretization.Entropy(forceAttribute=1,
                                                                                           maxNumberOfIntervals=2))
        self.prune_arguments = prune_arguments
        # evc evaluator
        evdGet = Orange.core.EVDistGetter_Standard()
        self.rule_finder.evaluator = RuleEvaluator_mEVC(m=m, evDistGetter=evdGet, min_improved=min_improved, min_improved_perc=min_improved_perc)
        self.rule_finder.evaluator.returnExpectedProb = True
        self.rule_finder.evaluator.optimismReduction = opt_reduction
        self.rule_finder.evaluator.ruleAlpha = rule_sig
        self.rule_finder.evaluator.attributeAlpha = att_sig
        self.rule_finder.evaluator.validator = RuleValidator_LRS(alpha=1.0, min_quality=min_quality, min_coverage=min_coverage, max_rule_complexity=max_rule_complexity - 1)

        # learn stopping criteria
        self.rule_stopping = None
        self.data_stopping = RuleDataStoppingCriteria_NoPositives()
        # evd fitting
        self.evd_creator = EVDFitter(self, n=nsampling)
        self.evd = evd
        self.evd_arguments = evd_arguments
        # classifier
        self.add_sub_rules = add_sub_rules
        self.classifier = PILAR(alternative_learner=alternative_learner, min_cl_sig=min_cl_sig, min_beta=min_beta, set_prefix_rules=set_prefix_rules)
        self.debug = debug
        # arbitrary parameters
        self.__dict__.update(kwds)


    def __call__(self, examples, weight_id=0):
        # initialize progress bar
        progress = getattr(self, "progressCallback", None)
        if progress:
            progress.start = 0.0
            progress.end = 0.0
            distrib = Orange.statistics.distribution.Distribution(
                             examples.domain.class_var, examples, weight_id)
            distrib.normalize()

        # we begin with an empty set of rules
        all_rules = RuleList()

        # th en, iterate through all classes and learn rule for each class separately
        for cl_i, cl in enumerate(examples.domain.class_var):
            if progress:
                step = distrib[cl] / 2.
                progress.start = progress.end
                progress.end += step

            if self.learn_for_class and not self.learn_for_class in [cl, cl_i]:
                continue

            # rules for this class only
            rules = RuleList()

            # create dichotomous class
            dich_data = self.create_dich_class(examples, cl)

            # preparation of the learner (covering, evd, etc.)
            self.prepare_settings(dich_data, weight_id, cl_i, progress)

            # learn argumented rules first ...
            self.turn_ABML_mode(dich_data, weight_id, cl_i)
            # first specialize all unspecialized arguments
            # dich_data = self.specialise_arguments(dich_data, weight_id)
            # comment: specialisation of arguments is within learning of an argumented rule;
            #          this is now different from the published algorithm
            if progress:
                progress.start = progress.end
                progress.end += step

            aes = self.get_argumented_examples(dich_data)
            aes = self.sort_arguments(aes, dich_data)
            while aes:
                if self.analyse_argument > -1 and \
                   (isinstance(self.analyse_argument, Orange.core.Example) and not Orange.core.Example(dich_data.domain, self.analyse_argument) == aes[0] or \
                    isinstance(self.analyse_argument, int) and not dich_data[self.analyse_argument] == aes[0]):
                    aes = aes[1:]
                    continue
                ae = aes[0]
                rule = self.learn_argumented_rule(ae, dich_data, weight_id) # target class is always first class (0)
                if self.debug and rule:
                    print "learned arg rule", Orange.classification.rules.rule_to_string(rule)
                elif self.debug:
                    print "no rule came out of ", ae
                if rule:
                    rules.append(rule)
                    aes = filter(lambda x: not rule(x), aes)
                else:
                    aes = aes[1:]
                aes = aes[1:]

            if not progress and self.debug:
                print " arguments finished ... "

            # remove all examples covered by rules
            for rule in rules:
                dich_data = self.remove_covered_examples(rule, dich_data, weight_id)
            if progress:
                progress(self.remaining_probability(dich_data), None)

            # learn normal rules on remaining examples
            if self.analyse_argument == -1:
                self.turn_normal_mode(dich_data, weight_id, cl_i)
                while dich_data:
                    # learn a rule
                    rule = self.learn_normal_rule(dich_data, weight_id, self.apriori)
                    if not rule:
                        break
                    if self.debug:
                        print "rule learned: ", Orange.classification.rules.rule_to_string(rule), rule.quality
                    dich_data = self.remove_covered_examples(rule, dich_data, weight_id)
                    if progress:
                        progress(self.remaining_probability(dich_data), None)
                    rules.append(rule)
                    if self.learn_one_rule:
                        break

            # prune unnecessary rules
            rules = self.prune_unnecessary_rules(rules, dich_data, weight_id)

            if self.add_sub_rules:
                rules = self.add_sub_rules_call(rules, dich_data, weight_id)

            # restore domain and class in rules, add them to all_rules
            for r in rules:
                all_rules.append(self.change_domain(r, cl, examples, weight_id))

            if progress:
                progress(1.0, None)
        # create a classifier from all rules        
        return self.create_classifier(all_rules, examples, weight_id)

    def learn_argumented_rule(self, ae, examples, weight_id):
        # prepare roots of rules from arguments
        positive_args = self.init_pos_args(ae, examples, weight_id)
        if not positive_args: # something wrong
            raise "There is a problem with argumented example %s" % str(ae)
            return None
        negative_args = self.init_neg_args(ae, examples, weight_id)

        # set negative arguments in refiner
        self.rule_finder.refiner.notAllowedSelectors = negative_args
        self.rule_finder.refiner.example = ae
        # set arguments to filter
        self.rule_finder.ruleFilter.setArguments(examples.domain, positive_args)

        # learn a rule
        self.rule_finder.evaluator.bestRule = None
        self.rule_finder(examples, weight_id, 0, positive_args)

        # return best rule
        return self.rule_finder.evaluator.bestRule

    def prepare_settings(self, examples, weight_id, cl_i, progress):
        # apriori distribution
        self.apriori = Orange.statistics.distribution.Distribution(
                                examples.domain.class_var, examples, weight_id)

        # prepare covering mechanism
        self.coverAndRemove = CovererAndRemover_Prob(examples, weight_id, 0, self.apriori, self.argument_id)
        self.rule_finder.evaluator.probVar = examples.domain.getmeta(self.cover_and_remove.probAttribute)

        # compute extreme distributions
        # TODO: why evd and evd_this????
        if self.rule_finder.evaluator.optimismReduction > 0 and not self.evd:
            self.evd_this = self.evd_creator.computeEVD(examples, weight_id, target_class=0, progress=progress)
        if self.evd:
            self.evd_this = self.evd[cl_i]

    def turn_ABML_mode(self, examples, weight_id, cl_i):
        # evaluator
        if self.rule_finder.evaluator.optimismReduction > 0 and self.argument_id:
            if self.evd_arguments:
                self.rule_finder.evaluator.evDistGetter.dists = self.evd_arguments[cl_i]
            else:
                self.rule_finder.evaluator.evDistGetter.dists = self.evd_this # self.evd_creator.computeEVD_example(examples, weight_id, target_class=0)
        # rule refiner
        self.rule_finder.refiner = self.refiner_arguments
        self.rule_finder.refiner.argument_id = self.argument_id
        self.rule_finder.ruleFilter = self.ruleFilter_arguments

    def create_dich_class(self, examples, cl):
        """
        Create dichotomous class.
        """
        (newDomain, targetVal) = create_dichotomous_class(examples.domain, examples.domain.class_var, str(cl), negate=0)
        newDomainmetas = newDomain.getmetas()
        newDomain.addmeta(Orange.feature.new_meta_id(), examples.domain.class_var) # old class as meta
        dichData = examples.select(newDomain)
        if self.argument_id:
            for d in dichData: # remove arguments given to other classes
                if not d.getclass() == targetVal:
                    d[self.argument_id] = "?"
        return dichData

    def get_argumented_examples(self, examples):
        if not self.argument_id:
            return None

        # get argumented examples
        return ArgumentFilter_hasSpecial()(examples, self.argument_id, target_class=0)

    def sort_arguments(self, arg_examples, examples):
        if not self.argument_id:
            return None
        evaluateAndSortArguments(examples, self.argument_id)
        if len(arg_examples) > 0:
            # sort examples by their arguments quality (using first argument as it has already been sorted)
            sorted = arg_examples.native()
            sorted.sort(lambda x, y:-cmp(x[self.argument_id].value.positive_arguments[0].quality,
                                         y[self.argument_id].value.positive_arguments[0].quality))
            return Orange.data.Table(examples.domain, sorted)
        else:
            return None

    def turn_normal_mode(self, examples, weight_id, cl_i):
        # evaluator
        if self.rule_finder.evaluator.optimismReduction > 0:
            if self.evd:
                self.rule_finder.evaluator.evDistGetter.dists = self.evd[cl_i]
            else:
                self.rule_finder.evaluator.evDistGetter.dists = self.evd_this # self.evd_creator.computeEVD(examples, weight_id, target_class=0)
        # rule refiner
        self.rule_finder.refiner = self.refiner
        self.rule_finder.ruleFilter = self.ruleFilter

    def learn_normal_rule(self, examples, weight_id, apriori):
        if hasattr(self.rule_finder.evaluator, "bestRule"):
            self.rule_finder.evaluator.bestRule = None
        rule = self.rule_finder(examples, weight_id, 0, RuleList())
        if hasattr(self.rule_finder.evaluator, "bestRule") and self.rule_finder.evaluator.returnExpectedProb:
            rule = self.rule_finder.evaluator.bestRule
            self.rule_finder.evaluator.bestRule = None
        if self.postpruning:
            rule = self.postpruning(rule, examples, weight_id, 0, aprior)
        return rule

    def remove_covered_examples(self, rule, examples, weight_id):
        nexamples, nweight = self.cover_and_remove(rule, examples, weight_id, 0)
        return nexamples


    def prune_unnecessary_rules(self, rules, examples, weight_id):
        return self.cover_and_remove.getBestRules(rules, examples, weight_id)

    def change_domain(self, rule, cl, examples, weight_id):
        rule.filter = Orange.core.Filter_values(domain=examples.domain,
                                        conditions=rule.filter.conditions)
        rule.filterAndStore(examples, weight_id, cl)
        if hasattr(rule, "learner") and hasattr(rule.learner, "arg_example"):
            rule.learner.arg_example = Orange.data.Instance(examples.domain, rule.learner.arg_example)
        return rule

    def create_classifier(self, rules, examples, weight_id):
        return self.classifier(rules, examples, weight_id)

    def add_sub_rules_call(self, rules, examples, weight_id):
        apriori = Orange.statistics.distribution.Distribution(
                            examples.domain.class_var, examples, weight_id)
        new_rules = RuleList()
        for r in rules:
            new_rules.append(r)

        # loop through rules
        for r in rules:
            tmpList = RuleList()
            tmpRle = r.clone()
            tmpRle.filter.conditions = r.filter.conditions[:r.requiredConditions] # do not split argument
            tmpRle.parentRule = None
            tmpRle.filterAndStore(examples, weight_id, r.classifier.default_val)
            tmpRle.complexity = 0
            tmpList.append(tmpRle)
            while tmpList and len(tmpList[0].filter.conditions) <= len(r.filter.conditions):
                tmpList2 = RuleList()
                for tmpRule in tmpList:
                    # evaluate tmpRule
                    oldREP = self.rule_finder.evaluator.returnExpectedProb
                    self.rule_finder.evaluator.returnExpectedProb = False
                    tmpRule.quality = self.rule_finder.evaluator(tmpRule, examples, weight_id, r.classifier.default_val, apriori)
                    self.rule_finder.evaluator.returnExpectedProb = oldREP
                tmpList.sort(lambda x, y:-cmp(x.quality, y.quality))
                tmpList = tmpList[:self.rule_filter.width]

                for tmpRule in tmpList:
                    # if rule not in rules already, add it to the list
                    if not True in [Orange.classification.rules.rules_equal(ri, tmpRule) for ri in new_rules] and len(tmpRule.filter.conditions) > 0 and tmpRule.quality > apriori[r.classifier.default_val] / apriori.abs:
                        new_rules.append(tmpRule)
                    # create new tmpRules, set parent Rule, append them to tmpList2
                    if not True in [Orange.classification.rules.rules_equal(ri, tmpRule) for ri in new_rules]:
                        for c in r.filter.conditions:
                            tmpRule2 = tmpRule.clone()
                            tmpRule2.parentRule = tmpRule
                            tmpRule2.filter.conditions.append(c)
                            tmpRule2.filterAndStore(examples, weight_id, r.classifier.default_val)
                            tmpRule2.complexity += 1
                            if tmpRule2.class_distribution.abs < tmprule.class_distribution.abs:
                                tmpList2.append(tmpRule2)
                tmpList = tmpList2
        return new_rules

    def init_pos_args(self, ae, examples, weight_id):
        pos_args = RuleList()
        # prepare arguments
        for p in ae[self.argument_id].value.positive_arguments:
            new_arg = Rule(filter=ArgFilter(argument_id=self.argument_id,
                                                   filter=self.newFilter_values(p.filter),
                                                   arg_example=ae),
                                                   complexity=0)
            new_arg.valuesFilter = new_arg.filter.filter
            pos_args.append(new_arg)


        if hasattr(self.rule_finder.evaluator, "returnExpectedProb"):
            old_exp = self.rule_finder.evaluator.returnExpectedProb
            self.rule_finder.evaluator.returnExpectedProb = False

        # argument pruning (all or just unfinished arguments)
        # if pruning is chosen, then prune arguments if possible
        for p in pos_args:
            p.filterAndStore(examples, weight_id, 0)
            if not p.learner:
                p.learner = DefaultLearner(default_value=ae.getclass())
            # pruning on: we check on all conditions and take only best
            if self.prune_arguments:
                allowed_conditions = [c for c in p.filter.conditions]
                pruned_conditions = self.prune_arg_conditions(ae, allowed_conditions, examples, weight_id)
                p.baseDist = Orange.statistics.distribution.Distribution(examples.domain.classVar, examples, weight_id)
                p.filter.conditions = pruned_conditions
                p.learner.setattr("arg_length", 0)

            else: # prune only unspecified conditions
                spec_conditions = [c for c in p.filter.conditions if not c.unspecialized_condition]
                unspec_conditions = [c for c in p.filter.conditions if c.unspecialized_condition]
                # let rule cover now all examples filtered by specified conditions
                p.filter.conditions = spec_conditions
                p.filterAndStore(examples, weight_id, 0)
                p.baseDist = p.classDistribution
                p.learner.setattr("arg_length", len(p.filter.conditions))
                pruned_conditions = self.prune_arg_conditions(ae, unspec_conditions, p.examples, p.weightID)
                p.filter.conditions.extend(pruned_conditions)
                p.filter.filter.conditions.extend(pruned_conditions)
                # if argument does not contain all unspecialized reasons, add those reasons with minimum values
                at_oper_pairs = [(c.position, c.oper) for c in p.filter.conditions if type(c) == Orange.core.ValueFilter_continuous]
                for u in unspec_conditions:
                    if not (u.position, u.oper) in at_oper_pairs:
                        # find minimum value
                        if u.oper == Orange.core.ValueFilter_continuous.Greater or u.oper == Orange.core.ValueFilter_continuous.GreaterEqual:
                            u.ref = min([float(e[u.position]) - 10. for e in p.examples])
                        else:
                            u.ref = max([float(e[u.position]) + 10. for e in p.examples])
                        p.filter.conditions.append(u)
                        p.filter.filter.conditions.append(u)

        # set parameters to arguments
        for p_i, p in enumerate(pos_args):
            p.filterAndStore(examples, weight_id, 0)
            p.filter.domain = examples.domain
            p.classifier = p.learner(p.examples, p.weightID)
            p.requiredConditions = len(p.filter.conditions)
            p.learner.setattr("arg_example", ae)
            p.complexity = len(p.filter.conditions)

        if hasattr(self.rule_finder.evaluator, "returnExpectedProb"):
            self.rule_finder.evaluator.returnExpectedProb = old_exp

        return pos_args

    def newFilter_values(self, filter):
        newFilter = Orange.core.Filter_values()
        newFilter.conditions = filter.conditions[:]
        newFilter.domain = filter.domain
        newFilter.negate = filter.negate
        newFilter.conjunction = filter.conjunction
        return newFilter

    def init_neg_args(self, ae, examples, weight_id):
        return ae[self.argument_id].value.negative_arguments

    def remaining_probability(self, examples):
        return self.cover_and_remove.covered_percentage(examples)

    def prune_arg_conditions(self, crit_example, allowed_conditions, examples, weight_id):
        if not allowed_conditions:
            return []
        cn2_learner = Orange.classification.rules.CN2UnorderedLearner()
        cn2_learner.rule_finder = RuleBeamFinder()
        cn2_learner.rule_finder.refiner = SelectorArgConditions(crit_example, allowed_conditions)
        cn2_learner.rule_finder.evaluator = Orange.classification.rules.MEstimateEvaluator(self.rule_finder.evaluator.m)
        rule = cn2_learner.rule_finder(examples, weight_id, 0, RuleList())
        return rule.filter.conditions

ABCN2 = deprecated_members({"beamWidth": "beam_width",
                     "ruleFinder": "rule_finder",
                     "ruleStopping": "rule_stopping",
                     "dataStopping": "data_stopping",
                     "coverAndRemove": "cover_and_remove",
                     "storeInstances": "store_instances",
                     "targetClass": "target_class",
                     "baseRules": "base_rules",
                     "weightID": "weight_id",
                     "argumentID": "argument_id"})(ABCN2)

class CN2EVCUnorderedLearner(ABCN2):
    """
    CN2-SD (see Lavrac et al.; 2004) induces a set of unordered rules in a
    simmilar manner as
    :class:`~Orange.classification.rules.CN2SDUnorderedLearner`. This
    implementation uses the EVC rule evaluation.
    
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.

    :param evaluator: an object that evaluates a rule from covered instances.
        By default, weighted relative accuracy is used.
    :type evaluator: :class:`~Orange.classification.rules.RuleEvaluator`
    
    :param beam_width: width of the search beam.
    :type beam_width: int
    
    :param alpha: significance level of the likelihood ratio statistics to
        determine whether rule is better than the default rule.
    :type alpha: float
    
    :param mult: multiplicator for weights of covered instances.
    :type mult: float
    """
    def __init__(self, width=5, nsampling=100, rule_sig=1.0, att_sig=1.0, \
        min_coverage=1., max_rule_complexity=5.):
        ABCN2.__init__(self, width=width, nsampling=nsampling,
            rule_sig=rule_sig, att_sig=att_sig, min_coverage=int(min_coverage),
            max_rule_complexity=int(max_rule_complexity))

class DefaultLearner(Orange.core.Learner):
    """
    Default lerner - returns default classifier with predefined output class.
    """
    def __init__(self, default_value=None):
        self.default_value = default_value
    def __call__(self, examples, weight_id=0):
        return Orange.classification.majority.ConstantClassifier(self.default_value, defaultDistribution=Orange.core.Distribution(examples.domain.class_var, examples, weight_id))

class ABCN2Ordered(ABCN2):
    """
    Rules learned by ABCN2 are ordered and used as a decision list.
    """
    def __init__(self, argument_id=0, **kwds):
        ABCN2.__init__(self, argument_id=argument_id, **kwds)
        self.classifier.set_prefix_rules = True
        self.classifier.optimize_betas = False

class ABCN2M(ABCN2):
    """
    Argument based rule learning with m-estimate as evaluation function.
    """
    def __init__(self, argument_id=0, **kwds):
        ABCN2.__init__(self, argument_id=argument_id, **kwds)
        self.opt_reduction = 0
        self.rule_finder.evaluator.optimismReduction = self.opt_reduction
        self.classifier = CN2UnorderedClassifier

class ABCN2MLRC(ABCN2):
    """
    Argument based rule learning with m-estimate as evaluation function. LRC is used as a classification method.
    """
    def __init__(self, argument_id=0, **kwds):
        ABCN2.__init__(self, argument_id=argument_id, **kwds)
        self.opt_reduction = 0
        self.rule_finder.evaluator.optimismReduction = self.opt_reduction

class ABCN2_StandardClassification(ABCN2):
    """
    Argument based rule learning with the original classification technique.
    """
    def __init__(self, argument_id=0, **kwds):
        ABCN2.__init__(self, argument_id=argument_id, **kwds)
        self.classifier = CN2UnorderedClassifier


class RuleStopping_Apriori(RuleStoppingCriteria):
    def __init__(self, apriori=None):
        self.apriori = None

    def __call__(self, rules, rule, instances, data):
        if not self.apriori:
            return False
        if not type(rule.classifier) == Orange.classification.ConstantClassifier:
            return False
        ruleAcc = rule.class_distribution[rule.classifier.default_val] / rule.class_distribution.abs
        aprioriAcc = self.apriori[rule.classifier.default_val] / self.apriori.abs
        if ruleAcc > aprioriAcc:
            return False
        return True


class RuleStopping_SetRules(RuleStoppingCriteria):
    def __init__(self, validator):
        self.rule_stopping = RuleStoppingCriteria_NegativeDistribution()
        self.validator = validator

    def __call__(self, rules, rule, instances, data):
        ru_st = self.rule_stopping(rules, rule, instances, data)
        if not ru_st:
            self.validator.rules.append(rule)
        return bool(ru_st)


class LengthValidator(RuleValidator):
    """ prune rules with more conditions than self.length. """
    def __init__(self, length= -1):
        self.length = length

    def __call__(self, rule, data, weight_id, target_class, apriori):
        if self.length >= 0:
            return len(rule.filter.conditions) <= self.length
        return True


class NoDuplicatesValidator(RuleValidator):
    def __init__(self, alpha=.05, min_coverage=0, max_rule_length=0, rules=RuleList()):
        self.rules = rules
        self.validator = RuleValidator_LRS(alpha=alpha, \
            min_coverage=min_coverage, max_rule_length=max_rule_length)

    def __call__(self, rule, data, weight_id, target_class, apriori):
        if rule_in_set(rule, self.rules):
            return False
        return bool(self.validator(rule, data, weight_id, target_class, apriori))



class RuleClassifier_BestRule(RuleClassifier):
    def __init__(self, rules, instances, weight_id=0, **argkw):
        self.rules = rules
        self.examples = instances
        self.class_var = instances.domain.class_var
        self.__dict__.update(argkw)
        self.prior = Orange.statistics.distribution.Distribution(
                    instances.domain.class_var, instances)

    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue):
        retDist = Orange.statistics.distribution.Distribution(instance.domain.class_var)
        bestRule = None
        for r in self.rules:
            if r(instance) and (not bestRule or r.quality > bestRule.quality):
                for v_i, v in enumerate(instance.domain.class_var):
                    retDist[v_i] = r.class_distribution[v_i]
                bestRule = r
        if not bestRule:
            retDist = self.prior
        else:
            bestRule.used += 1
        sumdist = sum(retDist)
        if sumdist > 0.0:
            for c in self.examples.domain.class_var:
                retDist[c] /= sumdisc
        else:
            retDist.normalize()
        # return classifier(instance, result_type=result_type)
        if result_type == Orange.classification.Classifier.GetValue:
          return retDist.modus()
        if result_type == Orange.classification.Classifier.GetProbabilities:
          return retDist
        return (retDist.modus(), retDist)

    def __str__(self):
        retStr = ""
        for r in self.rules:
            retStr += rule_to_string(r) + " " + str(r.class_distribution) + "\n"
        return retStr


class CovererAndRemover_MultWeights(RuleCovererAndRemover):
    """
    Covering and removing of instances using weight multiplication:
    
    :param mult: weighting multiplication factor
    :type mult: float    
    """

    def __init__(self, mult=0.7):
        self.mult = mult
    def __call__(self, rule, instances, weights, target_class):
        if not weights:
            weights = Orange.feature.new_meta_id()
            instances.addMetaAttribute(weights, 1.)
            instances.domain.addmeta(weights, Orange.feature.\
                Continuous("weights-" + str(weights)), True)
        newWeightsID = Orange.feature.new_meta_id()
        instances.addMetaAttribute(newWeightsID, 1.)
        instances.domain.addmeta(newWeightsID, Orange.feature.\
            Continuous("weights-" + str(newWeightsID)), True)
        for instance in instances:
            if rule(instance) and instance.getclass() == rule.classifier(\
                instance, Orange.classification.Classifier.GetValue):
                instance[newWeightsID] = instance[weights] * self.mult
            else:
                instance[newWeightsID] = instance[weights]
        return (instances, newWeightsID)


class CovererAndRemover_AddWeights(RuleCovererAndRemover):
    """
    Covering and removing of instances using weight addition.
    
    """

    def __call__(self, rule, instances, weights, target_class):
        if not weights:
            weights = Orange.feature.new_meta_id()
            instances.addMetaAttribute(weights, 1.)
            instances.domain.addmeta(weights, Orange.feature.\
                Continuous("weights-" + str(weights)), True)
        try:
            coverage = instances.domain.getmeta("Coverage")
        except:
            coverage = Orange.feature.Continuous("Coverage")
            instances.domain.addmeta(Orange.feature.new_meta_id(), coverage, True)
            instances.addMetaAttribute(coverage, 0.0)
        newWeightsID = Orange.feature.new_meta_id()
        instances.addMetaAttribute(newWeightsID, 1.)
        instances.domain.addmeta(newWeightsID, Orange.feature.\
            Continuous("weights-" + str(newWeightsID)), True)
        for instance in instances:
            if rule(instance) and instance.getclass() == rule.classifier(instance, \
                    Orange.classification.Classifier.GetValue):
                try:
                    instance[coverage] += 1.0
                except:
                    instance[coverage] = 1.0
                instance[newWeightsID] = 1.0 / (instance[coverage] + 1)
            else:
                instance[newWeightsID] = instance[weights]
        return (instances, newWeightsID)


class CovererAndRemover_Prob(RuleCovererAndRemover):
    """ This class impements probabilistic covering. """
    def __init__(self, examples, weight_id, target_class, apriori, argument_id):
        self.best_rule = [None] * len(examples)
        self.prob_attribute = Orange.feature.new_meta_id()
        self.apriori_prob = apriori[target_class] / apriori.abs
        examples.addMetaAttribute(self.prob_attribute, self.apriori_prob)
        examples.domain.addmeta(self.prob_attribute,
            Orange.feature.Continuous("Probs"))
        self.argument_id = argument_id

    def getBestRules(self, current_rules, examples, weight_id):
        best_rules = RuleList()
        for r_i, r in enumerate(self.best_rule):
            if r and not rule_in_set(r, best_rules) and int(examples[r_i].getclass()) == int(r.classifier.default_value):
                if hasattr(r.learner, "arg_example"):
                    setattr(r, "best_example", r.learner.arg_example)
                else:
                    setattr(r, "best_example", examples[r_i])
                best_rules.append(r)
        return best_rules

    def __call__(self, rule, examples, weights, target_class):
        """ if example has an argument, then the rule must be consistent with the argument. """
        example = getattr(rule.learner, "arg_example", None)
        for ei, e in enumerate(examples):
            if e == example:
                e[self.prob_attribute] = 1.0
                self.best_rule[ei] = rule
            elif rule(e) and rule.quality > e[self.prob_attribute]:
                e[self.prob_attribute] = rule.quality + 0.001 # 0.001 is added to avoid numerical errors
                self.best_rule[ei] = rule
        return (examples, weights)

    def filter_covers_example(self, example, filter):
        filter_indices = RuleCoversArguments.filterIndices(filter)
        if filter(example):
            try:
                if example[self.argument_id].value and len(example[self.argument_id].value.positive_arguments) > 0: # example has positive arguments
                    # conditions should cover at least one of the positive arguments
                    one_arg_covered = False
                    for pA in example[self.argument_id].value.positive_arguments:
                        arg_covered = [self.condIn(c, filter_indices) for c in pA.filter.conditions]
                        one_arg_covered = one_arg_covered or len(arg_covered) == sum(arg_covered) #arg_covered
                        if one_arg_covered:
                            break
                    if not one_arg_covered:
                        return False
                if example[self.argument_id].value and len(example[self.argument_id].value.negative_arguments) > 0: # example has negative arguments
                    # condition should not cover neither of negative arguments
                    for pN in example[self.argument_id].value.negative_arguments:
                        arg_covered = [self.condIn(c, filter_indices) for c in pN.filter.conditions]
                        if len(arg_covered) == sum(arg_covered):
                            return False
            except:
                return True
            return True
        return False

    def condIn(self, cond, filter_indices): # is condition in the filter?
        condInd = RuleCoversArguments.conditionIndex(cond)
        if operator.or_(condInd, filter_indices[cond.position]) == filter_indices[cond.position]:
            return True
        return False


    def covered_percentage(self, examples):
        p = 0.0
        for ei, e in enumerate(examples):
            p += (e[self.prob_attribute] - self.apriori_prob) / (1.0 - self.apriori_prob)
        return p / len(examples)




@deprecated_keywords({"showDistribution": "show_distribution"})
def rule_to_string(rule, show_distribution=True):
    """
    Write a string presentation of rule in human readable format.
    
    :param rule: rule to pretty-print.
    :type rule: :class:`~Orange.classification.rules.Rule`
    
    :param show_distribution: determines whether presentation should also
        contain the distribution of covered instances
    :type show_distribution: bool
    
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
    if len(conds) == 0:
        ret = ret + "TRUE"

    for i, c in enumerate(conds):
        if i > 0:
            ret += " AND "
        if type(c) == Orange.core.ValueFilter_discrete:
            ret += domain[c.position].name + "=" + str([domain[c.position].\
                values[int(v)] for v in c.values])
        elif type(c) == Orange.core.ValueFilter_continuous:
            ret += domain[c.position].name + selectSign(c.oper) + str(c.ref)
    if rule.classifier and type(rule.classifier) == Orange.classification.ConstantClassifier\
            and rule.classifier.default_val:
        ret = ret + " THEN " + domain.class_var.name + "=" + \
        str(rule.classifier.default_value)
        if show_distribution:
            ret += str(rule.class_distribution)
    elif rule.classifier and type(rule.classifier) == Orange.classification.ConstantClassifier\
            and type(domain.class_var) == Orange.core.EnumVariable:
        ret = ret + " THEN " + domain.class_var.name + "=" + \
        str(rule.class_distribution.modus())
        if show_distribution:
            ret += str(rule.class_distribution)
    return ret

def supervisedClassCheck(instances):
    if not instances.domain.class_var:
        raise Exception("Class variable is required!")
    if instances.domain.class_var.varType == Orange.core.VarTypes.Continuous:
        raise Exception("CN2 requires a discrete class!")


def rule_in_set(rule, rules):
    for r in rules:
        if rules_equal(rule, r):
            return True
    return False

def rules_equal(rule1, rule2):
    if not len(rule1.filter.conditions) == len(rule2.filter.conditions):
        return False
    for c1 in rule1.filter.conditions:
        found = False # find the same condition in the other rule
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
                found = True
                break
            except:
                pass
        if not found:
            return False
    return True

# Miscellaneous - utility functions
def avg(l):
    if len(l) == 0:
        return 0.
    return sum(l) / len(l)

def var(l):
    if len(l) < 2:
        return 0.
    av = avg(l)
    vars = [math.pow(li - av, 2) for li in l]
    return sum(vars) / (len(l) - 1)

def median(l):
    if len(l) == 0:
        return 0.
    l.sort()
    le = len(l)
    if le % 2 == 1:
        return l[(le - 1) / 2]
    else:
        return (l[le / 2 - 1] + l[le / 2]) / 2

def perc(l, p):
    l.sort()
    return l[int(math.floor(p * len(l)))]

class EVDFitter:
    """ Randomizes a dataset and fits an extreme value distribution onto it. """

    def __init__(self, learner, n=200, randomseed=100):
        self.learner = learner
        self.n = n
        self.randomseed = randomseed
        # initialize random seed to make experiments repeatable
        random.seed(self.randomseed)


    def createRandomDataSet(self, data):
        newData = Orange.core.ExampleTable(data)
        # shuffle data
        cl_num = newData.toNumpy("C")
        random.shuffle(cl_num[0][:, 0])
        clData = Orange.core.ExampleTable(Orange.core.Domain([newData.domain.classVar]), cl_num[0])
        for d_i, d in enumerate(newData):
            d[newData.domain.classVar] = clData[d_i][newData.domain.classVar]
        return newData

    def createEVDistList(self, evdList):
        l = Orange.core.EVDistList()
        for el in evdList:
            l.append(Orange.core.EVDist(mu=el[0], beta=el[1], percentiles=el[2]))
        return l


    # estimated fisher tippett parameters for a set of values given in vals list (+ deciles)
    def compParameters(self, vals, oldMi, oldBeta, oldPercs, fixedBeta=False):
        # compute percentiles
        vals.sort()
        N = len(vals)
        percs = [avg(vals[int(float(N) * i / 10):int(float(N) * (i + 1) / 10)]) for i in range(10)]
        if N < 10:
            return oldMi, oldBeta, percs
        if not fixedBeta:
            beta = min(2.0, math.sqrt(6 * var(vals) / math.pow(math.pi, 2)))#min(2.0, max(oldBeta, math.sqrt(6*var(vals)/math.pow(math.pi,2))))
        else:
            beta = oldBeta
        mi = max(oldMi, percs[-1] + beta * math.log(-math.log(0.95)))
        mi = percs[-1] + beta * math.log(-math.log(0.95))
        return max(oldMi, numpy.average(vals) - beta * 0.5772156649), beta, None

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
        self.learner.ruleFinder.ruleFilter = Orange.core.RuleBeamFilter_Width(width=5)


    def restore_learner(self):
        self.learner.ruleFinder.evaluator = self.evaluator
        self.learner.ruleFinder.ruleStoppingValidator = self.oldStopper
        self.learner.ruleFinder.refiner = self.refiner
        self.learner.ruleFinder.validator = self.validator
        self.learner.ruleFinder.ruleFilter = self.ruleFilter

    def computeEVD(self, data, weightID=0, target_class=0, progress=None):
        import time
        # prepare learned for distribution computation        
        self.prepare_learner()

        # loop through N (sampling repetitions)
        extremeDists = [(0, 1, [])]
        self.learner.ruleFinder.ruleStoppingValidator.max_rule_complexity = self.oldStopper.max_rule_complexity
        maxVals = [[] for l in range(self.oldStopper.max_rule_complexity + 1)]
        for d_i in range(self.n):
            if not progress:
                if self.learner.debug:
                    print d_i,
            else:
                progress(float(d_i) / self.n, None)
            # create data set (remove and randomize)
            a = time.time()
            tempData = self.createRandomDataSet(data)
            a = time.time()
            self.learner.ruleFinder.evaluator.rules = RuleList()
            a = time.time()
            for l in range(self.oldStopper.max_rule_complexity + 2):
               self.learner.ruleFinder.evaluator.rules.append(None)
            a = time.time()
            # Next, learn a rule
            self.learner.ruleFinder(tempData, weightID, target_class, RuleList())
            a = time.time()
            for l in range(self.oldStopper.max_rule_complexity + 1):
                if self.learner.ruleFinder.evaluator.rules[l]:
                    maxVals[l].append(self.learner.ruleFinder.evaluator.rules[l].quality)
                else:
                    maxVals[l].append(0)
##                qs = [r.quality for r in self.learner.ruleFinder.evaluator.rules if r.complexity == l+1]
####                if qs:
####                    for r in self.learner.ruleFinder.evaluator.rules:
####                        if r.quality == max(qs) and r.classDistribution.abs == 16 and r.classDistribution[0] == 16:
####                            print "best rule", orngCN2.ruleToString(r), r.quality
##                if qs:
##                    maxVals[l].append(max(qs))
##                else:
##                    maxVals[l].append(0)
            a = time.time()

        # longer rule should always be better than shorter rule 
        for l in range(self.oldStopper.max_rule_complexity):
            for i in range(len(maxVals[l])):
                if maxVals[l + 1][i] < maxVals[l][i]:
                    maxVals[l + 1][i] = maxVals[l][i]
##        print
##        for mi, m in enumerate(maxVals):
##            print "mi=",mi,m

        mu, beta, perc = 1.0, 2.0, [0.0] * 10
        for mi, m in enumerate(maxVals):
##            if mi == 0:
##                mu, beta, perc = self.compParameters(m, mu, beta, perc)
##            else:
            mu, beta, perc = self.compParameters(m, mu, beta, perc, fixedBeta=True)
            extremeDists.append((mu, beta, perc))
            extremeDists.extend([(0, 1, [])] * (mi))
            if self.learner.debug:
                print mi, mu, beta, perc

        self.restore_learner()
        return self.createEVDistList(extremeDists)

class ABBeamFilter(Orange.core.RuleBeamFilter):
    """
    ABBeamFilter: Filters beam;
        - leaves first N rules (by quality)
        - leaves first N rules that have only of arguments in condition part
    """
    def __init__(self, width=5):
        self.width = width
        self.pArgs = None

    def __call__(self, rulesStar, examples, weight_id):
        newStar = Orange.core.RuleList()
        rulesStar.sort(lambda x, y:-cmp(x.quality, y.quality))
        argsNum = 0
        for r_i, r in enumerate(rulesStar):
            if r_i < self.width: # either is one of best "width" rules
                newStar.append(r)
            elif self.onlyPositives(r):
                if argsNum < self.width:
                    newStar.append(r)
                    argsNum += 1
        return newStar

    def setArguments(self, domain, positive_arguments):
        self.pArgs = positive_arguments
        self.domain = domain
        self.argTab = [0] * len(self.domain.attributes)
        for arg in self.pArgs:
            for cond in arg.filter.conditions:
                self.argTab[cond.position] = 1

    def onlyPositives(self, rule):
        if not self.pArgs:
            return False

        ruleTab = [0] * len(self.domain.attributes)
        for cond in rule.filter.conditions:
            ruleTab[cond.position] = 1
        return map(operator.or_, ruleTab, self.argTab) == self.argTab


class RuleCoversArguments:
    """
    Class determines if rule covers one out of a set of arguments.
    """
    def __init__(self, arguments):
        self.arguments = arguments
        self.indices = []
        for a in self.arguments:
            indNA = getattr(a.filter, "indices", None)
            if not indNA:
                a.filter.setattr("indices", RuleCoversArguments.filterIndices(a.filter))
            self.indices.append(a.filter.indices)

    def __call__(self, rule):
        if not self.indices:
            return False
        if not getattr(rule.filter, "indices", None):
            rule.filter.indices = RuleCoversArguments.filterIndices(rule.filter)
        for index in self.indices:
            if map(operator.or_, rule.filter.indices, index) == rule.filter.indices:
                return True
        return False

    def filterIndices(filter):
        if not filter.domain:
            return []
        ind = [0] * len(filter.domain.attributes)
        for c in filter.conditions:
            ind[c.position] = operator.or_(ind[c.position],
                                         RuleCoversArguments.conditionIndex(c))
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
            if at > -1 and not ind == argIndices[r_i]: # need two changes
                return (-1, 0)
            if not ind == argIndices[r_i]:
                if argIndices[r_i] in [1, 3, 5]:
                    at, type = r_i, argIndices[r_i]
                if argIndices[r_i] == 6:
                    if ind == 3:
                        at, type = r_i, 5
                    if ind == 5:
                        at, type = r_i, 3
        return at, type
    oneSelectorToCover = staticmethod(oneSelectorToCover)


class SelectorAdder(Orange.core.RuleBeamRefiner):
    """
    Selector adder, this function is a refiner function:
       - refined rules are not consistent with any of negative arguments.
    """
    def __init__(self, example=None, not_allowed_selectors=[], argument_id=None,
                 discretizer=Orange.feature.discretization.Entropy(forceAttribute=True)):
        # required values - needed values of attributes
        self.example = example
        self.argument_id = argument_id
        self.not_allowed_selectors = not_allowed_selectors
        self.discretizer = discretizer

    def __call__(self, oldRule, data, weight_id, target_class= -1):
        inNotAllowedSelectors = RuleCoversArguments(self.not_allowed_selectors)
        new_rules = Orange.core.RuleList()

        # get positive indices (selectors already in the rule)
        indices = getattr(oldRule.filter, "indices", None)
        if not indices:
            indices = RuleCoversArguments.filterIndices(oldRule.filter)
            oldRule.filter.setattr("indices", indices)

        # get negative indices (selectors that should not be in the rule)
        negative_indices = [0] * len(data.domain.attributes)
        for nA in self.not_allowed_selectors:
            #print indices, nA.filter.indices
            at_i, type_na = RuleCoversArguments.oneSelectorToCover(indices, nA.filter.indices)
            if at_i > -1:
                negative_indices[at_i] = operator.or_(negative_indices[at_i], type_na)

        #iterate through indices = attributes 
        for i, ind in enumerate(indices):
            if not self.example[i] or self.example[i].isSpecial():
                continue
            if ind == 1:
                continue
            if data.domain[i].varType == Orange.core.VarTypes.Discrete and not negative_indices[i] == 1: # DISCRETE attribute
                if self.example:
                    values = [self.example[i]]
                else:
                    values = data.domain[i].values
                for v in values:
                    tempRule = oldRule.clone()
                    tempRule.filter.conditions.append(Orange.core.ValueFilter_discrete(position=i,
                                                                                  values=[Orange.core.Value(data.domain[i], v)],
                                                                                  acceptSpecial=0))
                    tempRule.complexity += 1
                    tempRule.filter.indices[i] = 1 # 1 stands for discrete attribute (see RuleCoversArguments.conditionIndex)
                    tempRule.filterAndStore(oldRule.examples, oldRule.weightID, target_class)
                    if len(tempRule.examples) < len(oldRule.examples):
                        new_rules.append(tempRule)
            elif data.domain[i].varType == Orange.core.VarTypes.Continuous and not negative_indices[i] == 7: # CONTINUOUS attribute
                try:
                    at = data.domain[i]
                    at_d = self.discretizer(at, oldRule.examples)
                except:
                    continue # discretization failed !
                # If discretization makes sense? then:
                if len(at_d.values) > 1:
                    for p in at_d.getValueFrom.transformer.points:
                        #LESS
                        if not negative_indices[i] == 3:
                            tempRule = self.getTempRule(oldRule, i, Orange.core.ValueFilter_continuous.LessEqual, p, target_class, 3)
                            if len(tempRule.examples) < len(oldRule.examples) and self.example[i] <= p:# and not inNotAllowedSelectors(tempRule):
                                new_rules.append(tempRule)
                        #GREATER
                        if not negative_indices[i] == 5:
                            tempRule = self.getTempRule(oldRule, i, Orange.core.ValueFilter_continuous.Greater, p, target_class, 5)
                            if len(tempRule.examples) < len(oldRule.examples) and self.example[i] > p:# and not inNotAllowedSelectors(tempRule):
                                new_rules.append(tempRule)
        for r in new_rules:
            r.parentRule = oldRule
            r.valuesFilter = r.filter.filter
        return new_rules

    def getTempRule(self, oldRule, pos, oper, ref, target_class, atIndex):
        tempRule = oldRule.clone()

        tempRule.filter.conditions.append(Orange.core.ValueFilter_continuous(position=pos,
                                                                        oper=oper,
                                                                        ref=ref,
                                                                        acceptSpecial=0))
        tempRule.complexity += 1
        tempRule.filter.indices[pos] = operator.or_(tempRule.filter.indices[pos], atIndex) # from RuleCoversArguments.conditionIndex
        tempRule.filterAndStore(oldRule.examples, tempRule.weightID, target_class)
        return tempRule

    def setCondition(self, oldRule, target_class, ci, condition):
        tempRule = oldRule.clone()
        tempRule.filter.conditions[ci] = condition
        tempRule.filter.conditions[ci].setattr("specialized", 1)
        tempRule.filterAndStore(oldRule.examples, oldRule.weightID, target_class)
        return tempRule

SelectorAdder = deprecated_members({"notAllowedSelectors": "not_allowed_selectors",
                     "argumentID": "argument_id"})(SelectorAdder)

# This filter is the ugliest code ever! Problem is with Orange, I had some problems with inheriting deepCopy
# I should take another look at it.
class ArgFilter(Orange.core.Filter):
    """ This class implements AB-covering principle. """
    def __init__(self, argument_id=None, filter=Orange.core.Filter_values(), arg_example=None):
        self.filter = filter
        self.indices = getattr(filter, "indices", [])
        if not self.indices and len(filter.conditions) > 0:
            self.indices = RuleCoversArguments.filterIndices(filter)
        self.argument_id = argument_id
        self.domain = self.filter.domain
        self.conditions = filter.conditions
        self.arg_example = arg_example

    def condIn(self, cond): # is condition in the filter?
        condInd = ruleCoversArguments.conditionIndex(cond)
        if operator.or_(condInd, self.indices[cond.position]) == self.indices[cond.position]:
            return True
        return False

    def __call__(self, example):
##        print "in", self.filter(example)#, self.filter.conditions[0](example)
##        print self.filter.conditions[1].values
        if self.filter(example) and example != self.arg_example:
            return True
        elif self.filter(example):
            try:
                if example[self.argument_id].value and len(example[self.argument_id].value.positiveArguments) > 0: # example has positive arguments
                    # conditions should cover at least one of the positive arguments
                    oneArgCovered = False
                    for pA in example[self.argument_id].value.positiveArguments:
                        argCovered = [self.condIn(c) for c in pA.filter.conditions]
                        oneArgCovered = oneArgCovered or len(argCovered) == sum(argCovered) #argCovered
                        if oneArgCovered:
                            break
                    if not oneArgCovered:
                        return False
                if example[self.argument_id].value and len(example[self.argument_id].value.negativeArguments) > 0: # example has negative arguments
                    # condition should not cover neither of negative arguments
                    for pN in example[self.argument_id].value.negativeArguments:
                        argCovered = [self.condIn(c) for c in pN.filter.conditions]
                        if len(argCovered) == sum(argCovered):
                            return False
            except:
                return True
            return True
        else:
            return False

    def __setattr__(self, name, obj):
        self.__dict__[name] = obj
        self.filter.setattr(name, obj)

    def deep_copy(self):
        newFilter = ArgFilter(argument_id=self.argument_id)
        newFilter.filter = Orange.core.Filter_values() #self.filter.deepCopy()
        newFilter.filter.conditions = self.filter.conditions[:]
        newFilter.domain = self.filter.domain
        newFilter.negate = self.filter.negate
        newFilter.conjunction = self.filter.conjunction
        newFilter.domain = self.filter.domain
        newFilter.conditions = newFilter.filter.conditions
        newFilter.indices = self.indices[:]
        return newFilter

ArgFilter = deprecated_members({"argumentID": "argument_id"})(ArgFilter)

class SelectorArgConditions(Orange.core.RuleBeamRefiner):
    """
    Selector adder, this function is a refiner function:
      - refined rules are not consistent with any of negative arguments.
    """
    def __init__(self, example, allowed_selectors):
        # required values - needed values of attributes
        self.example = example
        self.allowed_selectors = allowed_selectors

    def __call__(self, oldRule, data, weight_id, target_class= -1):
        if len(oldRule.filter.conditions) >= len(self.allowed_selectors):
            return Orange.core.RuleList()
        new_rules = Orange.core.RuleList()
        for c in self.allowed_selectors:
            # normal condition
            if not c.unspecialized_condition:
                tempRule = oldRule.clone()
                tempRule.filter.conditions.append(c)
                tempRule.filterAndStore(oldRule.examples, oldRule.weightID, target_class)
                if len(tempRule.examples) < len(oldRule.examples):
                    new_rules.append(tempRule)
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
                        tempRule.filterAndStore(oldRule.examples, oldRule.weightID, target_class)
                        if len(tempRule.examples) < len(oldRule.examples):
                            new_rules.append(tempRule)
##        print " NEW RULES "
##        for r in new_rules:
##            print Orange.classification.rules.rule_to_string(r)
        for r in new_rules:
            r.parentRule = oldRule
##            print Orange.classification.rules.rule_to_string(r)
        return new_rules


class CrossValidation:
    def __init__(self, folds=5, random_generator=150):
        self.folds = folds
        self.random_generator = random_generator

    def __call__(self, learner, examples, weight):
        res = orngTest.crossValidation([learner], (examples, weight), folds=self.folds, random_generator=self.random_generator)
        return self.get_prob_from_res(res, examples)

    def get_prob_from_res(self, res, examples):
        prob_dist = Orange.core.DistributionList()
        for tex in res.results:
            d = Orange.core.Distribution(examples.domain.class_var)
            for di in range(len(d)):
                d[di] = tex.probabilities[0][di]
            prob_dist.append(d)
        return prob_dist


class PILAR:
    """
    PILAR (Probabilistic improvement of learning algorithms with rules).
    """
    def __init__(self, alternative_learner=None, min_cl_sig=0.5, min_beta=0.0, set_prefix_rules=False, optimize_betas=True):
        self.alternative_learner = alternative_learner
        self.min_cl_sig = min_cl_sig
        self.min_beta = min_beta
        self.set_prefix_rules = set_prefix_rules
        self.optimize_betas = optimize_betas
        self.selected_evaluation = CrossValidation(folds=5)

    def __call__(self, rules, examples, weight=0):
        rules = self.add_null_rule(rules, examples, weight)
        if self.alternative_learner:
            prob_dist = self.selected_evaluation(self.alternative_learner, examples, weight)
            classifier = self.alternative_learner(examples, weight)
##            prob_dist = Orange.core.DistributionList()
##            for e in examples:
##                prob_dist.append(classifier(e,Orange.core.GetProbabilities))
            cl = Orange.core.RuleClassifier_logit(rules, self.min_cl_sig, self.min_beta, examples, weight, self.set_prefix_rules, self.optimize_betas, classifier, prob_dist)
        else:
            cl = Orange.core.RuleClassifier_logit(rules, self.min_cl_sig, self.min_beta, examples, weight, self.set_prefix_rules, self.optimize_betas)

##        print "result"
        for ri, r in enumerate(cl.rules):
            cl.rules[ri].setattr("beta", cl.ruleBetas[ri])
##            if cl.ruleBetas[ri] > 0:
##                print Orange.classification.rules.rule_to_string(r), r.quality, cl.ruleBetas[ri]
        cl.all_rules = cl.rules
        cl.rules = self.sort_rules(cl.rules)
        cl.ruleBetas = [r.beta for r in cl.rules]
        cl.setattr("data", examples)
        return cl

    def add_null_rule(self, rules, examples, weight):
        for cl in examples.domain.class_var:
            tmpRle = Orange.core.Rule()
            tmpRle.filter = Orange.core.Filter_values(domain=examples.domain)
            tmpRle.parentRule = None
            tmpRle.filterAndStore(examples, weight, int(cl))
            tmpRle.quality = tmpRle.class_distribution[int(cl)] / tmpRle.class_distribution.abs
            rules.append(tmpRle)
        return rules

    def sort_rules(self, rules):
        new_rules = Orange.core.RuleList()
        foundRule = True
        while foundRule:
            foundRule = False
            bestRule = None
            for r in rules:
                if r in new_rules:
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
                if len(r.filter.conditions) == len(bestRule.filter.conditions) and r.beta > bestRule.beta:
                    bestRule = r
                    foundRule = True
                    continue
            if bestRule:
                new_rules.append(bestRule)
        return new_rules

PILAR = deprecated_members({"sortRules": "sort_rules"})(PILAR)


class RuleClassifier_bestRule(Orange.core.RuleClassifier):
    """
    A very simple classifier, it takes the best rule of each class and
    normalizes probabilities.
    """
    def __init__(self, rules, examples, weight_id=0, **argkw):
        self.rules = rules
        self.examples = examples
        self.apriori = Orange.core.Distribution(examples.domain.class_var, examples, weight_id)
        self.apriori_prob = [a / self.apriori.abs for a in self.apriori]
        self.weight_id = weight_id
        self.__dict__.update(argkw)
        self.default_class_index = -1

    @deprecated_keywords({"retRules": "ret_rules"})
    def __call__(self, example, result_type=Orange.classification.Classifier.GetValue, ret_rules=False):
        example = Orange.core.Example(self.examples.domain, example)
        tempDist = Orange.core.Distribution(example.domain.class_var)
        best_rules = [None] * len(example.domain.class_var.values)

        for r in self.rules:
            if r(example) and not self.default_class_index == int(r.classifier.default_val) and \
               (not best_rules[int(r.classifier.default_val)] or r.quality > tempDist[r.classifier.default_val]):
                tempDist[r.classifier.default_val] = r.quality
                best_rules[int(r.classifier.default_val)] = r
        for b in best_rules:
            if b:
                used = getattr(b, "used", 0.0)
                b.setattr("used", used + 1)
        nonCovPriorSum = sum([tempDist[i] == 0. and self.apriori_prob[i] or 0. for i in range(len(self.apriori_prob))])
        if tempDist.abs < 1.:
            residue = 1. - tempDist.abs
            for a_i, a in enumerate(self.apriori_prob):
                if tempDist[a_i] == 0.:
                    tempDist[a_i] = self.apriori_prob[a_i] * residue / nonCovPriorSum
            final_dist = tempDist #Orange.core.Distribution(example.domain.class_var)
        else:
            tempDist.normalize() # prior probability
            tmp_examples = Orange.core.ExampleTable(self.examples)
            for r in best_rules:
                if r:
                    tmp_examples = r.filter(tmp_examples)
            tmpDist = Orange.core.Distribution(tmp_examples.domain.class_var, tmp_examples, self.weight_id)
            tmpDist.normalize()
            probs = [0.] * len(self.examples.domain.class_var.values)
            for i in range(len(self.examples.domain.class_var.values)):
                probs[i] = tmpDist[i] + tempDist[i] * 2
            final_dist = Orange.core.Distribution(self.examples.domain.class_var)
            for cl_i, cl in enumerate(self.examples.domain.class_var):
                final_dist[cl] = probs[cl_i]
            final_dist.normalize()

        if ret_rules: # Do you want to return rules with classification?
            if result_type == Orange.classification.Classifier.GetValue:
              return (final_dist.modus(), best_rules)
            if result_type == Orange.core.GetProbabilities:
              return (final_dist, best_rules)
            return (final_dist.modus(), final_dist, best_rules)
        if result_type == Orange.classification.Classifier.GetValue:
          return final_dist.modus()
        if result_type == Orange.core.GetProbabilities:
          return final_dist
        return (final_dist.modus(), final_dist)

RuleClassifier_bestRule = deprecated_members({"defaultClassIndex": "default_class_index"})(RuleClassifier_bestRule)
