.. py:currentmodule:: Orange.classification.rules

.. index:: rule induction

.. index:: 
   single: classification; rule induction

**************************
Rule induction (``rules``)
**************************

Module ``rules`` implements supervised rule induction algorithms and
rule-based classification methods. Rule induction is based on a
comprehensive framework of components that can be modified or
replaced. For ease of use, the module already provides multiple
variations of `CN2 induction algorithm
<http://www.springerlink.com/content/k6q2v76736w5039r/>`_.

CN2 algorithm
=============

.. index:: 
   single: classification; CN2

The use of rule learning algorithms is consistent with a typical
learner usage in Orange:

:download:`rules-cn2.py <code/rules-cn2.py>`

.. literalinclude:: code/rules-cn2.py
    :lines: 7-

::
    
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
    
.. autoclass:: Orange.classification.rules.CN2Learner(evaluator=Evaluator_Entropy, beam_width=5, alpha=1)
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

.. autoclass:: Orange.classification.rules.CN2UnorderedLearner(evaluator=Evaluator_Laplace(), beam_width=5, alpha=1.0)
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
   
.. autoclass:: Orange.classification.rules.CN2SDUnorderedLearner(evaluator=WRACCEvaluator(), beam_width=5, alpha=0.05, mult=0.7)
   :members:
   :show-inheritance:
   :exclude-members: baseRules, beamWidth, coverAndRemove, dataStopping,
      ruleFinder, ruleStopping, storeInstances, targetClass, weightID
   
.. autoclass:: Orange.classification.rules.CN2EVCUnorderedLearner
   :members:
   :show-inheritance:
   

..
    This part is commented out since
    - there is no documentation on how to provide arguments
    - the whole thing is represent original research work particular to
      a specific project and belongs to an
      extension rather than to the main package

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

    Thismodule has many more undocumented argument-based learning
    related classed; see the source code for reference.

    References
    ----------

    * Bratko, Mozina, Zabkar. `Argument-Based Machine Learning
      <http://www.springerlink.com/content/f41g17t1259006k4/>`_. Lecture Notes in
      Computer Science: vol. 4203/2006, 11-17, 2006.


Rule induction framework
========================

The classes described above are based on a more general framework that
can be fine-tuned to specific needs by replacing individual components.
Here is an example:

part of :download:`rules-customized.py <code/rules-customized.py>`

.. literalinclude:: code/rules-customized.py
    :lines: 7-17

::

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

In the example, we wanted to use a rule evaluator based on the
m-estimate and set ``m`` to 50. The evaluator is a subcomponent of the
:obj:`rule_finder` component. Thus, to be able to set the evaluator,
we first set the :obj:`rule_finder` component, then we added the
desired subcomponent and set its options. All other components, which
are left unspecified, are provided by the learner at the training time
and removed afterwards.

Continuing with the example, assume that we wish to set a different
validation function and a different beam width.

part of :download:`rules-customized.py <code/rules-customized.py>`

.. literalinclude:: code/rules-customized.py
    :lines: 19-23

.. py:class:: Orange.classification.rules.Rule(filter, classifier, lr, dist, ce, w = 0, qu = -1)
   
   Represents a single rule. Constructor arguments correspond to the
   first seven of the attributes (from :obj:`filter` to
   :obj:`quality`) below.
   
   .. attribute:: filter
   
      Rule condition; an instance of
      :class:`Orange.data.filter.Filter`, typically an instance of a
      class derived from :class:`Orange.data.filter.Values`
   
   .. attribute:: classifier
      
      A rule predicts the class by calling an embedded classifier that
      must be an instance of
      :class:`~Orange.classification.Classifier`, typically
      :class:`~Orange.classification.ConstantClassifier`. This
      classifier is called by the rule classifier, such as
      :obj:`RuleClassifier`.
   
   .. attribute:: learner
      
      Learner that is used for constructing a classifier. It must be
      an instance of :class:`~Orange.classification.Learner`,
      typically
      :class:`~Orange.classification.majority.MajorityLearner`.
   
   .. attribute:: class_distribution
      
      Distribution of class in data instances covered by this rule
      (:class:`~Orange.statistics.distribution.Distribution`).
   
   .. attribute:: instances
      
      Data instances covered by this rule (:class:`~Orange.data.Table`).
   
   .. attribute:: weight_id
   
      ID of the weight meta-attribute for the stored data instances
      (``int``).
   
   .. attribute:: quality
      
      Quality of the rule. Rules with higher quality are better
      (``float``).
   
   .. attribute:: complexity
   
      Complexity of the rule (``float``), typically the number of
      selectors (conditions) in the rule. Complexity is used for
      choosing between rules with the same quality; rules with lower
      complexity are preferred.
   
   .. method:: filter_and_store(instances, weight_id=0, target_class=-1)
   
      Filter passed data instances and store them in :obj:`instances`.
      Also, compute :obj:`class_distribution`, set the weight of
      stored examples and create a new classifier using :obj:`learner`.
      
      :param weight_id: ID of the weight meta-attribute.
      :type weight_id: int
      :param target_class: index of target class; -1 for all classes.
      :type target_class: int
   
   .. method:: __call__(instance)
   
      Return ``True`` if the given instance matches the rule condition.
      
      :param instance: data instance
      :type instance: :class:`Orange.data.Instance`
      
   .. method:: __call__(instances, ref=True, negate=False)

      Return a table of instances that match the rule condition.
      
      :param instances: a table of data instances
      :type instances: :class:`Orange.data.Table`
      :param ref: if ``True`` (default), the constructed table contains
          references to the original data instances; if ``False``, the
          data is copied
      :type ref: bool
      :param negate: inverts the selection
      :type negate: bool



.. py:class:: Orange.classification.rules.RuleLearner(store_instances=True, target_class=-1, base_rules=Orange.classification.rules.RuleList())
   
   Bases: :class:`Orange.classification.Learner`
   
   A base rule induction learner. The algorithm follows
   separate-and-conquer strategy, which has its origins in the AQ
   family of algorithms (Fuernkranz J.; Separate-and-Conquer Rule
   Learning, Artificial Intelligence Review 13, 3-54, 1999). Such
   algorithms search for the optimal rule for the current training
   set, remove the covered training instances (`separate`) and repeat
   the process (`conquer`) on the remaining data.
   
   :param store_instances: if ``True`` (default), the induced rules
       contain a table with references to the stored data instances.
   :type store_instances: bool
    
   :param target_class: index of a specific class to learn; if -1
        there is no target class
   :type target_class: int
   
   :param base_rules: An optional list of initial rules for constraining the :obj:`rule_finder`.
   :type base_rules: :class:`~Orange.classification.rules.RuleList`

   The class' functionality is best explained by its ``__call__``
   function.
   
   .. parsed-literal::

      def \_\_call\_\_(self, instances, weight_id=0):
          rule_list = Orange.classification.rules.RuleList()
          all_instances = Orange.data.Table(instances)
          while not self.\ **data_stopping**\ (instances, weight_id, self.target_class):
              new_rule = self.\ **rule_finder**\ (instances, weight_id, self.target_class, self.base_rules)
              if self.\ **rule_stopping**\ (rule_list, new_rule, instances, weight_id):
                  break
              instances, weight_id = self.\ **cover_and_remove**\ (new_rule, instances, weight_id, self.target_class)
              rule_list.append(new_rule)
          return Orange.classification.rules.RuleClassifier_FirstRule(
              rules=rule_list, instances=all_instances)
       
   The customizable components are :obj:`data_stopping`,
   :obj:`rule_finder`, :obj:`cover_and_remove` and :obj:`rule_stopping`
   objects.
   
   .. attribute:: data_stopping
   
      An instance of
      :class:`~Orange.classification.rules.DataStoppingCriteria`
      that determines whether to continue the induction. The default
      component,
      :class:`~Orange.classification.rules.DataStoppingCriteria_NoPositives`
      returns ``True`` if there are no more instances of the target class. 
   
   .. attribute:: rule_finder
      
      An instance of :class:`~Orange.classification.rules.Finder`
      that learns a single rule. Default is
      :class:`~Orange.classification.rules.BeamFinder`.

   .. attribute:: rule_stopping
      
      An instance of
      :class:`~Orange.classification.rules.StoppingCriteria` that
      decides whether to use the induced rule or to discard it and stop
      the induction. If ``None`` (default) all rules are accepted.
       
   .. attribute:: cover_and_remove
       
      An instance of :class:`RuleCovererAndRemover` that removes
      instances covered by the rule and returns remaining
      instances. The default implementation
      (:class:`RuleCovererAndRemover_Default`) removes the instances
      that belong to given target class; if the target is not
      specified (:obj:`target_class` == -1), it removes all covered
      instances.    


Rule finders
------------

.. class:: Orange.classification.rules.Finder

   Base class for rule finders, which learn a single rule from
   instances.
   
   .. method:: __call__(table, weight_id, target_class, base_rules)
   
      Induce a new rule from the given data.
      
      :param table: training data instances
      :type table: :class:`Orange.data.Table`
      
      :param weight_id: ID of the weight meta-attribute
      :type weight_id: int
      
      :param target_class: index of a specific class being learned; -1 for all.
      :type target_class: int 
      
      :param base_rules: A list of initial rules for constraining the search space
      :type base_rules: :class:`~Orange.classification.rules.RuleList`


.. class:: Orange.classification.rules.BeamFinder
   
   Bases: :class:`~Orange.classification.rules.Finder`
   
   Beam search for the best rule. This is the default finder for
   :obj:`RuleLearner`. Pseudo code of the algorithm is as follows.

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
          
   Modifiable components are shown in bold. These are:

   .. attribute:: initializer
   
      An instance of
      :obj:`~Orange.classification.rules.BeamInitializer` that
      is used to construct the initial list of rules. The default,
      :class:`~Orange.classification.rules.BeamInitializer_Default`,
      returns :obj:`base_rules`, or a rule with no conditions if
      :obj:`base_rules` is not set.
   
   .. attribute:: candidate_selector
   
      An instance of
      :class:`~Orange.classification.rules.BeamCandidateSelector`
      used to separate a subset of rules from the current
      :obj:`rules_star` that will be further specialized.  The default
      component, an instance of
      :class:`~Orange.classification.rules.BeamCandidateSelector_TakeAll`,
      selects all rules.
    
   .. attribute:: refiner
   
      An instance of
      :class:`~Orange.classification.rules.BeamRefiner` that is
      used for refining the rules. Refined rule should cover a strict
      subset of instances covered by the given rule. Default component
      (:class:`~Orange.classification.rules.BeamRefiner_Selector`)
      adds a conjunctive selector to selectors present in the rule.
    
   .. attribute:: rule_filter
   
      An instance of
      :class:`~Orange.classification.rules.BeamFilter` that is
      used for filtering rules to trim the search beam. The default
      component,
      :class:`~Orange.classification.rules.BeamFilter_Width`\
      *(m=5)*\, keeps the five best rules.

   .. method:: __call__(data, weight_id, target_class, base_rules)

       Determines the optimal rule to cover the given data instances.

       :param data: data instances.
       :type data: :class:`Orange.data.Table`

       :param weight_id: index of the weight meta-attribute.
       :type weight_id: int

       :param target_class: index of the target class.
       :type target_class: int

       :param base_rules: existing rules
       :type base_rules: :class:`~Orange.classification.rules.RuleList`

Rule evaluators
---------------

.. class:: Orange.classification.rules.Evaluator

   Base class for rule evaluators that evaluate the quality of the
   rule based on the data instances they cover.
   
   .. method:: __call__(rule, instances, weight_id, target_class, prior)
   
      Calculate a (non-negative) rule quality.
      
      :param rule: rule to evaluate
      :type rule: :class:`~Orange.classification.rules.Rule`
      
      :param instances: data instances covered by the rule
      :type instances: :class:`Orange.data.Table`
      
      :param weight_id: index of the weight meta-attribute
      :type weight_id: int
      
      :param target_class: index of target class of this rule
      :type target_class: int
      
      :param prior: prior class distribution
      :type prior: :class:`Orange.statistics.distribution.Distribution`

.. autoclass:: Orange.classification.rules.LaplaceEvaluator
   :members:
   :show-inheritance:
   :exclude-members: targetClass, weightID

.. autoclass:: Orange.classification.rules.WRACCEvaluator
   :members:
   :show-inheritance:
   :exclude-members: targetClass, weightID
   
.. class:: Orange.classification.rules.Evaluator_Entropy

   Bases: :class:`~Orange.classification.rules.Evaluator`
    
.. class:: Orange.classification.rules.Evaluator_LRS

   Bases: :class:`~Orange.classification.rules.Evaluator`

.. class:: Orange.classification.rules.Evaluator_Laplace

   Bases: :class:`~Orange.classification.rules.Evaluator`

.. class:: Orange.classification.rules.Evaluator_mEVC

   Bases: :class:`~Orange.classification.rules.Evaluator`
   
Instance covering and removal
-----------------------------

.. class:: RuleCovererAndRemover

   Base class for rule coverers and removers that, when invoked, remove
   instances covered by the rule and return remaining instances.

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

