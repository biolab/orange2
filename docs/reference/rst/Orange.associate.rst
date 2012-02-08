.. py:currentmodule:: Orange.associate

#######################################################
Association rules and frequent itemsets (``associate``)
#######################################################

Orange provides two algorithms for induction of
`association rules <http://en.wikipedia.org/wiki/Association_rule_learning>`_,
a standard `Apriori algorithm <http://en.wikipedia.org/wiki/Apriori_algorithm>`_ [AgrawalSrikant1994]_ for sparse (basket) data analysis
and a variant of Apriori for attribute-value data sets. Both algorithms also support mining of frequent itemsets.

For example, consider a simple market basket data::

    Bread, Milk
    Bread, Diapers, Beer, Eggs
    Milk, Diapers, Beer, Cola
    Bread, Milk, Diapers, Beer
    Bread, Milk, Diapers, Cola

The following script induces association rules with items that appear in at least 30% of data instances
(transactions):

.. literalinclude:: code/associate-market.py

The code reports on support and confidence first five rules found::

    Supp Conf  Rule
     0.4  1.0  Cola -> Diapers
     0.4  0.5  Diapers -> Cola
     0.4  1.0  Cola -> Diapers Milk
     0.4  1.0  Cola Diapers -> Milk
     0.4  1.0  Cola Milk -> Diapers

In Apriori, association rule induction is two-stage algorithm first finds itemsets that frequently appear in
the data and have sufficient support, and then splits them to rules of sufficient confidence. Function `getItemsets`
reports on itemsets alone and skips rule induction:

.. literalinclude:: code/associate-frequent-itemsets.py

The above script lists frequent itemsets and their support::

    (0.40) Cola
    (0.40) Cola Diapers
    (0.40) Cola Diapers Milk
    (0.40) Cola Milk
    (0.60) Beer

======================================
Association rules induction algorithms
======================================

:class:`AssociationRulesSparseInducer` induces frequent itemsets and association rules from sparse data sets. These
can be either provided in the basket format (see :doc:`Orange.data.formats`) or in an attribute-value format where any
entry in the data table is considered as presence of a feature in the transaction (an item),
and any unknown (empty) entry signifies its absence. :class:`AssociationRulesInducer` works feature-value data,
where am item is a combination of feature and its value (e.g., `astigmatic=yes`).

Sparse (basket) data sets
-------------------------

.. class:: AssociationRulesSparseInducer

    .. attribute:: support

        Minimal support for the rule. Depending on the data set it should be set to sufficiently high value
        to avoid running out of working memory (default: 0.3).

    .. attribute:: confidence

        Minimal confidence for the rule.

    .. attribute:: store_examples

        Store the examples covered by each rule and
        those confirming it.

    .. attribute:: max_item_sets

        The maximal number of itemsets induced. Orange will stop with inference of
        frequent itemsets once this number of itemsets is reached.

    .. method:: __call__(data, weight_id)

        Induce rules from the provided data set.

    .. method:: get_itemsets(data)

        For a given data set, return a list of frequent itemsets. List elements are pairs,
        where the first element includes indices of features in the item set (negative for sparse data) and
        the second element a list of indices supporting the itemset.
        If :obj:`store_examples` is False, the second
        element is None.

To test this rule inducer, we will first create a sparse data sets consisting of list of words in sentences from a brief description
of Spanish Inquisition, given by Palin et al.:

    NOBODY expects the Spanish Inquisition! Our chief weapon is surprise...surprise and fear...fear and surprise.... Our two weapons are fear and surprise...and ruthless efficiency.... Our *three* weapons are fear, surprise, and ruthless efficiency...and an almost fanatical devotion to the Pope.... Our *four*...no... *Amongst* our weapons.... Amongst our weaponry...are such elements as fear, surprise.... I'll come in again.

    NOBODY expects the Spanish Inquisition! Amongst our weaponry are such diverse elements as: fear, surprise, ruthless efficiency, an almost fanatical devotion to the Pope, and nice red uniforms - Oh damn!

After some cleaning (e.g., removal of stopwords and punctuation marks),
our data set looks like (:download:`inquisition.basket <code/inquisition.basket>`):

.. literalinclude:: code/inquisition.basket

The following script induces the association rules:

.. literalinclude:: code/associate-inquisition.py

The induced rules are surprisingly fear-full::

    0.500   1.000   fear -> surprise
    0.500   1.000   surprise -> fear
    0.500   1.000   fear -> surprise our
    0.500   1.000   fear surprise -> our
    0.500   1.000   fear our -> surprise
    0.500   1.000   surprise -> fear our
    0.500   1.000   surprise our -> fear
    0.500   0.714   our -> fear surprise
    0.500   1.000   fear -> our
    0.500   0.714   our -> fear
    0.500   1.000   surprise -> our
    0.500   0.714   our -> surprise

To get only a list of supported item sets, one should call the method
get_itemsets::

    inducer = Orange.associate.AssociationRulesSparseInducer(support = 0.5, store_examples = True)
    itemsets = inducer.get_itemsets(data)

Now itemsets is a list of itemsets along with the examples supporting them
since we set store_examples to True. ::

    >>> itemsets[5]
    ((-11, -7), [1, 2, 3, 6, 9])
    >>> [data.domain[i].name for i in itemsets[5][0]]
    ['surprise', 'our']

The sixth itemset contains features with indices -11 and -7, that is, the
words "surprise" and "our". The examples supporting it are those with
indices 1,2, 3, 6 and 9.

This way of representing the itemsets is memory efficient and faster than using
objects like :obj:`~Orange.feature.Descriptor` and :obj:`~Orange.data.Instance`.

.. _non-sparse-examples:

Feature-value (non-sparse) data sets
------------------------------------

:class:`AssociationRulesInducer` works with non-sparse data.


.. class:: AssociationRulesInducer

    Association rule induction from non-sparse data sets. An item is a feature-value combination. Unknown values in
    the data table are ignored. The algorithm can also be used to search only for classification rules where the
    feature on the right-hand side is the class variable.

    .. attribute:: support

       Minimal support of the induced rule (default: 0.3)

    .. attribute:: confidence

        Minimal confidence of the induced rule.

    .. attribute:: classification_rules

        If True, the classification rules are constructed instead
        of general association rules (default: False).

    .. attribute:: store_examples

        Store the examples covered by each rule and those
        confirming it.

    .. attribute:: max_item_sets

        The maximal number of itemsets induced. After reaching this limit the inference algorithm will stop.

    .. method:: __call__(data, weight_id)

        Induce rules from the given data set.

    .. method:: get_itemsets(data)

        For a given data set, return a list of frequent itemsets. The list consists of pairs, where
        the first element includes indices of features in the item set (negative for sparse data), and
        the second element a list of indices supporting the item set. If :obj:`store_examples` is False, the second
        element is None.

Following is an example script that uses :class:`AssociationRulesInducer`:

.. literalinclude:: code/associate-lenses.py

Script reports the following rules (first colon is support, second confidence)::

    0.333  0.533  lenses=none -> prescription=hypermetrope
    0.333  0.667  prescription=hypermetrope -> lenses=none
    0.333  0.533  lenses=none -> astigmatic=yes
    0.333  0.667  astigmatic=yes -> lenses=none
    0.500  0.800  lenses=none -> tear_rate=reduced
    0.500  1.000  tear_rate=reduced -> lenses=none

To infer classification rules we can use a similar script but set `classificationRules` to 1:

.. literalinclude:: code/associate-lenses-classification.py
    :lines: 4-5

These rules are a subset of association rules that in a consequent include only a class variable::

    0.333  0.667  prescription=hypermetrope -> lenses=none
    0.333  0.667  astigmatic=yes -> lenses=none
    0.500  1.000  tear_rate=reduced -> lenses=none

Frequent itemsets are induced in a similar fashion as for sparse data, except that the
first element of the tuple, the item set, is represented not by indices of
features, as before, but with tuples (feature-index, value-index):

.. literalinclude:: code/associate-lenses-itemsets.py
    :lines: 4-6

The script prints out::

    (((2, 1), (4, 0)), [2, 6, 10, 14, 15, 18, 22, 23])

reporting that the ninth itemset contains the second value of the third feature
(2, 1), and the first value of the fifth (4, 0).

=======================
Representation of rules
=======================

Methods for induction of association rules return the induced rules in
:class:`AssociationRules`, which is basically a list of :class:`AssociationRule` instances.

.. class:: AssociationRule

    .. method:: __init__(left, right, n_applies_left, n_applies_right, n_applies_both, n_examples)

        Construct an association rule and compute evaluation scores (see below) based on counts given in the
        arguments of the call.

    .. method:: __init__(left, right, support, confidence)

        Construct association rule and compute its support and confidence. For manual construction of such such a rule set other attributes
        manually, as AssociationRules's constructor cannot compute anything only from support and
        confidence.

    .. method:: __init__(rule)

        Given an association rule as the argument, constructor a copy of the rule.

    .. attribute:: left, right

        The left and the right side of the rule. Both are given as :class:`Orange.data.Instance`.
        In rules created by :class:`AssociationRulesSparseInducer` from data instances that
        contain all values as meta-values, left and right are data instances in the
        same form. Otherwise, values in left that do not appear in the rule
        are "don't care", and value in right are "don't know". Both can,
        however, be tested by :meth:`~Orange.data.Value.is_special`.

    .. attribute:: n_left, n_right

        The number of items on the left and on the
        right side of the rule.

    .. attribute:: n_applies_left, n_applies_right, n_applies_both

        The number of data instances matching the left, right and both sides of the rule, correspondingly.

    .. attribute:: n_examples

        The total number of training instances.

    .. attribute:: support

        nAppliesBoth/nExamples.

    .. attribute:: confidence

        n_applies_both/n_applies_left.

    .. attribute:: coverage

        n_applies_left/n_examples.

    .. attribute:: strength

        n_applies_right/n_applies_left.

    .. attribute:: lift

        n_examples * n_applies_both / (n_applies_left * n_applies_right).

    .. attribute:: leverage

        (n_Applies_both * n_examples - n_applies_left * n_applies_right).

    .. attribute:: examples, match_left, match_both

        If store_examples was `True` during induction, examples contain a copy
        of the data table used to induce the rules. Attributes `match_left`
        and `match_both` are lists of indices of data instances that match the left,
        right and both sides of the rule, respectively.

    .. method:: applies_left(data_instance)

    .. method:: applies_right(data_instance)

    .. method:: applies_both(data_instance)

        Tests if data instance is matched by the left, right or both sides of
        the rule, respectively. The data instance must be in the same representation as data from which the rule
        was inferred.

Association rule inducers do not store information on supporting data instances from training data set.
Let us write a script that finds the data instances that
match the rule (fit both sides of it) and those that contradict it (fit the
left-hand side but not the right):

.. literalinclude:: code/associate-traceback.py

The latter printouts get simpler and faster if we instruct the inducer to
store the examples::

    print "Match left: "
    print "\n".join(str(rule.examples[i]) for i in rule.match_left)
    print "\nMatch both: "
    print "\n".join(str(rule.examples[i]) for i in rule.match_both)

The "contradicting" examples are those whose indices are found in
match_left but not in match_both. The memory friendlier and the faster way
to compute this is::

    >>> [x for x in rule.match_left if not x in rule.match_both]
    [0, 2, 8, 10, 16, 17, 18]
    >>> set(rule.match_left) - set(rule.match_both)
    set([0, 2, 8, 10, 16, 17, 18])

=========
Utilities
=========

.. autofunction:: print_rules

.. autofunction:: sort

.. rubric:: References

.. [AgrawalSrikant1994] R Agrawal and R Srikant: Fast algorithms for mining association
   rules in large databases. In Proc. 20th International Conference on Very Large Data Bases, pages 487-499,
   Santiago, Chile, September 1994.

.. [TanSteinbachKumar2005] P-N Tan, M Steinbach and V Kumar: Introduction to Data Mining,
   chapter on `Association analysis: basic concepts and algorithms
   <http://www-users.cs.umn.edu/~kumar/dmbook/ch6.pdf>`_, Addison Wesley, 2005.
