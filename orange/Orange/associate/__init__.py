"""
=================
Introduction
=================

Orange provides two algorithms for induction of association rules. One is the basic Agrawal's algorithm with dynamic induction of supported itemsets and rules that is designed specifically for datasets with a large number of different items. This is, however, not really suitable for attribute-based machine learning problems, which are at the primary focus of Orange. We have thus adapted the original algorithm to be more efficient for the latter type of data, and to induce the rules in which, for contrast to Agrawal's rules, both sides don't only contain attributes (like "bread, butter -> jam") but also their values ("bread = wheat, butter = yes -> jam = plum"). As a further variation, the algorithm can be limited to search only for classification rules in which the sole attribute to appear on the right side of the rule is the class attribute.

It is also possible to extract item sets instead of association rules. These are often more interesting than the rules themselves.

Besides association rule inducer, Orange also provides a rather simplified method for classification by association rules.

=================
Association rules inducer with Agrawal's algorithm
=================

The class that induces rules by Agrawal's algorithm, accepts the data examples of two forms. The first is the standard form in which each examples is described by values of a fixed list of attributes, defined in domain. The algorithm, however, disregards the attribute values but only checks whether the value is defined or not. The rule shown above, "bread, butter -> jam" actually means that if "bread" and "butter" are defined, then "jam" is defined as well. It is expected that most of values will be undefined - if this is not so, you need to use the other association rules inducer, described in the next chapter.

Since the usual representation of examples described above is rather unsuitable for sparse examples, AssociationRulesSparseInducer can also use examples represented a bit differently. Sparse examples have no fixed attributes - the examples' domain is empty, there are neither ordinary nor class attributes. All values assigned to example are given as meta-attributes. All meta-attributes need, however, be `registered with the domain descriptor <http://orange.biolab.si/doc/reference/Domain.htm#meta-attributes>`_. If you have data of this kind, the most suitable format for it is the `basket format <http://orange.biolab.si/doc/reference/fileformats.htm#basket>`_.

In both cases, the examples are first translated into an internal AssociationRulesSparseInducer's internal format for sparse datasets. The algorithm first dynamically builds all itemsets (sets of attributes) that have at least the prescribed support. Each of these is then used to derive rules with requested confidence.

If examples were given in the sparse form, so are the left and right side of the induced rules. If examples were given in the standard form, so are the examples in association rules.

.. class:: Orange.associate.AssociationRulesSparseInducer

    .. attribute:: support
    Minimal support for the rule.
    
    .. attribute:: confidence
    Minimal confidence for the rule.
    
    .. attribute:: storeExamples
    Tells the inducer to store the examples covered by each rule and those confirming it.
    
    .. attribute:: maxItemSets
    The maximal number of itemsets.
        
The last attribute deserves some explanation. The algorithm's running time (and its memory consumption) depends on the minimal support; the lower the requested support, the more eligible itemsets will be found. There is no general rule for knowing the itemset in advance (generally, value should be around 0.3, but this depends upon the number of different items, the diversity of examples...) so it's very easy to set the limit too low. In this case, the algorithm can induce hundreds of thousands of itemsets until it runs out of memory. To prevent this, it will stop inducing itemsets and report an error when the prescribed maximum maxItemSets is exceeded. In this case, you should increase the required support. On the other hand, you can (reasonably) increase the maxItemSets to as high as you computer is able to handle.

        
Examples for AssociationRulesSparseInducer
========

We shall test the rule inducer on a dataset consisting of a brief description of Spanish Inquisition, given by Palin et al:

    NOBODY expects the Spanish Inquisition! Our chief weapon is surprise...surprise and fear...fear and surprise.... Our two weapons are fear and surprise...and ruthless efficiency.... Our *three* weapons are fear, surprise, and ruthless efficiency...and an almost fanatical devotion to the Pope.... Our *four*...no... *Amongst* our weapons.... Amongst our weaponry...are such elements as fear, surprise.... I'll come in again.

    NOBODY expects the Spanish Inquisition! Amongst our weaponry are such diverse elements as: fear, surprise, ruthless efficiency, an almost fanatical devotion to the Pope, and nice red uniforms - Oh damn!
    
The text needs to be cleaned of punctuation marks and capital letters at beginnings of the sentences, each sentence needs to be put in a new line and commas need to be inserted between the words.

inquisition.basket ::

    nobody, expects, the, Spanish, Inquisition
    our, chief, weapon, is, surprise, surprise, and, fear,fear, and, surprise
    our, two, weapons, are, fear, and, surprise, and, ruthless, efficiency
    our, three, weapons, are, fear, surprise, and, ruthless, efficiency, and, an, almost, fanatical, devotion, to, the, Pope
    our, four, no
    amongst, our, weapons
    amongst, our, weaponry, are, such, elements, as, fear, surprise
    I'll, come, in, again
    nobody, expects, the, Spanish, Inquisition
    amongst, our, weaponry, are, such, diverse, elements, as, fear, surprise, ruthless, efficiency, an, almost, fanatical, devotion, to, the, Pope, and, nice, red, uniforms, oh damn
    
Inducing the rules is trivial.

assoc-agrawal.py (uses inquisition.basket) ::

    import Orange
    data = Orange.data.Table("inquisition")

    rules = Orange.associate.AssociationRulesSparseInducer(data, support = 0.5)

    print "%5s   %5s" % ("supp", "conf")
    for r in rules:
        print "%5.3f   %5.3f   %s" % (r.support, r.confidence, r)

The induced rules are surprisingly fear-full. ::

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

If examples are weighted, weight can be passed as an additional argument to call operator.

To get only a list of supported item sets, one should call the method getItemsets. The result is a list whose elements are tuples with two elements. The first is a tuple with indices of attributes in the item set. Sparse examples are usually represented with meta attributes, so this indices will be negative. The second element is  a list of indices supporting the item set, that is, containing all the items in the set. If storeExamples is False, the second element is None.

assoc-agrawal.py (uses inquisition.basket) ::

    inducer = Orange.associate.AssociationRulesSparseInducer(support = 0.5, storeExamples = True)
    itemsets = inducer.getItemsets(data)
    
Now itemsets is a list of itemsets along with the examples supporting them since we set storeExamples to True. ::

    >>> itemsets[5]
    ((-11, -7), [1, 2, 3, 6, 9])
    >>> [data.domain[i].name for i in itemsets[5][0]]
    ['surprise', 'our']   
    
The sixth itemset contains attributes with indices -11 and -7, that is, the words "surprise" and "our". The examples supporting it are those with indices 1,2, 3, 6 and 9.

This way of representing the itemsets is not very programmer-friendly, but it is much more memory efficient than and faster to work with than using objects like Variable and Example.

=================
Association rules for non-sparse examples
=================

The other algorithm for association rules provided by Orange, AssociationRulesInducer is optimized for non-sparse examples in the usual Orange form. Each example is described by values of a fixed set of attributes. Unknown values are ignored, while values of attributes are not (as opposite to the above-described algorithm for sparse rules). In addition, the algorithm can be directed to search only for classification rules, in which the only attribute on the right-hand side is the class attribute.

.. class:: Orange.associate.AssociationRulesInducer

    .. attribute:: support
    Minimal support for the rule.
    
    .. attribute:: confidence
    Minimal confidence for the rule.
    
    .. attribute:: classificationRules
    If 1 (default is 0), the algorithm constructs classification rules instead of general association rules.
    
    .. attribute:: storeExamples
    Tells the inducer to store the examples covered by each rule and those confirming it
    
    .. attribute:: maxItemSets
    The maximal number of itemsets.

Meaning of all attributes (except the new one, classificationRules) is the same as for AssociationRulesSparseInducer. See the description of maxItemSet there.

assoc.py (uses lenses.tab) ::

    import orange

    data = orange.ExampleTable("lenses")

    print "Association rules"
    rules = orange.AssociationRulesInducer(data, supp = 0.5)
    for r in rules:
        print "%5.3f  %5.3f  %s" % (r.support, r.confidence, r)
        
The found rules are ::

    0.333  0.533  lenses=none -> prescription=hypermetrope
    0.333  0.667  prescription=hypermetrope -> lenses=none
    0.333  0.533  lenses=none -> astigmatic=yes
    0.333  0.667  astigmatic=yes -> lenses=none
    0.500  0.800  lenses=none -> tear_rate=reduced
    0.500  1.000  tear_rate=reduced -> lenses=none
    
To limit the algorithm to classification rules, set classificationRules to 1. ::

    import orange

    data = orange.ExampleTable("inquisition")
    rules = orange.AssociationRulesSparseInducer(data,
                support = 0.5, storeExamples = True)

    print "%5s   %5s" % ("supp", "conf")
    for r in rules:
        print "%5.3f   %5.3f   %s" % (r.support, r.confidence, r)

The found rules are, naturally, a subset of the above rules. ::

    0.333  0.667  prescription=hypermetrope -> lenses=none
    0.333  0.667  astigmatic=yes -> lenses=none
    0.500  1.000  tear_rate=reduced -> lenses=none
    
AssociationRulesInducer can also work with weighted examples; the ID of weight attribute should be passed as an additional argument in a call.

Itemsets are induced in a similar fashion as for sparse data, except that the first element of the tuple, the item set, is represented not by indices of attributes, as before, but with tuples (attribute-index, value-index). ::

    inducer = orange.AssociationRulesInducer(support = 0.3, storeExamples = True)
    itemsets = inducer.getItemsets(data)
    print itemsets[8]
    
This prints out ::

    (((2, 1), (4, 0)), [2, 6, 10, 14, 15, 18, 22, 23])
    
meaning that the ninth itemset contains the second value of the third attribute, (2, 1), and the first value of the fifth, (4, 0).

=================
Association rule
=================

Both classes for induction of association rules return the induced rules in AssociationRules which is basically a list of instances of AssociationRule.

.. class:: Orange.associate.AssociationRules

    .. attribute:: left, right
    The left and the right side of the rule. Both are given as Example. In rules created by AssociationRulesSparseInducer from examples that contain all values as meta-values, left and right are examples in the same form. Otherwise, values in left that do not appear in the rule are don't care, and value in right are don't know. Both can, however, be tested by isSpecial (see documentation on  `Value <http://orange.biolab.si/doc/reference/Value.htm>`_).
    
    .. attribute:: nLeft, nRight
    The number of attributes (ie defined values) on the left and on the right side of the rule.
    
    .. attribute:: nAppliesLeft, nAppliesRight, nAppliesBoth
    The number of (learning) examples that conform to the left, the right and to both sides of the rule.
    
    .. attribute:: nExamples
    The total number of learning examples.
    
    .. attribute:: support
    nAppliesBoth/nExamples.

    .. attribute:: confidence
    nAppliesBoth/nAppliesLeft.
    
    .. attribute:: coverage
    nAppliesLeft/nExamples.

    .. attribute:: strength
    nAppliesRight/nAppliesLeft.
    
    .. attribute:: lift
    nExamples * nAppliesBoth / (nAppliesLeft * nAppliesRight).
    
    .. attribute:: leverage
    (nAppliesBoth * nExamples - nAppliesLeft * nAppliesRight).
    
    .. attribute:: examples, matchLeft, matchBoth
    If storeExamples was True during induction, examples contains a copy of the example table used to induce the rules. Attributes matchLeft and matchBoth are lists of integers, representing the indices of examples which match the left-hand side of the rule and both sides, respectively.    

    .. method:: AssociationRule(left, right, nAppliesLeft, nAppliesRight, nAppliesBoth, nExamples)
    Constructs an association rule and computes all measures listed above.
    
    .. method:: AssociationRule(left, right, support, confidence]])
    Construct association rule and sets its support and confidence. If you intend to pass such a rule to someone that expects more things to be set, you should set the manually - AssociationRules's constructor cannot compute anything from these two arguments.
    
    .. method:: AssociationRule(rule)
    Given an association rules as the argument, constructor copies the rule into a new rule.
    
    .. method:: appliesLeft(example), appliesRight(example), appliesBoth(example)
    Tells whether the example fits into the left, right and both sides of the rule, respectively. If the rule is represented by sparse examples, the given example must be sparse as well.
    
Association rule inducers do not store evidence about which example supports which rule (although this is available during induction, the information is discarded afterwards). Let us write a function that find the examples that confirm the rule (ie fit both sides of it) and those that contradict it (fit the left-hand side but not the right). ::

    import orange

    data = orange.ExampleTable("lenses")

    rules = orange.AssociationRulesInducer(data, supp = 0.3)
    rule = rules[0]

    print
    print "Rule: ", rule
    print

    print "Supporting examples:"
    for example in data:
        if rule.appliesBoth(example):
            print example
    print

    print "Contradicting examples:"
    for example in data:
        if rule.appliesLeft(example) and not rule.appliesRight(example):
            print example
    print

The latter printouts get simpler and (way!) faster if we instruct the inducer to store the examples. We can then do, for instance, this. ::

    print "Match left: "
    print "\\n".join(str(rule.examples[i]) for i in rule.matchLeft)
    print "\\nMatch both: "
    print "\\n".join(str(rule.examples[i]) for i in rule.matchBoth)

The "contradicting" examples are then those whose indices are find in matchLeft but not in matchBoth. The memory friendlier and the faster ways to compute this are as follows. ::

    >>> [x for x in rule.matchLeft if not x in rule.matchBoth]
    [0, 2, 8, 10, 16, 17, 18]
    >>> set(rule.matchLeft) - set(rule.matchBoth)
    set([0, 2, 8, 10, 16, 17, 18])
    
"""

from orange import \
    AssociationRule, \
    AssociationRules, \
    AssociationRulesInducer, \
    AssociationRulesSparseInducer, \
    ItemsetNodeProxy, \
    ItemsetsSparseInducer
