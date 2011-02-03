"""

=================
Distributions and their characteristics
=================

Objects derived from Distribution are used throughout Orange to store
various distributions. These often - but not necessarily - apply to
distribution of values of certain attribute on some dataset.
You will most often encounter two classes derived from Distribution:
DiscDistribution stores discrete and ContDistribution stores continuous
distributions. To some extent, they both resemble dictionaries,
with attribute values as keys and number of examples with particular
value as elements.

=================
General Distributions
=================

Class Distribution contains the common methods for different types of
distributions. Even more, its constructor can be used to construct objects
of type DiscDistribution and ContDistribution (class Distribution itself
is abstract, so no instances of that class can actually exist).



**distributions-test.py** (uses inquisition.basket) ::

    import Orange
    import distributions

    import Orange.core as orange

    myData = Orange.data.Table("adult_sample.tab")
    disc = Orange.data.value.Distribution("workclass", myData)

    print disc
    print type(disc) 




"""


from orange import \
     BasicAttrStat, \
     DomainBasicAttrStat, \
     DomainContingency, \
     DomainDistributions, \
     DistributionList, \
     ComputeDomainContingency, \
     ConditionalProbabilityEstimator, \
     ConditionalProbabilityEstimator_ByRows, \
     ConditionalProbabilityEstimator_FromDistribution, \
     ConditionalProbabilityEstimatorConstructor, \
     ConditionalProbabilityEstimatorConstructor_ByRows, \
     ConditionalProbabilityEstimatorConstructor_loess, \
     ConditionalProbabilityEstimatorList, \
     Contingency, \
     ContingencyAttrAttr, \
     ContingencyClass, \
     ContingencyAttrClass, \
     ContingencyClassAttr
