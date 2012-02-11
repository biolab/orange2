.. py:currentmodule:: Orange.feature.scoring

#####################
Scoring (``scoring``)
#####################

.. index:: feature scoring

.. index::
   single: feature; feature scoring

Feature score is an assessment of the usefulness of the feature for
prediction of the dependant (class) variable. Orange provides classes
that compute the common feature scores for :ref:`classification
<classification>` and regression :ref:`regression <regression>`.

The script below computes the information gain of feature "tear_rate"
in the Lenses data set (loaded into ``data``):

    >>> print Orange.feature.scoring.InfoGain("tear_rate", data)
    0.548795044422

Calling the scorer by passing the variable and the data to the
constructor, like above is convenient. However, when scoring multiple
variables, some methods run much faster if the scorer is constructed,
stored and called for each variable.

    >>> gain = Orange.feature.scoring.InfoGain()
    >>> for feature in data.domain.features:
    ...     print feature.name, gain(feature, data)
    age 0.0393966436386
    prescription 0.0395109653473
    astigmatic 0.377005338669
    tear_rate 0.548795044422

The speed gain is most noticable in Relief, which computes the scores of
all features in parallel.

The module also provides a convenience function :obj:`score_all` that
computes the scores for all attributes. The following example computes
feature scores, both with :obj:`score_all` and by scoring each feature
individually, and prints out the best three features.

.. literalinclude:: code/scoring-all.py
    :lines: 7-

The output::

    Feature scores for best three features (with score_all):
    0.613 physician-fee-freeze
    0.255 el-salvador-aid
    0.228 synfuels-corporation-cutback

    Feature scores for best three features (scored individually):
    0.613 physician-fee-freeze
    0.255 el-salvador-aid
    0.228 synfuels-corporation-cutback

.. comment
    The next script uses :obj:`GainRatio` and :obj:`Relief`.

    .. literalinclude:: code/scoring-relief-gainRatio.py
        :lines: 7-

    Notice that on this data the ranks of features match::

        Relief GainRt Feature
        0.613  0.752  physician-fee-freeze
        0.255  0.444  el-salvador-aid
        0.228  0.414  synfuels-corporation-cutback
        0.189  0.382  crime
        0.166  0.345  adoption-of-the-budget-resolution

It is also possible to score features that do not appear in the data
but can be computed from it. A typical case are discretized features:

.. literalinclude:: code/scoring-info-iris.py
    :lines: 7-11

.. _callingscore:

=======================
Calling scoring methods
=======================

Scorers can be called with different type of arguments. For instance,
when given the data, most scoring methods first compute the
corresponding contingency tables. If these are already known, they can
be given to the scorer instead of the data to save some time.

Not all classes accept all kinds of arguments. :obj:`Relief`,
for instance, only supports the form with instances on the input.

.. method:: Score.__call__(attribute, data[, apriori_class_distribution][, weightID])

    :param attribute: the chosen feature, either as a descriptor,
      index, or a name.
    :type attribute: :class:`Orange.feature.Descriptor` or int or string
    :param data: data.
    :type data: `Orange.data.Table`
    :param weightID: id for meta-feature with weight.

    All scoring methods support this form.

.. method:: Score.__call__(attribute, domain_contingency[, apriori_class_distribution])

    :param attribute: the chosen feature, either as a descriptor,
      index, or a name.
    :type attribute: :class:`Orange.feature.Descriptor` or int or string
    :param domain_contingency:
    :type domain_contingency: :obj:`Orange.statistics.contingency.Domain`

.. method:: Score.__call__(contingency, class_distribution[, apriori_class_distribution])

    :param contingency:
    :type contingency: :obj:`Orange.statistics.contingency.VarClass`
    :param class_distribution: distribution of the class
      variable. If :obj:`unknowns_treatment` is :obj:`IgnoreUnknowns`,
      it should be computed on instances where feature value is
      defined. Otherwise, class distribution should be the overall
      class distribution.
    :type class_distribution:
      :obj:`Orange.statistics.distribution.Distribution`
    :param apriori_class_distribution: Optional and most often
      ignored. Useful if the scoring method makes any probability estimates
      based on apriori class probabilities (such as the m-estimate).
    :return: Feature score - the higher the value, the better the feature.
      If the quality cannot be scored, return :obj:`Score.Rejected`.
    :rtype: float or :obj:`Score.Rejected`.

The code demonstrates using the different call signatures by computing
the score of the same feature with :obj:`GainRatio`.

.. literalinclude:: code/scoring-calls.py
    :lines: 7-

.. _classification:

==========================================
Feature scoring in classification problems
==========================================

.. Undocumented: MeasureAttribute_IM, MeasureAttribute_chiSquare, MeasureAttribute_gainRatioA, MeasureAttribute_logOddsRatio, MeasureAttribute_splitGain.

.. index::
   single: feature scoring; information gain

.. class:: InfoGain

    Information gain; the expected decrease of entropy. See `page on wikipedia
    <http://en.wikipedia.org/wiki/Information_gain_ratio>`_.

.. index::
   single: feature scoring; gain ratio

.. class:: GainRatio

    Information gain ratio; information gain divided by the entropy of the feature's
    value. Introduced in [Quinlan1986]_ in order to avoid overestimation
    of multi-valued features. It has been shown, however, that it still
    overestimates features with multiple values. See `Wikipedia
    <http://en.wikipedia.org/wiki/Information_gain_ratio>`_.

.. index::
   single: feature scoring; gini index

.. class:: Gini

    Gini index is the probability that two randomly chosen instances will have different
    classes. See `Gini coefficient on Wikipedia <http://en.wikipedia.org/wiki/Gini_coefficient>`_.

.. index::
   single: feature scoring; relevance

.. class:: Relevance

    The potential value for decision rules.

.. index::
   single: feature scoring; cost

.. class:: Cost

    Evaluates features based on the cost decrease achieved by knowing the value of
    feature, according to the specified cost matrix.

    .. attribute:: cost

        Cost matrix, an instance of :obj:`Orange.misc.CostMatrix`.

    If the cost of predicting the first class of an instance that is actually in
    the second is 5, and the cost of the opposite error is 1, than an appropriate
    score can be constructed as follows::


        >>> meas = Orange.feature.scoring.Cost()
        >>> meas.cost = ((0, 5), (1, 0))
        >>> meas(3, data)
        0.083333350718021393

    Knowing the value of feature 3 would decrease the
    classification cost for approximately 0.083 per instance.

    .. comment   opposite error - is this term correct? TODO

.. index::
   single: feature scoring; ReliefF

.. class:: Relief

    Assesses features' ability to distinguish between very similar
    instances from different classes. This scoring method was first
    developed by Kira and Rendell and then improved by  Kononenko. The
    class :obj:`Relief` works on discrete and continuous classes and
    thus implements ReliefF and RReliefF.

    ReliefF is slow since it needs to find k nearest neighbours for
    each of m reference instances. As we normally compute ReliefF for
    all features in the dataset, :obj:`Relief` caches the results for
    all features, when called to score a certain feature.  When called
    again, it uses the stored results if the domain and the data table
    have not changed (data table version and the data checksum are
    compared). Caching will only work if you use the same object.
    Constructing new instances of :obj:`Relief` for each feature,
    like this::

        for attr in data.domain.attributes:
            print Orange.feature.scoring.Relief(attr, data)

    runs much slower than reusing the same instance::

        meas = Orange.feature.scoring.Relief()
        for attr in table.domain.attributes:
            print meas(attr, data)


    .. attribute:: k

       Number of neighbours for each instance. Default is 5.

    .. attribute:: m

        Number of reference instances. Default is 100. When -1, all
        instances are used as reference.

    .. attribute:: check_cached_data

        Check if the cached data is changed, which may be slow on large
        tables.  Defaults to :obj:`True`, but should be disabled when it
        is certain that the data will not change while the scorer is used.

.. autoclass:: Orange.feature.scoring.Distance

.. autoclass:: Orange.feature.scoring.MDL

.. _regression:

======================================
Feature scoring in regression problems
======================================

.. class:: Relief

    Relief is used for regression in the same way as for
    classification (see :class:`Relief` in classification
    problems).

.. index::
   single: feature scoring; mean square error

.. class:: MSE

    Implements the mean square error score.

    .. attribute:: unknowns_treatment

        Decides the treatment of unknown values. See
        :obj:`Score.unknowns_treatment`.

    .. attribute:: m

        Parameter for m-estimate of error. Default is 0 (no m-estimate).

============
Base Classes
============

Implemented methods for scoring relevances of features are subclasses
of :obj:`Score`. Those that compute statistics on conditional
distributions of class values given the feature values are derived from
:obj:`ScoreFromProbabilities`.

.. class:: Score

    Abstract base class for feature scoring. Its attributes describe which
    types of features it can handle which kind of data it requires.

    **Capabilities**

    .. attribute:: handles_discrete

        Indicates whether the scoring method can handle discrete features.

    .. attribute:: handles_continuous

        Indicates whether the scoring method can handle continuous features.

    .. attribute:: computes_thresholds

        Indicates whether the scoring method implements the :obj:`threshold_function`.

    **Input specification**

    .. attribute:: needs

        The type of data needed indicated by one the constants
        below. Classes with use :obj:`DomainContingency` will also handle
        generators. Those based on :obj:`Contingency_Class` will be able
        to take generators and domain contingencies.

        .. attribute:: Generator

            Constant. Indicates that the scoring method needs an instance
            generator on the input as, for example, :obj:`Relief`.

        .. attribute:: DomainContingency

            Constant. Indicates that the scoring method needs
            :obj:`Orange.statistics.contingency.Domain`.

        .. attribute:: Contingency_Class

            Constant. Indicates, that the scoring method needs the contingency
            (:obj:`Orange.statistics.contingency.VarClass`), feature
            distribution and the apriori class distribution (as most
            scoring methods).

    **Treatment of unknown values**

    .. attribute:: unknowns_treatment

        Defined in classes that are able to treat unknown values. It
        should be set to one of the values below.

        .. attribute:: IgnoreUnknowns

            Constant. Instances for which the feature value is unknown are removed.

        .. attribute:: ReduceByUnknown

            Constant. Features with unknown values are
            punished. The feature quality is reduced by the proportion of
            unknown values. For impurity scores the impurity decreases
            only where the value is defined and stays the same otherwise.

        .. attribute:: UnknownsToCommon

            Constant. Undefined values are replaced by the most common value.

        .. attribute:: UnknownsAsValue

            Constant. Unknown values are treated as a separate value.

    **Methods**

    .. method:: __call__

        Abstract. See :ref:`callingscore`.

    .. method:: threshold_function(attribute, instances[, weightID])

        Abstract.

        Assess different binarizations of the continuous feature
        :obj:`attribute`.  Return a list of tuples. The first element
        is a threshold (between two existing values), the second is
        the quality of the corresponding binary feature, and the third
        the distribution of instances below and above the threshold.
        Not all scorers return the third element.

        To show the computation of thresholds, we shall use the Iris
        data set:

        .. literalinclude:: code/scoring-info-iris.py
            :lines: 13-16

    .. method:: best_threshold(attribute, instances)

        Return the best threshold for binarization, that is, the threshold
        with which the resulting binary feature will have the optimal
        score.

        The script below prints out the best threshold for
        binarization of an feature. ReliefF is used scoring:

        .. literalinclude:: code/scoring-info-iris.py
            :lines: 18-19

.. class:: ScoreFromProbabilities

    Bases: :obj:`Score`

    Abstract base class for feature scoring method that can be
    computed from contingency matrices.

    .. attribute:: estimator_constructor
    .. attribute:: conditional_estimator_constructor

        The classes that are used to estimate unconditional
        and conditional probabilities of classes, respectively.
        Defaults use relative frequencies; possible alternatives are,
        for instance, :obj:`ProbabilityEstimatorConstructor_m` and
        :obj:`ConditionalProbabilityEstimatorConstructor_ByRows`
        (with estimator constructor again set to
        :obj:`ProbabilityEstimatorConstructor_m`), respectively.

============
Other
============

.. autoclass:: Orange.feature.scoring.OrderAttributes
   :members:

.. autofunction:: Orange.feature.scoring.score_all

.. rubric:: Bibliography

.. [Kononenko2007] Igor Kononenko, Matjaz Kukar: Machine Learning and Data Mining,
  Woodhead Publishing, 2007.

.. [Quinlan1986] J R Quinlan: Induction of Decision Trees, Machine Learning, 1986.

.. [Breiman1984] L Breiman et al: Classification and Regression Trees, Chapman and Hall, 1984.

.. [Kononenko1995] I Kononenko: On biases in estimating multi-valued attributes, International Joint Conference on Artificial Intelligence, 1995.
