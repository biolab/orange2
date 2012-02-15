.. automodule:: Orange.statistics.estimate

.. index:: Probability Estimation

=======================================
Probability Estimation (``estimate``)
=======================================

Probability estimators compute probabilities of values of class variable.
They come in two flavours:

#. for unconditional probabilities (:math:`p(C=c)`, where :math:`c` is a
   class) and

#. for conditional probabilities (:math:`p(C=c|V=v)`,
   where :math:`v` is a feature value).

A duality much like the one between learners and classifiers exists between
probability estimator constructors and probability estimators: when a
probability estimator constructor is called with data, it constructs a
probability estimator that can then be called with a value of class variable
to obtain a probability of that value. This duality is mainly needed to
enable probability estimation for continuous variables,
where it is not possible to generate a list of probabilities of all possible
values in advance.

First, probability estimation constructors for common probability estimation
techniques are enumerated. Base classes, knowledge of which is needed to
develop new techniques, are described later in this document.

Probability Estimation Constructors
===================================

.. class:: RelativeFrequency

    Bases: :class:`EstimatorConstructor`

    Compute distribution using relative frequencies of classes.

    :rtype: :class:`EstimatorFromDistribution`

.. class:: Laplace

    Bases: :class:`EstimatorConstructor`

    Use Laplace estimation to compute distribution from frequencies of classes:

    .. math::

        p(c) = \\frac{Nc+1}{N+n}

    where :math:`Nc` is number of occurrences of an event (e.g. number of
    instances in class c), :math:`N` is the total number of events (instances)
    and :math:`n` is the number of different events (classes).

    :rtype: :class:`EstimatorFromDistribution`

.. class:: M

    Bases: :class:`EstimatorConstructor`

    .. method:: __init__(m)

        :param m: Parameter for m-estimation.
        :type m: int

    Use m-estimation to compute distribution from frequencies of classes:

    .. math::

        p(c) = \\frac{Nc+m*ap(c)}{N+m}

    where :math:`Nc` is number of occurrences of an event (e.g. number of
    instances in class c), :math:`N` is the total number of events (instances)
    and :math:`ap(c)` is the prior probability of event (class) c.

    :rtype: :class:`EstimatorFromDistribution`

.. class:: Kernel

    Bases: :class:`EstimatorConstructor`

    .. method:: __init__(min_impact, smoothing, n_points)

        :param min_impact: A requested minimal weight of a point (default:
            0.01); points with lower weights won't be taken into account.
        :type min_impact: float

        :param smoothing: Smoothing factor (default: 1.144).
        :type smoothing: float

        :param n_points: Number of points for the interpolating curve. If
            negative, say -3 (default), 3 points will be inserted between each
            data points.
        :type n_points: int

    Compute probabilities for continuous variable for certain number of points
    using Gaussian kernels. The resulting point-wise continuous distribution is
    stored as :class:`~Orange.statistics.distribution.Continuous`.

    Probabilities are always computed at all points that
    are present in the data (i.e. the existing values of the continuous
    feature). If :obj:`n_points` is positive and greater than the
    number of existing data points, additional points are inserted
    between the existing points to achieve the required number of
    points. Approximately equal number of new points is inserted between
    each adjacent existing point each data points. If :obj:`n_points` is
    negative, its absolute value determines the number of points to be added
    between each two data points.

    :rtype: :class:`EstimatorFromDistribution`

.. class:: Loess

    Bases: :class:`EstimatorConstructor`

    .. method:: __init__(window_proportion, n_points)

        :param window_proportion: A proportion of points in a window.
        :type window_proportion: float

        :param n_points: Number of points for the interpolating curve. If
            negative, say -3 (default), 3 points will be inserted between each
            data points.
        :type n_points: int

    Prepare a probability estimator that computes probability at point ``x``
    as weighted local regression of probabilities for points in the window
    around this point.

    The window contains a prescribed proportion of original data points. The
    window is as symmetric as possible in the sense that the leftmost point in
    the window is approximately as far from ``x`` as the rightmost. The
    number of points to the left of ``x`` might thus differ from the number
    of points to the right.

    Points are weighted by bi-cubic weight function; a weight of point
    at ``x'`` is :math:`(1-|t|^3)^3`, where :math:`t` is
    :math:`(x-x'>)/h` and :math:`h` is the distance to the farther
    of the two window edge points.

    :rtype: :class:`EstimatorFromDistribution`


.. class:: ConditionalLoess

    Bases: :class:`ConditionalEstimatorConstructor`

    .. method:: __init__(window_proportion, n_points)

        :param window_proportion: A proportion of points in a window.
        :type window_proportion: float

        :param n_points: Number of points for the interpolating curve. If
            negative, say -3 (default), 3 points will be inserted between each
            data points.
        :type n_points: int

    Construct a conditional probability estimator, in other aspects
    similar to the one constructed by :class:`Loess`.

    :rtype: :class:`ConditionalEstimatorFromDistribution`.


Base classes
=============

All probability estimators are derived from two base classes: one for
unconditional and the other for conditional probability estimation. The same
is true for probability estimator constructors.

.. class:: EstimatorConstructor

    Constructor of an unconditional probability estimator.

    .. method:: __call__([distribution[, prior]], [instances[, weight_id]])

        :param distribution: input distribution.
        :type distribution: :class:`~Orange.statistics.distribution.Distribution`

        :param priori: prior distribution.
        :type distribution: :class:`~Orange.statistics.distribution.Distribution`

        :param instances: input data.
        :type distribution: :class:`Orange.data.Table`

        :param weight_id: ID of the weight attribute.
        :type weight_id: int

        If distribution is given, it can be followed by prior class
        distribution. Similarly, instances can be followed by with
        the ID of meta attribute with instance weights. (Hint: to pass a
        prior distribution and instances, but no distribution,
        just pass :obj:`None` for the latter.) When both,
        distribution and instances are given, it is up to constructor to
        decide what to use.

.. class:: Estimator

    .. attribute:: supports_discrete

        Tells whether the estimator can handle discrete attributes.

    .. attribute:: supports_continuous

        Tells whether the estimator can handle continuous attributes.

    .. method:: __call__([value])

        If value is given, return the probability of the value.

        :rtype: float

        If the value is omitted, an attempt is made
        to return a distribution of probabilities for all values.

        :rtype: :class:`~Orange.statistics.distribution.Distribution`
            (usually :class:`~Orange.statistics.distribution.Discrete` for
            discrete and :class:`~Orange.statistics.distribution.Continuous`
            for continuous) or :obj:`NoneType`

.. class:: ConditionalEstimatorConstructor

    Constructor of a conditional probability estimator.

    .. method:: __call__([table[, prior]], [instances[, weight_id]])

        :param table: input distribution.
        :type table: :class:`Orange.statistics.contingency.Table`

        :param prior: prior distribution.
        :type distribution: :class:`~Orange.statistics.distribution.Distribution`

        :param instances: input data.
        :type distribution: :class:`Orange.data.Table`

        :param weight_id: ID of the weight attribute.
        :type weight_id: int

        If distribution is given, it can be followed by prior class
        distribution. Similarly, instances can be followed by with
        the ID of meta attribute with instance weights. (Hint: to pass a
        prior distribution and instances, but no distribution,
        just pass :obj:`None` for the latter.) When both,
        distribution and instances are given, it is up to constructor to
        decide what to use.

.. class:: ConditionalEstimator

    As a counterpart of :class:`Estimator`, this estimator can return
    conditional probabilities.

    .. method:: __call__([[value,] condition_value])

        When given two values, it returns a probability of :math:`p(value|condition)`.

        :rtype: float

        When given only one value, it is interpreted as condition; the estimator
        attempts to return a distribution of conditional probabilities for all
        values.

        :rtype: :class:`~Orange.statistics.distribution.Distribution`
            (usually :class:`~Orange.statistics.distribution.Discrete` for
            discrete and :class:`~Orange.statistics.distribution.Continuous`
            for continuous) or :obj:`NoneType`

        When called without arguments, it returns a
        matrix containing probabilities :math:`p(value|condition)` for each
        possible :math:`value` and :math:`condition` (a contingency table);
        condition is used as outer
        variable.

        :rtype: :class:`Orange.statistics.contingency.Table` or :obj:`NoneType`

        If estimator cannot return precomputed distributions and/or
        contingencies, it returns :obj:`None`.

Common Components
=================

.. class:: EstimatorFromDistribution

    Bases: :class:`Estimator`

    Probability estimator constructors that compute probabilities for all
    values in advance return this estimator with calculated
    quantities in the :obj:`probabilities` attribute.

    .. attribute:: probabilities

        A precomputed list of probabilities.

    .. method:: __call__([value])

        If value is given, return the probability of the value. For discrete
        variables, every value has an entry in the :obj:`probabilities`
        attribute. For continuous variables, a linear interpolation between
        two nearest points is used to compute the probability.

        :rtype: float

        If the value is omitted, a copy of :obj:`probabilities` is returned.

        :rtype: :class:`~Orange.statistics.distribution.Distribution`
            (usually :class:`~Orange.statistics.distribution.Discrete` for
            discrete and :class:`~Orange.statistics.distribution.Continuous`
            for continuous).

.. class:: ConditionalEstimatorFromDistribution

    Bases: :class:`ConditionalEstimator`

    Probability estimator constructors that compute the whole
    contingency table (:class:`Orange.statistics.contingency.Table`) of
    conditional probabilities in advance
    return this estimator with the table in the :obj:`probabilities` attribute.

    .. attribute:: probabilities

        A precomputed contingency table.

    .. method:: __call__([[value,] condition_value])

        For detailed description of handling of different combinations of
        parameters, see the inherited :obj:`ConditionalEstimator.__call__`.
        For behaviour with continuous variable distributions,
        see the unconditional counterpart :obj:`EstimatorFromDistribution.__call__`.

.. class:: ConditionalByRows

    Bases: :class:`ConditionalEstimator`

    .. attribute:: estimator_constructor

        An unconditional probability estimator constructor.

    Computes a conditional probability estimator using
    an unconditional probability estimator constructor. The result
    can be of type :class:`ConditionalEstimatorFromDistribution`
    or :class:`ConditionalEstimatorByRows`, depending on the type of
    constructor.

    .. method:: __call__([table[, prior]], [instances[, weight_id]], estimator)

        :param table: input distribution.
        :type table: :class:`Orange.statistics.contingency.Table`

        :param prior: prior distribution.
        :type distribution: :class:`~Orange.statistics.distribution.Distribution`

        :param instances: input data.
        :type distribution: :class:`Orange.data.Table`

        :param weight_id: ID of the weight attribute.
        :type weight_id: int

        :param estimator: unconditional probability estimator constructor.
        :type estimator: :class:`EstimatorConstructor`

        Compute contingency matrix if it has not been computed already. Then
        call :obj:`estimator_constructor` for each value of condition attribute.
        If all constructed estimators can return distribution of probabilities
        for all classes (usually either all or none can), the
        :class:`~Orange.statistics.distribution.Distribution` instances are put
        in a contingency table
        and :class:`ConditionalEstimatorFromDistribution`
        is constructed and returned. If constructed estimators are
        not capable of returning distribution of probabilities,
        a :class:`ConditionalEstimatorByRows` is constructed and the
        estimators are stored in its :obj:`estimator_list`.

        :rtype: :class:`ConditionalEstimatorFromDistribution` or :class:`ConditionalEstimatorByRows`

.. class:: ConditionalEstimatorByRows

    Bases: :class:`ConditionalEstimator`

    A conditional probability estimator constructors that itself uses a series
    of estimators, one for each possible condition,
    stored in its :obj:`estimator_list` attribute.

    .. attribute:: estimator_list

        A list of estimators; one for each value of :obj:`condition`.

    .. method:: __call__([[value,] condition_value])

        Uses estimators from :obj:`estimator_list`,
        depending on given `condition_value`.
        For detailed description of handling of different combinations of
        parameters, see the inherited :obj:`ConditionalEstimator.__call__`.

