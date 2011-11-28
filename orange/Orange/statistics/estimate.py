"""
=======================================
Probability Estimation (``estimate``)
=======================================

Probability estimators are compute value probabilities.

There are two branches of probability estimators:

#. for unconditional and

#. for conditional probabilities.

For naive Bayesian classification the first compute p(C)
and the second p(C|v), where C is a class and v is a feature value.

Since probability estimation is usually based on the data, the whole
setup is done in orange way. As for learning, where you use a learner
to construct a classifier, in probability estimation there are estimator
constructors whose purpose is to construct probability estimators.

This page is divided into three sections. The first describes the basic
classes, the second contains classes that are abstract or only support
"real" estimators - you would seldom use these directly. The last section
contains estimators and constructors that you would most often use. If
you are not interested details, skip the first two sections.

Basic classes
=============

Four basic abstract classes serve as roots of the hierarchy:
:class:`Estimator`, :class:`EstimatorConstructor`,
:class:`ConditionalEstimator` and
:class:`ConditionalEstimatorConstructor`.

.. class:: Estimator

    .. attribute:: supports_discrete
    
        Tells whether the estimator can handle discrete attributes.
        
    .. attribute:: supports_continuous
        
        Tells whether the estimator can handle continuous attributes.

    .. method:: __call__([value])

        If value is given, Return the  probability of the value
        (as float).  When the value is omitted, the object attempts
        to return a distribution of probabilities for all values (as
        :class:`~Orange.statistics.distribution.Distribution`). The
        result can be :class:`~Orange.statistics.distribution.Discrete`
        for discrete, :class:`~Orange.statistics.distribution.Continuous`
        for continuous features or an instance of some other class derived
        from :class:`~Orange.statistics.distribution.Distribution`. Note
        that it indeed makes sense to return continuous
        distribution. Although probabilities are stored
        point-wise (as something similar to Python's map, where
        keys are attribute values and items are probabilities,
        :class:`~Orange.statistics.distribution.Distribution` can compute
        probabilities between the recorded values by interpolation.

        The estimator does not necessarily support
        returning precomputed probabilities in form of
        :class:`~Orange.statistics.distribution.Distribution`; in this
        case, it simply returns None.

.. class:: EstimatorConstructor

    This is an abstract class; derived classes define call operators
    that return different probability estimators. The class is
    call-constructible (i.e., if called with appropriate parameters,
    the constructor returns a probability estimator, not a probability
    estimator constructor).

    The call operator can accept an already computed distribution of
    classes or a list of examples or both.

    .. method:: __call__([distribution[, apriori]], [examples[,weightID]])

        If distribution is given, it can be followed by apriori class
        distribution. Similarly, examples can be followed by with
        the ID of meta attribute with example weights. (Hint: if you
        want to have examples and a priori distribution, but don't have
        distribution ready, just pass None for distribution.) When both,
        distribution and examples are given, it is up to constructor to
        decide what to use.


.. class:: ConditionalEstimator

    As a counterpart of :class:`Estimator`, this estimator can return
    conditional probabilities.

    .. method:: __call__([[Value,] ConditionValue])

        When given two values, it returns a probability of
        p(Value|Condition) (as float). When given only one value,
        it is interpreted as condition; the estimator returns a
        :class:`~Orange.statistics.distribution.Distribution` with
        probabilities p(v|Condition) for each possible value v. When
        called without arguments, it returns a :class:`Orange.statistics.contingency.Table`
        matrix containing probabilities p(v|c) for each possible value
        and condition; condition is used as outer variable.

        If estimator cannot return precomputed distributions and/or
        contingencies, it returns None.

.. class:: ConditionalEstimatorConstructor

    A counterpart of :class:`EstimatorConstructor`. It has
    similar arguments, except that the first argument is not a
    :class:`~Orange.statistics.distribution.Distribution` but
    :class:`Orange.statistics.contingency.Table`.


Abstract and supporting classes 
===============================

    There are several abstract classes that simplify the actual classes
    for probability estimation.

.. class:: EstimatorFromDistribution

    .. attribute:: probabilities

        A precomputed list of probabilities.

    There are many estimator constructors that compute
    probabilities of classes from frequencies of classes
    or from list of examples. Probabilities are stored as
    :class:`~Orange.statistics.distribution.Distribution`, and
    :class:`EstimatorFromDistribution` is returned. This is done for
    estimators that use relative frequencies, Laplace's estimation,
    m-estimation and even estimators that compute continuous
    distributions.

    When asked about probability of certain value, the estimator
    returns a corresponding element of :obj:`probabilities`. Note that
    when distribution is continuous, linear interpolation between two
    points is used to compute the probability. When asked for a complete
    distribution, it returns a copy of :obj:`probabilities`.

.. class:: ConditionalEstimatorFromDistribution

    .. attribute:: probabilities

        A precomputed list of probabilities

    This counterpart of :class:`EstimatorFromDistribution` stores
    conditional probabilities in :class:`Orange.statistics.contingency.Table`.

.. class:: ConditionalEstimatorByRows

    .. attribute:: estimator_list

        A list of estimators; one for each value of
        :obj:`Condition`.

    This conditional probability estimator has different estimators for
    different values of conditional attribute. For instance, when used
    for computing p(c|A) in naive Bayesian classifier, it would have
    an estimator for each possible value of attribute A. This does not
    mean that the estimators were constructed by different constructors,
    i.e. using different probability estimation methods. This class is
    normally used when we only have a probability estimator constructor
    for unconditional probabilities but need to construct a conditional
    probability estimator; the constructor is used to construct estimators
    for subsets of original example set and the resulting estimators
    are stored in :class:`ConditionalEstimatorByRows`.

.. class:: ConditionalByRows

    .. attribute:: estimator_constructor

        An unconditional probability estimator constructor.

    This class computes a conditional probability estimator using
    an unconditional probability estimator constructor. The result
    can be of type :class:`ConditionalEstimatorFromDistribution`
    or :class:`ConditionalEstimatorByRows`, depending on the type of
    constructor.

    The class first computes contingency matrix if it hasn't been
    computed already. Then it calls :obj:`estimator_constructor`
    for each value of condition attribute. If all constructed
    estimators can return distribution of probabilities
    for all classes (usually either all or none can), the
    :class:`~Orange.statistics.distribution.Distribution` are put in
    a contingency, and :class:`ConditionalEstimatorFromDistribution`
    is constructed and returned. If constructed estimators are
    not capable of returning distribution of probabilities,
    a :class:`ConditionalEstimatorByRows` is constructed and the
    estimators are stored in its :obj:`estimator_list`.


Concrete probability estimators and constructors
================================================

.. class:: RelativeFrequency

    Computes relative frequencies of classes, puts it into a Distribution
    and returns it as :class:`EstimatorFromDistribution`.

.. class:: Laplace

    Uses Laplace estimation to compute probabilities from frequencies
    of classes.

    .. math::

        p(c) = (Nc+1) / (N+n)

    where Nc is number of occurences of an event (e.g. number of examples
    in class c), N is the total number of events (examples) and n is
    the number of different events (classes).

    The resulting estimator is again of type
    :class:`EstimatorFromDistribution`.

.. class:: M

    .. attribute:: m

        Parameter for m-estimation

    Uses m-estimation to compute probabilities from frequencies of
    classes.

    .. math::

        p(c) = (Nc+m*ap(c)) / (N+m)

    where Nc is number of occurences of an event (e.g. number of examples
    in class c), N is the total number of events (examples) and ap(c)
    is the apriori probability of event (class) c.

    The resulting estimator is of type :class:`EstimatorFromDistribution`.

.. class:: Kernel

    .. attribute:: min_impact

        A requested minimal weight of a point (default: 0.01); points
        with lower weights won't be taken into account.

    .. attribute:: smoothing

        Smoothing factor (default: 1.144)

    .. attribute:: n_points

        Number of points for the interpolating curve. If negative, say -3
        (default), 3 points will be inserted between each data points.

    Useful for continuous distributions, this constructor computes
    probabilities for certain number of points using Gaussian
    kernels. The resulting point-wise continuous distribution is stored
    as :class:`~Orange.statistics.distribution.Continuous` and returned
    in :class:`EstimatorFromDistribution`.

    The points at which probabilities are computed are determined
    like this.  Probabilities are always computed at all points that
    are present in the data (i.e. the existing values of the continuous
    attribute). If :obj:`n_points` is positive and greater than the
    number of existing data points, additional points are inserted
    between the existing points to achieve the required number of
    points. Approximately equal number of new points is inserted between
    each adjacent existing point each data points.

.. class:: Loess

    .. attribute:: window_proportion

        A proportion of points in a window.

    .. attribute:: n_points

        Number of points for the interpolating curve. If negative, say -3
        (default), 3 points will be inserted between each data points.

    This method of probability estimation is similar to
    :class:`Kernel`. They both return a curve computed at certain number
    of points and the points are determined by the same procedure. They
    differ, however, at the method for estimating the probabilities.

    To estimate probability at point ``x``, :class:`Loess` examines a
    window containing a prescribed proportion of original data points. The
    window is as simetric as possible; the number of points to the left
    of ``x`` might differ from the number to the right, but the leftmost
    point is approximately as far from ``x`` as the rightmost. Let us
    denote the width of the windows, e.g. the distance to the farther
    of the two edge points, by ``h``.

    Points are weighted by bi-cubic weight function; a weight of point
    at ``x'`` is :math:`(1-|t|^3)^3`, where ``t`` is
    :math:`(x-x'>)/h`.

    Probability at point ``x`` is then computed as weighted local
    regression of probabilities for points in the window.

.. class:: ConditionalLoess

    .. attribute:: window_proportion

        A proportion of points in a window.

    .. attribute:: n_points

        Number of points for the interpolating curve. If negative, say -3
        (default), 3 points will be inserted between each data points.

    Constructs similar estimator as :class:`Loess`, except that
    it computes conditional probabilites. The result is of type
    :class:`ConditionalEstimatorFromDistribution`.

"""

import Orange
from Orange.core import ProbabilityEstimator as Estimator
from Orange.core import ProbabilityEstimator_FromDistribution as EstimatorFromDistribution
from Orange.core import ProbabilityEstimatorConstructor as EstimatorConstructor
from Orange.core import ProbabilityEstimatorConstructor_Laplace as Laplace
from Orange.core import ProbabilityEstimatorConstructor_kernel as Kernel
from Orange.core import ProbabilityEstimatorConstructor_loess as Loess
from Orange.core import ProbabilityEstimatorConstructor_m as M
from Orange.core import ProbabilityEstimatorConstructor_relative as RelativeFrequency
from Orange.core import ConditionalProbabilityEstimator as ConditionalEstimator
from Orange.core import ConditionalProbabilityEstimator_FromDistribution as ConditionalEstimatorFromDistribution
from Orange.core import ConditionalProbabilityEstimator_ByRows as ConditionalEstimatorByRows
from Orange.core import ConditionalProbabilityEstimatorConstructor_ByRows as ConditionalByRows
from Orange.core import ConditionalProbabilityEstimatorConstructor_loess as ConditionalLoess
