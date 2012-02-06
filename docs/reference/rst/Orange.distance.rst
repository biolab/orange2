.. py:currentmodule:: Orange.distance

##########################################
Distance (``distance``)
##########################################

Distance measures typically have to be adjusted to the data. For instance,
when the data set contains continuous features, the distances between
continuous values should be normalized to ensure that all features have
similar impats, e.g. by dividing the distance with the range.

Distance measures thus appear in pairs - a class that measures
the distance (:obj:`Distance`) and a class that constructs it based on the
data (:obj:`DistanceConstructor`).

Since most measures work on normalized distances between corresponding
features, an abstract class `DistanceNormalized` takes care of
normalizing.

Unknown values are treated correctly only by Euclidean and Relief
distance.  For other measures, a distance between unknown and known or
between two unknown values is always 0.5.

.. autofunction:: distance_matrix

.. class:: Distance

    .. method:: __call__(instance1, instance2)

        Return a distance between the given instances (as a floating point number).

.. class:: DistanceConstructor

    .. method:: __call__([instances, weightID][, distributions][, basic_var_stat])

        Constructs an :obj:`Distance`. Not all arguments are required.
        Most measures can be constructed from basic_var_stat; if it is
        not given, instances or distributions can be used.

.. class:: DistanceNormalized

    An abstract class that provides normalization.

    .. attribute:: normalizers

        A precomputed list of normalizing factors for feature values. They are:

        - 1/(max_value-min_value) for continuous and 1/number_of_values
          for ordinal features.
          If either feature is unknown, the distance is 0.5. Such factors
          are used to multiply differences in feature's values.
        - ``-1`` for nominal features; the distance
          between two values is 0 if they are same (or at least one is
          unknown) and 1 if they are different.
        - ``0`` for ignored features.

    .. attribute:: bases, averages, variances

        The minimal values, averages and variances
        (continuous features only).

    .. attribute:: domain_version

        The domain version changes each time a domain description is
        changed (i.e. features are added or removed).

    .. method:: feature_distances(instance1, instance2)

        Return a list of floats representing normalized distances between
        pairs of feature values of the two instances.

.. class:: Hamming
.. class:: HammingDistance

    The number of features in which the two instances differ. This measure
    is not appropriate for instances that contain continuous features.

.. class:: Maximal
.. class:: MaximalDistance

    The maximal distance
    between two feature values. If dist is the result of
    ~:obj:`DistanceNormalized.feature_distances`,
    then :class:`Maximal` returns ``max(dist)``.

.. class:: Manhattan
.. class:: ManhattanDistance

    The sum of absolute values
    of distances between pairs of features, e.g. ``sum(abs(x) for x in dist)``
    where dist is the result of ~:obj:`DistanceNormalized.feature_distances`.

.. class:: Euclidean
.. class:: EuclideanDistance

    The square root of sum of squared per-feature distances,
    i.e. ``sqrt(sum(x*x for x in dist))``, where dist is the result of
    ~:obj:`DistanceNormalized.feature_distances`.

    .. method:: distributions

        A :obj:`~Orange.statistics.distribution.Distribution` containing
        the distributions for all discrete features used for
        computation of distances between known and unknown values.

    .. method:: both_special_dist

        A list containing the distance between two unknown values for each
        discrete feature.

    Unknown values are handled by computing the
    expected square of distance based on the distribution from the
    "training" data. Squared distance between

        - A known and unknown continuous feature equals squared distance
          between the known and the average, plus variance.
        - Two unknown continuous features equals double variance.
        - A known and unknown discrete feature equals the probability
          that the unknown feature has different value than the known
          (i.e., 1 - probability of the known value).
        - Two unknown discrete features equals the probability that two
          random chosen values are equal, which can be computed as
          1 - sum of squares of probabilities.

    Continuous cases are handled as inherited from
    :class:`DistanceNormalized`. The data for discrete cases are
    stored in distributions (used for unknown vs. known value) and
    in :obj:`both_special_dist` (the precomputed distance between two
    unknown values).

.. class:: Relief
.. class:: ReliefDistance

    Relief is similar to Manhattan distance, but incorporates the
    treatment of undefined values, which is used by ReliefF measure.

    This class is derived directly from :obj:`Distance`.


.. autoclass:: PearsonR
    :members:

.. autoclass:: PearsonRDistance
    :members:

.. autoclass:: SpearmanR
    :members:

.. autoclass:: SpearmanRConstructor
    :members:


