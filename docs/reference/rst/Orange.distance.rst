.. py:currentmodule:: Orange.distance

.. automodule:: Orange.distance

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

.. class:: Distance

    .. method:: __call__(instance1, instance2)

        Return a distance between the given instances (as a floating point number).

.. class:: DistanceConstructor

    .. method:: __call__([instances, weightID][, distributions][, basic_var_stat])

        Constructs an :obj:`Distance`.  Not all the data needs to be
        given. Most measures can be constructed from basic_var_stat;
        if it is not given, they can help themselves either by instances
        or distributions. Some do not need any arguments.

.. class:: DistanceNormalized

    This abstract class provides a function which is given two instances
    and returns a list of normalized distances between values of their
    features. Many distance measuring classes need such a function and are
    therefore derived from this class

    .. attribute:: normalizers

        A precomputed list of normalizing factors for feature values

        - If a factor positive, differences in feature's values
          are multiplied by it; for continuous features the factor
          would be 1/(max_value-min_value) and for ordinal features
          the factor is 1/number_of_values. If either (or both) of
          features are unknown, the distance is 0.5
        - If a factor is -1, the feature is nominal; the distance
          between two values is 0 if they are same (or at least
          one is unknown) and 1 if they are different.
        - If a factor is 0, the feature is ignored.

    .. attribute:: bases, averages, variances

        The minimal values, averages and variances
        (continuous features only)

    .. attribute:: domain_version

        The domain version increases each time a domain description is
        changed (i.e. features are added or removed); this checks 
        that the user is not attempting to measure distances between
        instances that do not correspond to normalizers.

    .. method:: attribute_distances(instance1, instance2)

        Return a list of floats representing distances between pairs of
        feature values of the two instances.

.. class:: HammingConstructor
.. class:: Hamming

    Hamming distance between two instances is defined as the number of
    features in which the two instances differ. Note that this measure
    is not really appropriate for instances that contain continuous features.

.. class:: MaximalConstructor
.. class:: Maximal

    The maximal between two instances is defined as the maximal distance
    between two feature values. If dist is the result of
    DistanceNormalized.attribute_distances,
    then Maximal returns max(dist).

.. class:: ManhattanConstructor
.. class:: Manhattan

    Manhattan distance between two instances is a sum of absolute values
    of distances between pairs of features, e.g. ``sum(abs(x) for x in dist)``
    where dist is the result of ExamplesDistance_Normalized.attributeDistances.

.. class:: EuclideanConstructor
.. class:: Euclidean

    Euclidean distance is a square root of sum of squared per-feature distances,
    i.e. ``sqrt(sum(x*x for x in dist))``, where dist is the result of
    ExamplesDistance_Normalized.attributeDistances.

    .. method:: distributions

        An object of type
        :obj:`~Orange.statistics.distribution.Distribution` that holds
        the distributions for all discrete features used for
        computation of distances between known and unknown values.

    .. method:: bothSpecialDist

        A list containing the distance between two unknown values for each
        discrete feature.

    This measure of distance deals with unknown values by computing the
    expected square of distance based on the distribution obtained from the
    "training" data. Squared distance between

        - A known and unknown continuous attribute equals squared distance
          between the known and the average, plus variance
        - Two unknown continuous attributes equals double variance
        - A known and unknown discrete attribute equals the probability
          that the unknown attribute has different value than the known
          (i.e., 1 - probability of the known value)
        - Two unknown discrete attributes equals the probability that two
          random chosen values are equal, which can be computed as
          1 - sum of squares of probabilities.

    Continuous cases can be handled by averages and variances inherited from
    ExamplesDistance_normalized. The data for discrete cases are stored in
    distributions (used for unknown vs. known value) and in bothSpecial
    (the precomputed distance between two unknown values).

.. class:: ReliefConstructor
.. class:: Relief

    Relief is similar to Manhattan distance, but incorporates a more
    correct treatment of undefined values, which is used by ReliefF measure.

This class is derived directly from ExamplesDistance, not from ExamplesDistance_Normalized.


.. autoclass:: PearsonR
    :members:

.. autoclass:: SpearmanR
    :members:

.. autoclass:: PearsonRConstructor
    :members:

.. autoclass:: SpearmanRConstructor
    :members:
