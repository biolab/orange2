.. automodule:: Orange.distance.instances

#########################
Instances (``instances``)
#########################

###########################
Distances between Instances
###########################

This page describes a bunch of classes for different metrics for measure
distances (dissimilarities) between instances.

Typical (although not all) measures of distance between instances require
some "learning" - adjusting the measure to the data. For instance, when
the dataset contains continuous features, the distances between continuous
values should be normalized, e.g. by dividing the distance with the range
of possible values or with some interquartile distance to ensure that all
features have, in principle, similar impacts.

Different measures of distance thus appear in pairs - a class that measures
the distance and a class that constructs it based on the data. The abstract
classes representing such a pair are `ExamplesDistance` and
`ExamplesDistanceConstructor`.

Since most measures work on normalized distances between corresponding
features, there is an abstract intermediate class
`ExamplesDistance_Normalized` that takes care of normalizing.
The remaining classes correspond to different ways of defining the distances,
such as Manhattan or Euclidean distance.

Unknown values are treated correctly only by Euclidean and Relief distance.
For other measure of distance, a distance between unknown and known or between
two unknown values is always 0.5.

.. class:: ExamplesDistance

    .. method:: __call__(instance1, instance2)

        Returns a distance between the given instances as floating point number.

.. class:: ExamplesDistanceConstructor

    .. method:: __call__([instances, weightID][, distributions][, basic_var_stat])

        Constructs an instance of ExamplesDistance.
        Not all the data needs to be given. Most measures can be constructed
        from basic_var_stat; if it is not given, they can help themselves
        either by instances or distributions.
        Some (e.g. ExamplesDistance_Hamming) even do not need any arguments.

.. class:: ExamplesDistance_Normalized

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

    .. attribute:: domainVersion

        Stores a domain version for which the normalizers were computed.
        The domain version is increased each time a domain description is
        changed (i.e. features are added or removed); this is used for a quick
        check that the user is not attempting to measure distances between
        instances that do not correspond to normalizers.
        Since domains are practicably immutable (especially from Python),
        you don't need to care about this anyway.

    .. method:: attributeDistances(instance1, instance2)

        Returns a list of floats representing distances between pairs of
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
    ExamplesDistance_Normalized.attributeDistances,
    then Maximal returns max(dist).


.. class:: ManhattanConstructor
.. class:: Manhattan

    Manhattan distance between two instances is a sum of absolute values
    of distances between pairs of features, e.g. ``apply(add, [abs(x) for x in dist])``
    where dist is the result of ExamplesDistance_Normalized.attributeDistances.

.. class:: EuclideanConstructor
.. class:: Euclidean

    Euclidean distance is a square root of sum of squared per-feature distances,
    i.e. ``sqrt(apply(add, [x*x for x in dist]))``, where dist is the result of
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
        - A known and unknown discrete attribute equals the probabilit
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
