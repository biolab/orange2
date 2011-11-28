"""

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


.. class:: Hamming, HammingConstructor

    Hamming distance between two instances is defined as the number of
    features in which the two instances differ. Note that this measure
    is not really appropriate for instances that contain continuous features.


.. class:: Maximal, MaximalConstructor

    The maximal between two instances is defined as the maximal distance
    between two feature values. If dist is the result of
    ExamplesDistance_Normalized.attributeDistances,
    then Maximal returns max(dist).


.. class:: Manhattan, ManhattanConstructor

    Manhattan distance between two instances is a sum of absolute values
    of distances between pairs of features, e.g. ``apply(add, [abs(x) for x in dist])``
    where dist is the result of ExamplesDistance_Normalized.attributeDistances.

.. class:: Euclidean, EuclideanConstructor


    Euclidean distance is a square root of sum of squared per-feature distances,
    i.e. ``sqrt(apply(add, [x*x for x in dist]))``, where dist is the result of
    ExamplesDistance_Normalized.attributeDistances.

    .. method:: distributions 

        An object of type
        :obj:`Orange.statistics.distribution.Distribution` that holds
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

.. class:: Relief, ReliefConstructor

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


"""

import Orange

from Orange.core import \
     AlignmentList, \
     DistanceMap, \
     DistanceMapConstructor, \
     ExampleDistConstructor, \
     ExampleDistBySorting, \
     ExampleDistVector, \
     ExamplesDistance, \
     ExamplesDistance_Normalized, \
     ExamplesDistanceConstructor

from Orange.core import ExamplesDistance_Hamming as Hamming
from Orange.core import ExamplesDistance_DTW as DTW
from Orange.core import ExamplesDistance_Euclidean as Euclidean
from Orange.core import ExamplesDistance_Manhattan as Manhattan
from Orange.core import ExamplesDistance_Maximal as Maximal
from Orange.core import ExamplesDistance_Relief as Relief

from Orange.core import ExamplesDistanceConstructor_DTW as DTWConstructor
from Orange.core import ExamplesDistanceConstructor_Euclidean as EuclideanConstructor
from Orange.core import ExamplesDistanceConstructor_Hamming as HammingConstructor
from Orange.core import ExamplesDistanceConstructor_Manhattan as ManhattanConstructor
from Orange.core import ExamplesDistanceConstructor_Maximal as MaximalConstructor
from Orange.core import ExamplesDistanceConstructor_Relief as ReliefConstructor

import statc
import numpy
from numpy import linalg

class PearsonRConstructor(ExamplesDistanceConstructor):
    """Constructs an instance of PearsonR. Not all the data needs to be given."""
    
    def __new__(cls, data=None, **argkw):
        self = ExamplesDistanceConstructor.__new__(cls, **argkw)
        self.__dict__.update(argkw)
        if data:
            return self.__call__(data)
        else:
            return self

    def __call__(self, table):
        indxs = [i for i, a in enumerate(table.domain.attributes) \
                 if a.varType==Orange.data.Type.Continuous]
        return PearsonR(domain=table.domain, indxs=indxs)

class PearsonR(ExamplesDistance):
    """
    `Pearson correlation coefficient
    <http://en.wikipedia.org/wiki/Pearson_product-moment\
    _correlation_coefficient>`_
    """

    def __init__(self, **argkw):
        self.__dict__.update(argkw)
        
    def __call__(self, e1, e2):
        """
        :param e1: data instances.
        :param e2: data instances.
        
        Returns Pearson's disimilarity between e1 and e2,
        i.e. (1-r)/2 where r is Sprearman's rank coefficient.
        """
        X1 = []
        X2 = []
        for i in self.indxs:
            if not(e1[i].isSpecial() or e2[i].isSpecial()):
                X1.append(float(e1[i]))
                X2.append(float(e2[i]))
        if not X1:
            return 1.0
        try:
            return (1.0 - statc.pearsonr(X1, X2)[0]) / 2.
        except:
            return 1.0

class SpearmanRConstructor(ExamplesDistanceConstructor):
    """Constructs an instance of SpearmanR. Not all the data needs to be given."""
    
    def __new__(cls, data=None, **argkw):
        self = ExamplesDistanceConstructor.__new__(cls, **argkw)
        self.__dict__.update(argkw)
        if data:
            return self.__call__(data)
        else:
            return self

    def __call__(self, table):
        indxs = [i for i, a in enumerate(table.domain.attributes) \
                 if a.varType==Orange.data.Type.Continuous]
        return SpearmanR(domain=table.domain, indxs=indxs)

class SpearmanR(ExamplesDistance):  

    """`Spearman's rank correlation coefficient
    <http://en.wikipedia.org/wiki/Spearman%27s_rank_\
    correlation_coefficient>`_"""

    def __init__(self, **argkw):
        self.__dict__.update(argkw)
        
    def __call__(self, e1, e2):
        """
        :param e1: data instances.
        :param e2: data instances.
        
        Returns Sprearman's disimilarity between e1 and e2,
        i.e. (1-r)/2 where r is Sprearman's rank coefficient.
        """
        X1 = []; X2 = []
        for i in self.indxs:
            if not(e1[i].isSpecial() or e2[i].isSpecial()):
                X1.append(float(e1[i]))
                X2.append(float(e2[i]))
        if not X1:
            return 1.0
        try:
            return (1.0 - statc.spearmanr(X1, X2)[0]) / 2.
        except:
            return 1.0

class MahalanobisConstructor(ExamplesDistanceConstructor):
    """ Construct instance of Mahalanobis. """
    
    def __new__(cls, data=None, **argkw):
        self = ExamplesDistanceConstructor.__new__(cls, **argkw)
        self.__dict__.update(argkw)
        if data:
            return self.__call__(data)
        else:
            return self
    
    # Check attributtes a, b, c
    def __call__(self, table, a=None, b=None, c=None, **argkw):
        # Process data
        dc = Orange.core.DomainContinuizer()
        dc.classTreatment = Orange.core.DomainContinuizer.Ignore
        dc.continuousTreatment = Orange.core.DomainContinuizer.NormalizeBySpan
        dc.multinomialTreatment = Orange.core.DomainContinuizer.NValues
        
        newdomain = dc(table)
        newtable = table.translate(newdomain)
        
        data, cls, _ = newtable.to_numpy()
        
        covariance_matrix = numpy.cov(data, rowvar=0, bias=1)
        inverse_covariance_matrix = linalg.pinv(covariance_matrix, rcond=1e-10)
        
        return Mahalanobis(domain=newdomain, icm=inverse_covariance_matrix)

class Mahalanobis(ExamplesDistance):
    """`Mahalanobis distance
    <http://en.wikipedia.org/wiki/Mahalanobis_distance>`_"""

    def __init__(self, domain, icm, **argkw):
        self.domain = domain
        self.icm = icm
        self.__dict__.update(argkw)
        
    def __call__(self, e1, e2):
        """
        :param e1: data instances.
        :param e2: data instances.
        
        Returns Mahalanobis distance between e1 and e2.
        """
        e1 = Orange.data.Instance(self.domain, e1)
        e2 = Orange.data.Instance(self.domain, e2)
        
        diff = []
        for i in range(len(self.domain.attributes)):
            diff.append(e1[i].value - e2[i].value) if not(e1[i].isSpecial() or e2[i].isSpecial()) else 0.0
        diff = numpy.asmatrix(diff)
        res = diff * self.icm * diff.transpose()
        return res[0,0]**0.5
    
    
class PearsonRAbsoluteConstructor(PearsonRConstructor):
    """ Construct an instance of PearsonRAbsolute example distance estimator.
    """
    def __call__(self, data):
        indxs = [i for i, a in enumerate(data.domain.attributes) \
                 if a.varType==Orange.data.Type.Continuous]
        return PearsonRAbsolute(domain=data.domain, indxs=indxs)
    
    
class PearsonRAbsolute(PearsonR):
    """ An example distance estimator using absolute value of Pearson
    correlation coefficient.
    """
    def __call__(self, e1, e2):
        """
        Return absolute Pearson's dissimilarity between e1 and e2,
        i.e.
        
        .. math:: (1 - abs(r))/2
        
        where r is Pearson's correlation coefficient.
        """
        X1 = []; X2 = []
        for i in self.indxs:
            if not(e1[i].isSpecial() or e2[i].isSpecial()):
                X1.append(float(e1[i]))
                X2.append(float(e2[i]))
        if not X1:
            return 1.0
        try:
            return (1.0 - abs(statc.pearsonr(X1, X2)[0]))
        except:
            return 1.0
        
        
class SpearmanRAbsoluteConstructor(SpearmanRConstructor):
    """ Construct an instance of SpearmanRAbsolute example distance estimator.
    """
    def __call__(self, data):
        indxs = [i for i, a in enumerate(data.domain.attributes) \
                 if a.varType==Orange.data.Type.Continuous]
        return SpearmanRAbsolute(domain=data.domain, indxs=indxs)
    
    
class SpearmanRAbsolute(SpearmanR):
    def __call__(self, e1, e2):
        """
        Return absolute Spearman's dissimilarity between e1 and e2,
        i.e.
         
        .. math:: (1 - abs(r))/2
        
        where r is Spearman's correlation coefficient.
        """
        X1 = []; X2 = []
        for i in self.indxs:
            if not(e1[i].isSpecial() or e2[i].isSpecial()):
                X1.append(float(e1[i]))
                X2.append(float(e2[i]))
        if not X1:
            return 1.0
        try:
            return (1.0 - abs(statc.spearmanr(X1, X2)[0]))
        except:
            return 1.0
    
    
def distance_matrix(data, distance_constructor, progress_callback=None):
    """ A helper function that computes an obj:`Orange.core.SymMatrix` of all
    pairwise distances between instances in `data`.
    
    :param data: A data table
    :type data: :obj:`Orange.data.Table`
    
    :param distance_constructor: An ExamplesDistance_Constructor instance.
    :type distance_constructor: :obj:`Orange.distances.ExampleDistConstructor`
    
    """
    from Orange.misc import progressBarMilestones as progress_milestones
    matrix = Orange.core.SymMatrix(len(data))
    dist = distance_constructor(data)
    
    msize = len(data)*(len(data) - 1)/2
    milestones = progress_milestones(msize, 100)
    count = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            matrix[i, j] = dist(data[i], data[j])
            
            if progress_callback and count in milestones:
                progress_callback(100.0 * count / msize)
            count += 1
            
    return matrix
