import Orange

#%s/ExamplesDistanceConstructor/DistanceConstructor/gc
#%s/ExamplesDistance_Normalized/DistanceNormalized/gc
#ExampleDistance -> Distance
#Hamming -> HammingDistance
#DTW -> DTWDistance
#Euclidean -> EuclideanDistance
#Manhattan -> ...
#Maximal -> ...
#Relief -> ..
#DTWConstructor
#EuclideanConstructor
#HammingConstructor
#ManhattanConstructor
#MaximalConstructor
#ReliefConstructor
#PearsonRConstructor -> PearsonR
#PearsonR -> PearsonRDistance
#SpearmanRConstructor -> SpearmanR
#SpearmanR -> SpearmanRDistance
#MahalanobisConstructor ->  Mahalanobis
#Mahalanobis -> MahalanobisDistance

from Orange.core import \
    DistanceMap, \
    DistanceMapConstructor, \
    ExamplesDistance as Distance, \
    ExamplesDistance_Normalized as DistanceNormalized, \
    ExamplesDistanceConstructor as DistanceConstructor, \
    ExamplesDistance_Hamming as HammingDistance, \
    ExamplesDistance_DTW as DTWDistance, \
    ExamplesDistance_Euclidean as EuclideanDistance, \
    ExamplesDistance_Manhattan as ManhattanDistance, \
    ExamplesDistance_Maximal as MaximalDistance, \
    ExamplesDistance_Relief as ReliefDistance, \
    ExamplesDistanceConstructor_DTW as DTW, \
    ExamplesDistanceConstructor_Euclidean as Euclidean, \
    ExamplesDistanceConstructor_Hamming as Hamming, \
    ExamplesDistanceConstructor_Manhattan as Manhattan, \
    ExamplesDistanceConstructor_Maximal as Maximal, \
    ExamplesDistanceConstructor_Relief as Relief

from Orange import statc
import numpy
from numpy import linalg

class PearsonR(DistanceConstructor):
    """Constructs an instance of :obj:`PearsonRDistance`. Not all the data needs to be given."""
    
    def __new__(cls, data=None, **argkw):
        self = DistanceConstructor.__new__(cls, **argkw)
        self.__dict__.update(argkw)
        if data:
            return self.__call__(data)
        else:
            return self

    def __call__(self, table):
        indxs = [i for i, a in enumerate(table.domain.attributes) \
                 if a.varType==Orange.data.Type.Continuous]
        return PearsonRDistance(domain=table.domain, indxs=indxs)

class PearsonRDistance(Distance):
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

class SpearmanR(DistanceConstructor):
    """Constructs an instance of SpearmanR. Not all the data needs to be given."""
    
    def __new__(cls, data=None, **argkw):
        self = DistanceConstructor.__new__(cls, **argkw)
        self.__dict__.update(argkw)
        if data:
            return self.__call__(data)
        else:
            return self

    def __call__(self, table):
        indxs = [i for i, a in enumerate(table.domain.attributes) \
                 if a.varType==Orange.data.Type.Continuous]
        return SpearmanRDistance(domain=table.domain, indxs=indxs)

class SpearmanRDistance(Distance):  

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

class Mahalanobis(DistanceConstructor):
    """ Construct instance of Mahalanobis. """
    
    def __new__(cls, data=None, **argkw):
        self = DistanceConstructor.__new__(cls, **argkw)
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
        
        return MahalanobisDistance(domain=newdomain, icm=inverse_covariance_matrix)

class MahalanobisDistance(Distance):
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
    
    
class PearsonRAbsolute(PearsonR):
    """ Construct an instance of PearsonRAbsolute example distance estimator.
    """
    def __call__(self, data):
        indxs = [i for i, a in enumerate(data.domain.attributes) \
                 if a.varType==Orange.data.Type.Continuous]
        return PearsonRAbsoluteDistance(domain=data.domain, indxs=indxs)
    
    
class PearsonRAbsoluteDistance(PearsonRDistance):
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
        
        
class SpearmanRAbsolute(SpearmanR):
    """ Construct an instance of SpearmanRAbsolute example distance estimator.
    """
    def __call__(self, data):
        indxs = [i for i, a in enumerate(data.domain.attributes) \
                 if a.varType==Orange.data.Type.Continuous]
        return SpearmanRAbsoluteDistance(domain=data.domain, indxs=indxs)
    
    
class SpearmanRAbsoluteDistance(SpearmanRDistance):
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
    
    :param distance_constructor: An DistanceConstructor instance.
    :type distance_constructor: :obj:`Orange.distances.DistanceConstructor`
    
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
