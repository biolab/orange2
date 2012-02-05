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
