"""
.. index:: outlier detection

.. index::
   single: outlier; detection

********************************
Outlier detection (``outliers``)
********************************

.. autoclass:: OutlierDetection
    :members:

.. rubric:: Examples

The following example prints a list of Z-values of examples in bridges dataset
(:download:`outlier1.py <code/outlier1.py>`).

.. literalinclude:: code/outlier1.py

The following example prints 5 examples with highest Z-scores. Euclidean
distance is used as a distance measurement and average distance is calculated
over 3 nearest neighbours (:download:`outlier2.py <code/outlier2.py>`).

.. literalinclude:: code/outlier2.py

The output::

    ['M', 1838, 'HIGHWAY', ?, 2, 'N', 'THROUGH', 'WOOD', '?', 'S', 'WOOD'] Z-score: 1.732
    ['M', 1818, 'HIGHWAY', ?, 2, 'N', 'THROUGH', 'WOOD', 'SHORT', 'S', 'WOOD'] Z-score: 1.732
    ['A', 1853, 'RR', ?, 2, 'N', 'DECK', 'WOOD', '?', 'S', 'WOOD'] Z-score: 1.732
    ['A', 1829, 'AQUEDUCT', ?, 1, 'N', 'THROUGH', 'WOOD', '?', 'S', 'WOOD'] Z-score: 1.733
    ['A', 1848, 'AQUEDUCT', ?, 1, 'N', 'DECK', 'WOOD', '?', 'S', 'WOOD'] Z-score: 1.733

"""

import Orange
import statc

class OutlierDetection:
    """
    A class for detecting outliers.

    It calculates average distances of each example to other examples
    and converts them to Z-scores. Z-scores higher than zero denote an 
    example that is more distant to other examples than average.

    Detection of outliers can be performed directly on examples or on 
    an existant distance matrix. Also, the number of nearest neighbours
    used for averaging distances can be set. The default 0 means 
    that all examples are used when calculating average distances.
    """
 
    def __init__(self):
        self._clear()
        self.set_knn()
  
    def _clear(self):
        #distmatrix not calculated yet
        self.distmatrixC = 0
        
        #using distance measurment
        self.distance = None
        
        self.examples = None
        self.distmatrix = None
            
    def set_examples(self, examples, distance = None):
        """Set examples on which the outlier detection will be
        performed. Distance is a distance constructor for distances
        between examples. If omitted, Manhattan distance is used."""
        self._clear()
        self.examples = examples
        if (distance == None):
          distance = Orange.distance.instances.ManhattanConstructor(self.examples)
        self.distance = distance

    def set_distance_matrix(self, distances):
        """Set the distance matrix on which the outlier detection 
        will be performed.
        """
        self._clear()
        self.distmatrix = distances
        self.distmatrixC = 1

    def set_knn(self, knn=0):
        """
        Set the number of nearest neighbours considered in determinating.
        """
        self.knn = knn
  
    def _calc_distance_matrix(self):
        """
        other distance measures
        """
        self.distmatrix = Orange.core.SymMatrix(len(self.examples)) #FIXME 
        for i in range(len(self.examples)):
            for j in range(i+1):
                self.distmatrix[i, j] = self.distance(self.examples[i],
                                                      self.examples[j])
        self.distmatrixC = 1
      
    def distance_matrix(self):
        """
        Return the distance matrix of the dataset.
        """
        if (self.distmatrixC == 0): 
            self._calc_distance_matrix()
        return self.distmatrix
       
    def _average_means(self):
        means = []
        dm = self.distance_matrix()
        for i,dist in enumerate(dm):
            nearest = self._find_nearest_limited(i, dist, self.knn)
            means.append(statc.mean(nearest))
        return means
      
    def _find_nearest_limited(self, i, dist, knn):
        copy = []
        for el in dist:
            copy.append(el)
        #remove distance to same element
        copy[i:i+1] = []
        if (knn == 0):
            return copy
        else:
            takelimit = min(len(dist)-1, knn)
            copy.sort()
            return copy[:takelimit]
        
    def z_values(self):
        """ Return a list of Z values of average distances for each element 
        to others. N-th number in the list is the Z-value of N-th example. 
        """
        list = self._average_means()
        return [statc.z(list, e) for e in list]

Orange.misc.deprecated_members(
    {"setKNN": "set_knn", 
    "setExamples": "set_examples", 
    "setDistanceMatrix": "set_distance_matrix", 
    "distanceMatrix": "distance_matrix", 
    "zValues": "z_values"
    })(OutlierDetection)

