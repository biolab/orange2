""" 
.. index:: MultikNN Learner
   
.. index:: 
   single: multilabel;  MultikNN Learner

***************************************
MultikNN Learner
***************************************

MultikNN Classification is the base class of kNN method based multi-label
classification.

.. index:: MultikNN Learner
.. autoclass:: Orange.multilabel.MultikNNLearner
   :members:
   :show-inheritance:
 
   :param instances: a table of instances.
   :type instances: :class:`Orange.data.Table`

.. index:: MultikNN Classifier
.. autoclass:: Orange.multilabel.MultikNNClassifier
   :members:
   :show-inheritance:
   
"""
import random

import Orange
import multibase as _multibase

class MultikNNLearner(_multibase.MultiLabelLearner):
    """
    Class implementing the MultikNN (Multi-Label k Nearest Neighbours) algorithm. 
    
    .. attribute:: k
    
        Number of neighbors. The default value is 1 
    
    .. attribute:: num_labels
    
        Number of labels
    
    .. attribute:: label_indices
    
        The indices of labels in the domain 
    
    .. attribute:: knn
        
        :class:`Orange.classification.knn.FindNearest` for nearest neighbor search
    
    """
    def __new__(cls, k=1, **argkw):
        """
        Constructor of MultikNNLearner
                
        :param k: number of nearest neighbors used in classification
        :type k: int
        
        :rtype: :class:`MultikNNLearner`
        """
        
        self = _multibase.MultiLabelLearner.__new__(cls, **argkw)
        self.k = k
        return self
    
    def _build_knn(self, instances):
        nnc = Orange.classification.knn.FindNearestConstructor()
        nnc.distanceConstructor = Orange.core.ExamplesDistanceConstructor_Euclidean()
        
        weight_id = Orange.core.newmetaid()
        self.knn = nnc(instances, 0, weight_id)
        self.weight_id = weight_id

class MultikNNClassifier(_multibase.MultiLabelClassifier):
    pass
        