""" 
.. index:: Multi-class Multi-label Perceptron
   
.. index:: 
   single: multilabel;  Multi-class Multi-label Perceptron (MMP)

***************************************
Multi-class Multi-label Perceptron
***************************************

The multi-class multi-label perceptron (MMP) is a family of online algorithms for 
label ranking from multi-label data based on the perceptron algorithm. MMP maintains one perceptron 
for each label, but wight updates for each perceptron are performed so as to achieve a perfect ranking 
of all labels.

For more information, see G. Tsoumakas and I. Katakis. `Multi-label classification: An overview 
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.9401&rep=rep1&type=pdf>`_. 
International Journal of Data Warehousing and Mining, 3(3):1-13, 2007.

.. index:: Multi-class Multi-label Perceptron
.. autoclass:: Orange.multilabel.MMPLearner
   :members:
   :show-inheritance:
 
   .. method:: __new__(instances, **argkw) 
   MMPLearner Constructor
   
   :param instances: a table of instances, covered by the rule.
   :type instances: :class:`Orange.data.Table`

.. index:: MMP Classifier
.. autoclass:: Orange.multilabel.MMPClassifier
   :members:
   :show-inheritance:

   .. method:: __call__(self, example, result_type)
   :rtype: a list of :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both 
   
Examples
========

The following example demonstrates a straightforward invocation of
this algorithm (`mlc-classify.py`_, uses `multidata.tab`_):

.. literalinclude:: code/mlc-classify.py
   :lines: 1-3,29-32

.. _mlc-classify.py: code/mlc-br-example.py
.. _multidata.tab: code/multidata.tab

"""

import Orange
import label
import multibase as _multibase

class MMPLearner(_multibase.MultiLabelLearner):
    """
    Class implementing the MMP (Multi-class Multi-label Perceptron) algorithm. 
     
    .. attribute:: k
    
        Number of neighbors. If set to 0 (which is also the default value), 
        the square root of the number of instances is used.
    
    .. attribute:: smooth
    
        Smoothing parameter controlling the strength of uniform prior (Default value is set to 1 which yields the Laplace smoothing).
    
    .. attribute:: knn
        
        :class:`Orange.classification.knn.kNNLearner` for nearest neighbor search
    
    """
    def __new__(cls, instances = None, k=1, smooth = 1.0, **argkw):
        """
        Constructor of MLkNNLearner
        
        :param instances: a table of instances, covered by the rule.
        :type instances: :class:`Orange.data.Table`
        
        :param k: number of nearest neighbors used in classification
        :type k: int
        
        :param smooth: Smoothing parameter controlling the strength of uniform prior (Default value is set to 1 which yields the Laplace smoothing).
        :type smooth: Float
        
        :rtype: :class:`MLkNNLearner`
        """
        
        self = _multibase.MultiLabelLearner.__new__(cls, **argkw)
        
        self.k = k
        self.smooth = smooth
        
        if instances:
            self.instances = instances
            self.__init__(**argkw)
            return self.__call__(instances)
        else:
            return self

    def __call__(self, instances, **kwds):
        for k in kwds.keys():
            self.__dict__[k] = kwds[k]

        self.num_labels = label.get_num_labels(instances)
        self.label_indices = label.get_label_indices(instances)
        

        
        return MMPClassifier(instances = instances, label_indices = label_indices)

class MMPClassifier(_multibase.MultiLabelClassifier):      
    def __call__(self, example, result_type=Orange.classification.Classifier.GetValue):
        num_labels = len(self.label_indices)
        domain = self.instances.domain
        labels = []
        prob = []
        if num_labels == 0:
            raise ValueError, "has no label attribute: 'the multilabel data should have at last one label attribute' "
        
        disc = Orange.statistics.distribution.Discrete(prob)
        disc.variable = Orange.core.EnumVariable(
            values = [domain[val].name for index,val in enumerate(self.label_indices)])
        
        if result_type == Orange.classification.Classifier.GetValue:
            return labels
        if result_type == Orange.classification.Classifier.GetProbabilities:
            return disc
        return labels,disc
        