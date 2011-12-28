""" 
.. index:: Binary Relevance Learner

***************************************
Binary Relevance Learner
***************************************

The most common problem transformation method is Binary Relevance method. 
It learns :math:`|L|` binary classifiers :math:`H_l:X \\rightarrow \{l,-l\}`, 
one for each different label :math:`l` in :math:`L`.
It transforms the original data set into :math:`|L|` data sets :math:`D_l` 
that contain all examples of the original data set, labelled as
:math:`l` if the labels of the original example contained :math:`l` and 
as :math:`\\neg l` otherwise. It is the same solution used in order 
to deal with a single-label multi-class problem using a binary classifier. 
For more information, see G. Tsoumakas and I. Katakis. `Multi-label classification: An overview 
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.9401&rep=rep1&type=pdf>`_. 
International Journal of Data Warehousing and Mining, 3(3):1-13, 2007.

Note that a copy of the table is made for each label to enable construction of
a classifier. Due to technical limitations, that is currently unavoidable and
should be remedied in Orange 3.

.. index:: Binary Relevance Learner
.. autoclass:: Orange.multilabel.BinaryRelevanceLearner
   :members:
   :show-inheritance:
 
   .. method:: __new__(instances, base_learner, **argkw) 
   BinaryRelevanceLearner Constructor
   
   :param instances: a table of instances.
   :type instances: :class:`Orange.data.Table`
      
   :param base_learner: the binary learner, the default learner is BayesLearner
   :type base_learner: :class:`Orange.classification.Learner`

.. index:: Binary Relevance Classifier
.. autoclass:: Orange.multilabel.BinaryRelevanceClassifier
   :members:
   :show-inheritance:

   .. method:: __call__(self, example, result_type)
   :rtype: a list of :class:`Orange.data.Value`, 
              a list of :class:`Orange.statistics.Distribution` or a tuple
              with both 
   
Examples
========

The following example demonstrates a straightforward invocation of
this algorithm (`mlc-classify.py`_, uses `multidata.tab`_):

.. literalinclude:: code/mlc-classify.py
   :lines: 1-13

.. _mlc-classify.py: code/mlc-br-example.py
.. _multidata.tab: code/multidata.tab

"""

import Orange
from Orange.classification.bayes import NaiveLearner as _BayesLearner
import multibase as _multibase

class BinaryRelevanceLearner(_multibase.MultiLabelLearner):
    """
    Class that implements the Binary Relevance (BR) method. 
    """
    def __new__(cls, instances = None, base_learner = None, weight_id = 0, **argkw):
        self = _multibase.MultiLabelLearner.__new__(cls, **argkw)
        if base_learner:
            self.base_learner = base_learner
        else:
            self.base_learner = _BayesLearner
        
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances, weight_id)
        else:
            return self
        
    def __call__(self, instances, weight_id = 0, **kwds):
        if not Orange.multilabel.is_multilabel(instances):
            raise TypeError("The given data set is not a multi-label data set.")
        
        for k in kwds.keys():
            self.__dict__[k] = kwds[k]

        classifiers = []
            
        for c in instances.domain.class_vars:
            new_domain = Orange.data.Domain(instances.domain.attributes, c)
            
            #build the instances
            new_table = Orange.data.Table(new_domain, instances)
            classifer = self.base_learner(new_table)
            classifiers.append(classifer)
            
        #Learn from the given table of data instances.
        return BinaryRelevanceClassifier(instances = instances, 
                                         classifiers = classifiers,
                                         weight_id = weight_id)

class BinaryRelevanceClassifier(_multibase.MultiLabelClassifier):
    def __init__(self, **kwds):
        self.multi_flag = 1
        self.__dict__.update(kwds)
        
    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue):
        domain = self.instances.domain
        labels = []
        dists = []
        
        for c in self.classifiers:
            v, p = c(instance, Orange.classification.Classifier.GetBoth)
                
            labels.append(v)
            dists.append(p)
            
        if result_type == Orange.classification.Classifier.GetValue:
            return labels
        if result_type == Orange.classification.Classifier.GetProbabilities:
            return dists
        return labels, dists
        
#########################################################################################
# A quick test/example.

if __name__ == "__main__":
    data = Orange.data.Table("emotions.tab")

    classifier = Orange.multilabel.BinaryRelevanceLearner(data,Orange.classification.knn.kNNLearner)
    for i in range(10):
        c,p = classifier(data[i],Orange.classification.Classifier.GetBoth)
        print c,p