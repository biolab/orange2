""" 
.. index:: Binary Relevance Learner
   
.. index:: 
   single: multilabel;  Binary Relevance Learner

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

.. index:: Binary Relevance Learner
.. autoclass:: Orange.multilabel.BinaryRelevanceLearner
   :members:
   :show-inheritance:
 
   .. method:: __new__(instances, base_learner, **argkw) 
   BinaryRelevanceLearner Constructor
   
   :param instances: a table of instances, covered by the rule.
   :type instances: :class:`Orange.data.Table`
      
   :param base_learner: the binary learner, the default learner is BayesLearner
   :type base_learner: :class:`Orange.classification.Learner`

.. index:: Binary Relevance Classifier
.. autoclass:: Orange.multilabel.BinaryRelevanceClassifier
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
   :lines: 1-13

.. _mlc-classify.py: code/mlc-br-example.py
.. _multidata.tab: code/multidata.tab

"""

import Orange
from Orange.core import BayesLearner as _BayesLearner
import label
import multibase as _multibase

class BinaryRelevanceLearner(_multibase.MultiLabelLearner):
    """
    Class that implements the Binary Relevance (BR) method. 
    """
    def __new__(cls, instances = None, base_learner = None, **argkw):
        self = _multibase.MultiLabelLearner.__new__(cls, **argkw)
        if base_learner:
            self.base_learner = base_learner
        else:
            self.base_learner = _BayesLearner
        
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances,base_learner)
        else:
            return self
        
    def __call__(self, instances, base_learner = None, **kwds):
        for k in kwds.keys():
            self.__dict__[k] = kwds[k]

        num_labels = label.get_num_labels(instances)
        label_indices = label.get_label_indices(instances)
        
        classifiers = []
            
        for i in range(num_labels):
            # Indices of attributes to remove
            indices_remove = [var for index, var in enumerate(label_indices) if var <> label_indices[i]]
            sub_domain = label.remove_indices(instances,indices_remove)
            
            class_index = 0
            for index,var in enumerate(sub_domain):
                if var == instances.domain[ label_indices[i]]:
                     class_index = index
            
            #set the class label
            sub_domain = Orange.data.Domain(sub_domain,sub_domain[class_index])
           
            sub_instances = instances.translate(sub_domain)
                   
            #store the classifier
            classifer = self.base_learner(sub_instances)
            
            classifiers.append(classifer)
            
        #Learn from the given table of data instances.
        return BinaryRelevanceClassifier(instances = instances, label_indices = label_indices,classifiers = classifiers)

class BinaryRelevanceClassifier(_multibase.MultiLabelClassifier):
    def __init__(self, **kwds):
        self.multi_flag = 1
        self.__dict__.update(kwds)
        
    def __call__(self, example, result_type=Orange.classification.Classifier.GetValue):
        num_labels = len(self.label_indices)
        domain = self.instances.domain
        labels = []
        prob = []
        if num_labels == 0:
            raise ValueError, "has no label attribute: 'the multilabel data should have at last one label attribute' "
        
        for i in range(num_labels):
            c,p = self.classifiers[i](example,Orange.classification.Classifier.GetBoth)
            #get the index of label value that = 1, so as to locate the index of label in prob 
            label_index = -1
            values = domain[ self.label_indices[i] ].values
            if len(values) > 2:
                raise ValueError, "invalid label value: 'the label value in instances should be only 0 or 1' "
            
            if values[0] == '1' :
                label_index = 0
            elif values[1] == '1':
                label_index = 1
            else:
                raise ValueError, "invalid label value: 'the label value in instances should be only 0 or 1' "
                
            prob.append(p[label_index])
            labels.append(c)
            
            disc = Orange.statistics.distribution.Discrete(prob)
            disc.variable = Orange.core.EnumVariable(values = [domain[val].name for index,val in enumerate(self.label_indices)])
            
        if result_type == Orange.classification.Classifier.GetValue:
            return labels
        if result_type == Orange.classification.Classifier.GetProbabilities:
            return disc
        return labels,disc
        