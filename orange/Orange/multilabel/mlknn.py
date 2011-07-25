""" 
.. index:: ML-kNN Learner
   
.. index:: 
   single: ML-kNN;  ML-kNN Learner

***************************************
ML-kNN Learner
***************************************

ML-kNN Classification is a kind of adaptation method for multi-label classification. 
It is an adaptation of the kNN lazy learning algorithm for multi-label data. In essence,
ML-kNN uses the kNN algorithm independently for each label :math:'l': It finds the k nearest 
examples to the test instance and considers those that are labelled at least with :math:'l' 
as positive and the rest as negative. Actually this method follows the paradigm of Binary Relevance (BR).
What mainly differentiates this method from BR is the use of prior probabilities. ML-kNN has also
the capability of producing a ranking of the labels as an output.
For more information, see Zhang, M. and Zhou, Z. 2007. `ML-KNN: A lazy learning approach to multi-label learning <http://dx.doi.org/10.1016/j.patcog.2006.12.019>`_. 
Pattern Recogn. 40, 7 (Jul. 2007), 2038-2048.  

.. index:: ML-kNN Learner
.. autoclass:: Orange.multilabel.MLkNNLearner
   :members:
   :show-inheritance:
 
   .. method:: __new__(instances, base_learner, **argkw) 
   MLkNNLearner Constructor
   
   :param instances: a table of instances, covered by the rule.
   :type instances: :class:`Orange.data.Table`
      
   :param base_learner: the binary learner, the default learner is BayesLearner
   :type base_learner: :class:`Orange.classification.Learner`

.. index:: MLkNN Classifier
.. autoclass:: Orange.multilabel.MLkNNClassifier
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
   :lines: 1-3, 24-27

.. _mlc-classify.py: code/mlc-example.py
.. _multidata.tab: code/multidata.tab

"""

import Orange
import label
import multibase as _multibase

class MLkNNLearner(_multibase.MultiLabelLearner):
    """
    Class that implements the LabelPowerset (LP) method. 
    """
    def __new__(cls, instances = None, **argkw):
        self = _multibase.MultiLabelLearner.__new__(cls, **argkw)
                
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances)
        else:
            return self
                
    def __call__(self, instances, **kwds):
        for k in kwds.keys():
            self.__dict__[k] = kwds[k]

        num_labels = label.get_num_labels(instances)
        label_indices = label.get_label_indices(instances)
        
        #Learn from the given table of data instances.
        return MLkNNClassifier(instances = instances, label_indices = label_indices)

class MLkNNClassifier(_multibase.MultiLabelClassifier):      
    def __call__(self, example, result_type=Orange.classification.Classifier.GetValue):
        num_labels = len(self.label_indices)
        domain = self.instances.domain
        labels = []
        prob = []
        if num_labels == 0:
            raise ValueError, "has no label attribute: 'the multilabel data should have at last one label attribute' "
        
        c,p = self.classifier(example,Orange.classification.Classifier.GetBoth)
        str = c.value
        for i in range(len(str)):
            if str[i] == '0':
                labels.append(Orange.data.Value(domain[self.label_indices[i]],'0'))
                prob.append(0.0)
            elif str[i] == '1':
                labels.append(Orange.data.Value(domain[self.label_indices[i]],'1'))
                prob.append(1.0)
            else:
                raise ValueError, "invalid label value: 'the label value in instances should be only 0 or 1' "
        
        disc = Orange.statistics.distribution.Discrete(prob)
        disc.variable = Orange.core.EnumVariable(values = [domain[val].name for index,val in enumerate(self.label_indices)])
        
        if result_type == Orange.classification.Classifier.GetValue:
            return labels
        if result_type == Orange.classification.Classifier.GetProbabilities:
            return disc
        return labels,disc
        