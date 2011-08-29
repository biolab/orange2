""" 
.. index:: LabelPowerset Learner

***************************************
LabelPowerset Learner
***************************************

LabelPowerset Classification is another transformation method for multi-label classification. 
It considers each different set of labels that exist in the multi-label data as a 
single label. It so learns one single-label classifier :math:`H:X \\rightarrow P(L)`, where
:math:`P(L)` is the power set of L.
For more information, see G. Tsoumakas and I. Katakis. `Multi-label classification: An overview 
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.9401&rep=rep1&type=pdf>`_. 
International Journal of Data Warehousing and Mining, 3(3):1-13, 2007.

.. index:: LabelPowerset Learner
.. autoclass:: Orange.multilabel.LabelPowersetLearner
   :members:
   :show-inheritance:
 
   .. method:: __new__(instances, base_learner, **argkw) 
   LabelPowersetLearner Constructor
   
   :param instances: a table of instances, covered by the rule.
   :type instances: :class:`Orange.data.Table`
      
   :param base_learner: the binary learner, the default learner is BayesLearner
   :type base_learner: :class:`Orange.classification.Learner`

.. index:: LabelPowerset Classifier
.. autoclass:: Orange.multilabel.LabelPowersetClassifier
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
   :lines: 1-3, 15-22

.. _mlc-classify.py: code/mlc-example.py
.. _multidata.tab: code/multidata.tab

"""

import Orange
from Orange.core import BayesLearner as _BayesLearner
import label
import multibase as _multibase

class LabelPowersetLearner(_multibase.MultiLabelLearner):
    """
    Class that implements the LabelPowerset (LP) method. 
    """
    def __new__(cls, instances = None, base_learner = None, weight_id = 0, **argkw):
        self = _multibase.MultiLabelLearner.__new__(cls, **argkw)
        if base_learner:
            self.base_learner = base_learner
        else:
            self.base_learner = _BayesLearner
        
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances,base_learner,weight_id)
        else:
            return self
                
    def __call__(self, instances, base_learner = None, weight_id = 0, **kwds):
        for k in kwds.keys():
            self.__dict__[k] = kwds[k]

        num_labels = label.get_num_labels(instances)
        label_indices = label.get_label_indices(instances)
        
        #abtain the labels and use a string to represent it and store the classvalues
        new_class = Orange.data.variable.Discrete("label")
        
        for e in instances:
            class_value = label.get_label_bitstream(instances,e)
            new_class.add_value(class_value)
        
        #remove the label attributes
        indices_remove = [var for index, var in enumerate(label_indices)]
        new_domain = label.remove_indices(instances,indices_remove)
        
        #add the class attribute
        new_domain = Orange.data.Domain(new_domain,new_class)
        
        #build the instances
        new_table = Orange.data.Table(new_domain)
        for e in instances:
            new_row = Orange.data.Instance(
              new_domain, 
              [v.value for v in e if v.variable.attributes.has_key('label') <> 1] +
                    [label.get_label_bitstream(instances,e)])
            
            new_table.append(new_row)
             
        #store the classifier
        classifier = self.base_learner(new_table)
        
        #Learn from the given table of data instances.
        return LabelPowersetClassifier(instances = instances, 
                                       label_indices = label_indices,
                                       classifier = classifier,
                                       weight_id = weight_id)

class LabelPowersetClassifier(_multibase.MultiLabelClassifier):      
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
                #raise ValueError, "invalid label value: 'the label value in instances should be only 0 or 1' "
                labels.append(Orange.data.Value(domain[self.label_indices[i]],'?'))
                prob.append(0.0)
        
        disc = Orange.statistics.distribution.Discrete(prob)
        disc.variable = Orange.core.EnumVariable(values = [domain[val].name for index,val in enumerate(self.label_indices)])
        
        if result_type == Orange.classification.Classifier.GetValue:
            return labels
        if result_type == Orange.classification.Classifier.GetProbabilities:
            return disc
        return labels,disc

#########################################################################################
# Test the code, run from DOS prompt
# assume the data file is in proper directory

if __name__ == "__main__":
    data = Orange.data.Table("emotions.tab")

    classifier = Orange.multilabel.LabelPowersetLearner(data)
    for i in range(10):
        c,p = classifier(data[i],Orange.classification.Classifier.GetBoth)
        print c,p