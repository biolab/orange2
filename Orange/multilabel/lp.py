""" 
.. index:: LabelPowerset Learner

***************************************
LabelPowerset Learner
***************************************

LabelPowerset Classification is another transformation method for multi-label classification. 
It considers each different set of labels that exists in the multi-label data as a 
single class. Thus it learns a classification problem :math:`H:X \\rightarrow \\mathbb{P}(L)`, where
:math:`\\mathbb{P}(L)` is the power set of L.
For more information, see G. Tsoumakas and I. Katakis. `Multi-label classification: An overview 
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.9401&rep=rep1&type=pdf>`_. 
International Journal of Data Warehousing and Mining, 3(3):1-13, 2007.

.. index:: LabelPowerset Learner
.. autoclass:: Orange.multilabel.LabelPowersetLearner
   :members:
   :show-inheritance:
 
   :param instances: a table of instances.
   :type instances: :class:`Orange.data.Table`
      
   :param base_learner: the binary learner, the default learner is BayesLearner
   :type base_learner: :class:`Orange.classification.Learner`

.. index:: LabelPowerset Classifier
.. autoclass:: Orange.multilabel.LabelPowersetClassifier
   :members:
   :show-inheritance:
   
Examples
========

The following example demonstrates a straightforward invocation of
this algorithm (:download:`mlc-classify.py <code/mlc-classify.py>`, uses
:download:`emotions.tab <code/emotions.tab>`):

.. literalinclude:: code/mlc-classify.py
   :lines: 6, 19-21

"""

import Orange
from Orange.core import BayesLearner as _BayesLearner
import multibase as _multibase

def get_label_bitstream(e):
    return ''.join(lv.value for lv in e.get_classes())

def transform_to_powerset(instances):
    new_class = Orange.feature.Discrete("label")
    
    for e in instances:
        class_value = get_label_bitstream(e)
        new_class.add_value(class_value)
    
    new_domain = Orange.data.Domain(instances.domain.attributes, new_class)
    
    #build the instances
    new_table = Orange.data.Table(new_domain)
    for e in instances:
        new_row = Orange.data.Instance(
          new_domain,
          [e[a].value for a in instances.domain.attributes] +
          [get_label_bitstream(e)])
        
        new_table.append(new_row)
    
    return new_table

class LabelPowersetLearner(_multibase.MultiLabelLearner):
    """
    Class that implements the LabelPowerset (LP) method. 
    """
    def __new__(cls, instances = None, base_learner = None, weight_id = 0, **argkw):
        self = _multibase.MultiLabelLearner.__new__(cls, **argkw)
        
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances, base_learner, weight_id)
        else:
            return self
                
    def __call__(self, instances, base_learner = None, weight_id = 0, **kwds):
        if not Orange.multilabel.is_multilabel(instances):
            raise TypeError("The given data set is not a multi-label data set.")

        self.__dict__.update(kwds)

        new_table = transform_to_powerset(instances)
        
        #store the classifier
        base_learner = base_learner if base_learner else _BayesLearner
        classifier = base_learner(new_table)
        
        #Learn from the given table of data instances.
        return LabelPowersetClassifier(instances = instances, 
                                       classifier = classifier,
                                       weight_id = weight_id)

class LabelPowersetClassifier(_multibase.MultiLabelClassifier):
    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue):
        """
        :rtype: a list of :class:`Orange.data.Value`, a list of :class:`Orange.statistics.distribution.Distribution`, or a tuple with both
        """
        labels = []
        prob = []
        
        c = self.classifier(instance)
        for bit, lvar in zip(c.value, self.instances.domain.class_vars):
            labels.append(Orange.data.Value(lvar, bit))
            prob.append(float(bit == '1'))
        
        if result_type == Orange.classification.Classifier.GetValue:
            return labels
        
        disc = [Orange.statistics.distribution.Discrete([1-p, p]) for p in prob]
        for v, d in zip(self.instances.domain.class_vars, disc):
            d.variable = v
        
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
