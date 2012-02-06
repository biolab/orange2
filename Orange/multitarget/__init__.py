"""
Wrapper for constructing multi-target learners
==============================================

This module also contains a wrapper, an auxilary learner, that can be used
to construct simple multi-target learners from standard learners designed
for data with a single class. The wrapper uses the specified base learner
to construct independent models for each class.

.. index:: MultitargetLearner
.. autoclass:: Orange.multitarget.MultitargetLearner
    :members:
    :show-inheritance:

.. index:: MultitargetClassifier
.. autoclass:: Orange.multitarget.MultitargetClassifier
    :members:
    :show-inheritance:

Examples
========

The following example uses a simple multi-target data set (generated with
:download:`generate_multitarget.py <code/generate_multitarget.py>`) to show
some basic functionalities (part of
:download:`multitarget.py <code/multitarget.py>`, uses
:download:`multitarget-synthetic.tab <code/multitarget-synthetic.tab>`).

.. literalinclude:: code/multitarget.py
    :lines: 1-6

Multi-target learners can be used to build prediction models (classifiers)
which then predict (multiple) class values for a new instance (continuation of
:download:`multitarget.py <code/multitarget.py>`):

.. literalinclude:: code/multitarget.py
    :lines: 8-

"""

import Orange

# Other algorithms which also work with multitarget data
from Orange.regression import pls
# change the default value of multi_label=True in init
##from Orange.regression import earth


class MultitargetLearner(Orange.classification.Learner):
    """
    Wrapper for multitarget problems that constructs independent models
    of a base learner for each class variable.

    .. attribute:: learner

        The base learner used to learn models for each class.
    """

    def __new__(cls, learner, data=None, weight=0, **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)
        if data:
            self.__init__(learner, **kwargs)
            return self.__call__(data, weight)
        else:
            return self
    
    def __init__(self, learner, **kwargs):
        """

        :param learner: Base learner used to construct independent
                        models for each class.
                        
        """

        self.learner = learner
        self.__dict__.update(kwargs)

    def __call__(self, data, weight=0):
        """
        Learn independent models of the base learner for each class.

        :param data: Multitarget data instances (with more than 1 class).
        :type data: :class:`Orange.data.Table`

        :param weight: Id of meta attribute with weights of instances
        :type weight: :obj:`int`

        :rtype: :class:`Orange.multitarget.MultitargetClassifier`
        """

        if not data.domain.class_vars:
            raise Exception('No classes defined.')
        
        domains = [Orange.data.Domain(data.domain.attributes, y)
                   for y in data.domain.class_vars]
        classifiers = [self.learner(Orange.data.Table(dom, data), weight)
                       for dom in domains]
        return MultitargetClassifier(classifiers=classifiers, domains=domains)
        

class MultitargetClassifier(Orange.classification.Classifier):
    """
    Multitarget classifier returning a list of predictions from each
    of the independent base classifiers.

    .. attribute classifiers

        List of individual classifiers for each class.
    """

    def __init__(self, classifiers, domains):
        self.classifiers = classifiers
        self.domains = domains

    def __call__(self, instance, return_type=Orange.core.GetValue):
        """
        :param instance: Instance to be classified.
        :type instance: :class:`Orange.data.Instance`

        :param return_type: One of
            :class:`Orange.classification.Classifier.GetValue`,
            :class:`Orange.classification.Classifier.GetProbabilities` or
            :class:`Orange.classification.Classifier.GetBoth`
        """

        predictions = [c(Orange.data.Instance(dom, instance), return_type)
                       for c, dom in zip(self.classifiers, self.domains)]
        return zip(*predictions) if return_type == Orange.core.GetBoth \
               else predictions

