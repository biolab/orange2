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
        self.learner = learner
        self.__dict__.update(kwargs)

    def __call__(self, data, weight=0):
        """Learn independent models of the base learner for each class.

        :param data: Multitarget data instances (with more than 1 class).
        :type data: Orange.data.Table
        :param weight: Id of meta attribute with weights of instances
        :type weight: int
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
        predictions = [c(Orange.data.Instance(dom, instance), return_type)
                       for c, dom in zip(self.classifiers, self.domains)]
        return zip(*predictions) if return_type == Orange.core.GetBoth \
               else predictions

