import Orange
from Orange.regression.earth import data_label_mask


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

        label_mask = data_label_mask(data.domain)
        if sum(label_mask) == 0:
            raise 'No classes/labels defined.'
        x_vars = [v for v, label in zip(data.domain, label_mask) if not label]
        y_vars = [v for v, label in zip(data.domain, label_mask) if label]

        classifiers = [self.learner(Orange.data.Table(Orange.data.Domain(x_vars, y),
            data), weight) for y in y_vars]
        return MultitargetClassifier(classifiers=classifiers, x_vars=x_vars, y_vars=y_vars)
        

class MultitargetClassifier(Orange.classification.Classifier):
    """
    Multitarget classifier returning a list of predictions from each
    of the independent base classifiers.

    .. attribute classifiers

        List of individual classifiers for each class.
    """

    def __init__(self, classifiers, x_vars, y_vars):
        self.classifiers = classifiers
        self.x_vars = x_vars
        self.y_vars = y_vars

    def __call__(self, instance, return_type=Orange.core.GetValue):
        predictions = [c(Orange.data.Instance(Orange.data.Domain(self.x_vars, y),
            instance), return_type) for c, y in zip(self.classifiers, self.y_vars)]
        return zip(*predictions) if return_type == Orange.core.GetBoth else predictions

