import math
import random

import Orange.core as orange
import Orange

class BaggedLearner(orange.Learner):
    """
    BaggedLearner takes a learner and returns a bagged learner, which is 
    essentially a wrapper around the learner passed as an argument. If 
    instances are passed in arguments, BaggedLearner returns a bagged 
    classifier. Both learner and classifier then behave just like any 
    other learner and classifier in Orange.

    Bagging, in essence, takes training data and a learner, and builds *t* 
    classifiers, each time presenting a learner a bootstrap sample from the 
    training data. When given a test instance, classifiers vote on class, 
    and a bagged classifier returns a class with the highest number of votes. 
    As implemented in Orange, when class probabilities are requested, these 
    are proportional to the number of votes for a particular class.
    
    :param learner: learner to be bagged.
    :type learner: :class:`Orange.core.Learner`
    :param t: number of bagged classifiers, that is, classifiers created
        when instances are passed to bagged learner.
    :type t: int
    :param name: name of the resulting learner.
    :type name: str
    :rtype: :class:`Orange.ensemble.bagging.BaggedClassifier` or 
            :class:`Orange.ensemble.bagging.BaggedLearner`
    """
    def __new__(cls, learner, instances=None, weight_id=None, **kwargs):
        self = orange.Learner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(self, learner, **kwargs)
            return self.__call__(instances, weight_id)
        else:
            return self
        
    def __init__(self, learner, t=10, name='Bagging'):
        self.t = t
        self.name = name
        self.learner = learner

    def __call__(self, instances, weight=0):
        """
        Learn from the given table of data instances.
        
        :param instances: data instances to learn from.
        :type instances: Orange.data.Table
        :param weight: ID of meta feature with weights of instances
        :type weight: int
        :rtype: :class:`Orange.ensemble.bagging.BaggedClassifier`
        
        """
        r = random.Random()
        r.seed(0)
        
        n = len(instances)
        classifiers = []
        for i in range(self.t):
            selection = []
            for i in range(n):
                selection.append(r.randrange(n))
            instances = Orange.data.Table(instances)
            data = instances.getitems(selection)
            classifiers.append(self.learner(data, weight))
        return BaggedClassifier(classifiers = classifiers, name=self.name,\
                    class_var=instances.domain.class_var)
BaggedLearner = Orange.misc.deprecated_members({"weightId":"weight_id", "examples":"instances"})(BaggedLearner)

class BaggedClassifier(orange.Classifier):
    """
    A classifier that uses a bagging technique. Usually the learner
    (:class:`Orange.ensemble.bagging.BaggedLearner`) is used to construct the
    classifier.
    
    When constructing the classifier manually, the following parameters can
    be passed:

    :param classifiers: a list of boosted classifiers.
    :type classifiers: list
    
    :param name: name of the resulting classifier.
    :type name: str
    
    :param class_var: the class feature.
    :type class_var: :class:`Orange.feature.Descriptor`

    """

    def __init__(self, classifiers, name, class_var, **kwds):
        self.classifiers = classifiers
        self.name = name
        self.class_var = class_var
        self.__dict__.update(kwds)

    def __call__(self, instance, result_type = orange.GetValue):
        """
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        if self.class_var.var_type == Orange.feature.Type.Discrete:
            freq = [0.] * len(self.class_var.values)
            for c in self.classifiers:
                freq[int(c(instance))] += 1
            index = freq.index(max(freq))
            value = Orange.data.Value(self.class_var, index)
            if result_type == orange.GetValue:
                return value
            for i in range(len(freq)):
                freq[i] = freq[i]/len(self.classifiers)
            freq = Orange.statistics.distribution.Discrete(freq)
            if result_type == orange.GetProbabilities:
                return freq
            elif result_type == orange.GetBoth:
                return (value, freq)
            else:
                return value
            
        elif self.class_var.var_type ==Orange.feature.Type.Continuous:
            votes = [c(instance, orange.GetBoth if result_type==\
                orange.GetProbabilities else result_type) \
                for c in self.classifiers]
            wsum = float(len(self.classifiers))
            if result_type in [orange.GetBoth, orange.GetProbabilities]:
                pred = sum([float(c) for c, p in votes]) / wsum
#               prob = sum([float(p.modus()) for c, p in votes]) / wsum
                from collections import defaultdict
                prob = defaultdict(float)
                for c, p in votes:
                    for val, val_p in p.items():
                        prob[float(val)] += val_p / wsum
                    
                prob = Orange.statistics.distribution.Continuous(prob)
                return (self.class_var(pred), prob) if result_type == orange.GetBoth\
                    else prob
            elif result_type == orange.GetValue:
                pred = sum([float(c) for c in votes]) / wsum
                return self.class_var(pred)
            
    def __reduce__(self):
        return type(self), (self.classifiers, self.name, self.class_var), dict(self.__dict__)
BaggedClassifier = Orange.misc.deprecated_members({"example":"instance", "classVar":"class_var","resultType":"result_type"})(BaggedClassifier)
