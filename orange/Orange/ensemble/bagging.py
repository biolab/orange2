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
    def __new__(cls, learner, instances=None, weightId=None, **kwargs):
        self = orange.Learner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(self, learner, **kwargs)
            return self.__call__(instances, weightId)
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
                    classVar=instances.domain.classVar)

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
    
    :param classVar: the class feature.
    :type classVar: :class:`Orange.data.variable.Variable`

    """

    def __init__(self, classifiers, name, classVar, **kwds):
        self.classifiers = classifiers
        self.name = name
        self.classVar = classVar
        self.__dict__.update(kwds)

    def __call__(self, instance, resultType = orange.GetValue):
        """
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        if self.classVar.varType == Orange.data.Type.Discrete:
            freq = [0.] * len(self.classVar.values)
            for c in self.classifiers:
                freq[int(c(instance))] += 1
            index = freq.index(max(freq))
            value = Orange.data.Value(self.classVar, index)
            if resultType == orange.GetValue:
                return value
            for i in range(len(freq)):
                freq[i] = freq[i]/len(self.classifiers)
            if resultType == orange.GetProbabilities:
                return freq
            else:
                return (value, freq)
        elif self.classVar.varType ==Orange.data.Type.Continuous:
            votes = [c(instance, orange.GetBoth if resultType==\
                orange.GetProbabilities else resultType) \
                for c in self.classifiers]
            wsum = float(len(self.classifiers))
            if resultType in [orange.GetBoth, orange.GetProbabilities]:
                pred = sum([float(c) for c, p in votes]) / wsum
#               prob = sum([float(p.modus()) for c, p in votes]) / wsum
                from collections import defaultdict
                prob = defaultdict(float)
                for c, p in votes:
                    try: 
                        prob[float(c)] += p[c] / wsum
                    except IndexError: # p[c] sometimes fails with index error
                        prob[float(c)] += 1.0 / wsum
                prob = orange.ContDistribution(prob)
                return self.classVar(pred), prob if resultType == orange.GetBoth\
                    else prob
            elif resultType == orange.GetValue:
                pred = sum([float(c) for c in votes]) / wsum
                return self.classVar(pred)
            
    def __reduce__(self):
        return type(self), (self.classifiers, self.name, self.classVar), dict(self.__dict__)
    