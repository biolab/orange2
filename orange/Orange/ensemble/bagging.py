import math
import random

import Orange.core as orange
import Orange

#######################################################################
# Bagging

#def BaggedLearner(learner=None, t=10, name='Bagging', examples=None):
#    learner = BaggedLearnerClass(learner, t, name)
#    if examples:
#        return learner(examples)
#    else:
#        return learner

class BaggedLearner(orange.Learner):
    """
    BaggedLearner takes a learner and returns a bagged learner, which is 
    essentially a wrapper around the learner passed as an argument. If 
    examples are passed in arguments, BaggedLearner returns a bagged 
    classifiers. Both learner and classifier then behave just like any 
    other learner and classifier in Orange.
    
    Bagging, in essence, takes a training data and a learner, and builds t 
    classifiers each time presenting a learner a bootstrap sample from the 
    training data. When given a test example, classifiers vote on class, 
    and a bagged classifier returns a class with a highest number of votes. 
    As implemented in Orange, when class probabilities are requested, these 
    are proportional to the number of votes for a particular class.
    """
    def __new__(cls, learner, examples=None, weightId=None, **kwargs):
        self = orange.Learner.__new__(cls, **kwargs)
        if examples is not None:
            self.__init__(self, learner, **kwargs)
            return self.__call__(examples, weightId)
        else:
            return self
        
    def __init__(self, learner, t=10, name='Bagging'):
        """:param learner: A learner to be bagged.
        :type learner: :class:`Orange.core.Learner`
        :param examples: If examples are passed to BaggedLearner, this returns
            a BaggedClassifier, that is, creates t classifiers using learner 
            and a subset of examples, as appropriate for bagging.
        :type examples: :class:`Orange.data.Table`
        :param t: Number of bagged classifiers, that is, classifiers created
            when examples are passed to bagged learner.
        :type t: int
        :param name: The name of the learner.
        :type name: string"""
        self.t = t
        self.name = name
        self.learner = learner

    def __call__(self, examples, weight=0):
        r = random.Random()
        r.seed(0)
        
        n = len(examples)
        classifiers = []
        for i in range(self.t):
            selection = []
            for i in range(n):
                selection.append(r.randrange(n))
            examples = Orange.data.Table(examples)
            data = examples.getitems(selection)
            classifiers.append(self.learner(data, weight))
        return BaggedClassifier(classifiers = classifiers, name=self.name,\
                    classVar=examples.domain.classVar)

class BaggedClassifier(orange.Classifier):
    """Return classifier."""
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType = orange.GetValue):
        if self.classVar.varType == orange.data.Type.Discrete:
            freq = [0.] * len(self.classVar.values)
            for c in self.classifiers:
                freq[int(c(example))] += 1
            index = freq.index(max(freq))
            value = orange.data.Value(self.classVar, index)
            if resultType == orange.GetValue:
                return value
            for i in range(len(freq)):
                freq[i] = freq[i]/len(self.classifiers)
            if resultType == orange.GetProbabilities:
                return freq
            else:
                return (value, freq)
        elif self.classVar.varType == orange.data.Type.Continuous:
            votes = [c(example, orange.GetBoth if resultType==\
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