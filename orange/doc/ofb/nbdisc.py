# Version:     1.0
# Description: Class that embeds naive Bayesian classifier, but when learning discretizes the data with entropy-based discretization (which uses training data only)
# Category:    modelling

import orange

class Learner(object):
    def __new__(cls, examples=None, **kwds):
        learner = object.__new__(cls, **kwds)
        if examples:
            return learner(examples)
        else:
            return learner

    def __init__(self, name='discretized bayes'):
        self.name = name

    def __call__(self, data, weight=None):
        disc = orange.Preprocessor_discretize( \
            data, method=orange.EntropyDiscretization())
        model = orange.BayesLearner(disc, weight)
        return Classifier(classifier = model)

class Classifier:
    def __init__(self, **kwds):
        self.__dict__ = kwds

    def __call__(self, example, resultType = orange.GetValue):
        return self.classifier(example, resultType)
