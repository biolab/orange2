# Description: Class that embeds naive Bayesian classifier, but when learning discretizes the data with entropy-based discretization (which uses training data only)
# Category:    modelling
# Referenced:  c_nb_disc.htm

import orange

class Learner(object):
    def __new__(cls, examples=None, name='discretized bayes', **kwds):
        learner = object.__new__(cls)
        if examples:
            learner.__init__(name) # force init
            return learner(examples)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='discretized bayes'):
        self.name = name

    def __call__(self, data, weight=None):
        disc = orange.Preprocessor_discretize( \
            data, method=orange.EntropyDiscretization())
        model = orange.BayesLearner(disc, weight)
        return Classifier(classifier = model)

class Classifier:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType = orange.GetValue):
        return self.classifier(example, resultType)
