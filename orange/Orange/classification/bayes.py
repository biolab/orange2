""" 
   index:: naive bayes

=========================
Naive Bayesian Classifier
=========================

.. index:: Naive Bayesian Learner
.. autoclass:: Orange.classification.bayes.NaiveBayes
   :members:
   
"""

import Orange
from Orange.core import BayesClassifier as _BayesClassifier
from Orange.core import BayesLearner as _BayesLearner

class NaiveBayes(Orange.core.Learner):
    """
    Naive bayes learner
    """
    
    def __new__(cls, examples = None, weightID = 0, **argkw):
        self = Orange.core.Learner.__new__(cls, **argkw)
        if examples:
            self.__init__(**argkw)
            return self.__call__(examples, weightID)
        else:
            return self
        
    def __init__(self, normalizePredictions=True, adjustTreshold=False,
                 m=0, estimatorConstructor=None, conditionalEstimatorConstructor=None,
                 conditionalEstimatorConstructorContinuous=None,**argkw):
        """
        :param adjustTreshold: If set and the class is binary, the classifier's
                threshold will be set as to optimize the classification accuracy.
                The threshold is tuned by observing the probabilities predicted on
                learning data. Default is False (to conform with the usual naive
                bayesian classifiers), but setting it to True can increase the
                accuracy considerably.
        :type adjustTreshold: boolean
        :param m: m for m-estimate. If set, m-estimation of probabilities
                will be used using :class:`orange.ProbabilityEstimatorConstructor_m`
                This attribute is ignored if you also set estimatorConstructor.
        :type m: integer
        :param estimatorConstructor: Probability estimator constructor for
                prior class probabilities. Defaults to
                :`class:orange.ProbabilityEstimatorConstructor_relative`
                Setting this attribute disables the above described attribute m.
        :type estimatorConstructor: orange.ProbabilityEstimatorConstructor
        :param conditionalEstimatorConstructor: Probability estimator constructor
                for conditional probabilities for discrete features. If omitted,
                the estimator for a priori will be used.
                class probabilities.
        :type conditionalEstimatorConstructor: orange.ConditionalProbabilityEstimatorConstructor
        :param conditionalEstimatorConstructorContinuous: Probability estimator constructor
                for conditional probabilities for continuous features. Defaults to
                :class:`orange.ConditionalProbabilityEstimatorConstructor_loess`
        :type conditionalEstimatorConstructorContinuous: orange.ConditionalProbabilityEstimatorConstructor
        """
        self.adjustThreshold = adjustTreshold
        self.m = m
        self.estimatorConstructor = estimatorConstructor
        self.conditionalEstimatorConstructor = conditionalEstimatorConstructor
        self.conditionalEstimatorConstructorContinuous = conditionalEstimatorConstructorContinuous
        self.__dict__.update(argkw)

    def __call__(self, examples, weight=0):
        bayes = _BayesLearner()
        if self.estimatorConstructor:
            bayes.estimatorConstructor = self.estimatorConstructor
            if self.m:
                if not hasattr(bayes.estimatorConstructor, "m"):
                    raise AttributeError, "invalid combination of attributes: 'estimatorConstructor' does not expect 'm'"
                else:
                    self.estimatorConstructor.m = self.m
        elif self.m:
            bayes.estimatorConstructor = Orange.core.ProbabilityEstimatorConstructor_m(m = self.m)
        if self.conditionalEstimatorConstructor:
            bayes.conditionalEstimatorConstructor = self.conditionalEstimatorConstructor
        else:
            bayes.conditionalEstimatorConstructor = orange.ConditionalProbabilityEstimatorConstructor_ByRows()
            bayes.conditionalEstimatorConstructor.estimatorConstructor=bayes.estimatorConstructor
            
        if self.conditionalEstimatorConstructorContinuous:
            bayes.conditionalEstimatorConstructorContinuous = self.conditionalEstimatorConstructorContinuous
            
        return NaiveBayesClassifier(bayes(examples, weight))
            
class NaiveBayesClassifier(Orange.core.Classifier):
    def __init__(self, nbc):
        self.nativeBayesClassifier = nbc
        for k, v in self.nativeBayesClassifier.__dict__.items():
            self.__dict__[k] = v
  
    def __call__(self, *args, **kwdargs):
        self.nativeBayesClassifier(*args, **kwdargs)

    def __setattr__(self, name, value):
        if name == "nativeBayesClassifier":
            self.__dict__[name] = value
            return
        if name in self.nativeBayesClassifier.__dict__:
            self.nativeBayesClassifier.__dict__[name] = value
        self.__dict__[name] = value
    
    
    def printModel(self):
        nValues=len(self.classVar.values)
        frmtStr=' %10.3f'*nValues
        classes=" "*20+ ((' %10s'*nValues) % tuple([i[:10] for i in self.classVar.values]))
        print classes
        print "class probabilities "+(frmtStr % tuple(self.distribution))
        print
    
        for i in self.conditionalDistributions:
            print "Attribute", i.variable.name
            print classes
            for v in range(len(i.variable.values)):
                print ("%20s" % i.variable.values[v][:20]) + (frmtStr % tuple(i[v]))
            print
