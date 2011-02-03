""" 
   index:: naive bayes

=========================
Naive Bayesian Classifier
=========================

The most primitive bayesian classifier is :obj:`NaiveLearner`. The class
estimates conditional probabilities from train data and uses them for
prediction of new examples. 

Example (`bayes-run.py`_, uses `iris.tab`_)

.. literalinclude:: code/bayes-run.py
    :lines: 7-

.. index:: Naive Bayesian Learner
.. autoclass:: Orange.classification.bayes.NaiveLearner
   :members:
   :show-inheritance:
 
.. autoclass:: Orange.classification.bayes.NaiveClassifier
   :members:
   :show-inheritance:
   
Examples
========
Let us load the data, induce a classifier and see how it performs on the first
five examples.

>>> from Orange import *
>>> table = data.Table("lenses")
>>> bayes = classification.bayes.NaiveLearner(table)
>>>
>>> for ex in table[:5]:
...    print ex.getclass(), bayes(ex)
no no
no no
soft soft
no no
hard hard

The classifier is correct in all five cases. Interested in probabilities,
maybe?

>>> for ex in table[:5]:
...     print ex.getclass(), bayes(ex, Orange.classification.Classifier.GetProbabilities)
no <0.423, 0.000, 0.577>
no <0.000, 0.000, 1.000>
soft <0.000, 0.668, 0.332>
no <0.000, 0.000, 1.000>
hard <0.715, 0.000, 0.285>

While very confident about the second and the fourth example, the classifier
guessed the correct class of the first one only by a small margin of 42 vs.
58 percents.

Now, let us peek into the classifier.

>>> print bayes.estimator
None
>>> print bayes.distribution
<0.167, 0.208, 0.625>
>>> print bayes.conditionalEstimators
None
>>> print bayes.conditionalDistributions[0]
<'young': <0.250, 0.250, 0.500>, 'p_psby': <0.125, 0.250, 0.625>, (...)
>>> bayes.conditionalDistributions[0]["young"]
<0.250, 0.250, 0.500>

The classifier has no estimator since probabilities are stored in distribution.
The probability of the first class is 0.167, of the second 0.208 and the
probability of the third class is 0.625. Nor does it have 
conditionalEstimators, probabilities are stored in conditionalDistributions.
We printed the contingency matrix for the first attribute and, in the last
line, conditional probabilities of the three classes when the value of the
first attribute is "young".

Let us now use m-estimate instead of relative frequencies.

>>> bayesl = classification.bayes.NaiveLearner(m=2.0)
>>> bayes = bayesl(table)

The classifier is still correct for all examples.

>>> for ex in table[:5]:
...     print ex.getclass(), bayes(ex, Orange.classification.Classifier.GetBoth) 
no <0.375, 0.063, 0.562>;
no <0.016, 0.003, 0.981>
soft <0.021, 0.607, 0.372>
no <0.001, 0.039, 0.960>
hard <0.632, 0.030, 0.338>

Observing probabilities shows a shift towards the third, more frequent class -
as compared to probabilities above, where relative frequencies were used.

>>> print bayes.conditionalDistributions[0]
<'young': <0.233, 0.242, 0.525>, 'p_psby': <0.133, 0.242, 0.625>, (...)

Note that the change in error estimation did not have any effect on apriori
probabilities:

>>> print bayes.distribution
<0.167, 0.208, 0.625>

The reason for this is that this same distribution was used as apriori
distribution for m-estimation. (How to enforce another apriori distribution?
While the orange C++ core supports of it, this feature has not been exported
to Python yet.)

Finally, let us show an example with continuous attributes. We will take iris
dataset that contains four continuous and no discrete attributes.

>>> table = data.Table("iris")
>>> bayes = orange.BayesLearner(table)
>>> for exi in range(0, len(table), 20):
...     print data[exi].getclass(), bayes(table[exi], orange.Classifier.GetBoth)

The classifier works well. To see a glimpse of how it works, let us observe
conditional distributions for the first attribute. It is stored in
conditionalDistributions, as before, except that it now behaves as a dictionary,
not as a list like before (see information on distributions.

>>> print bayes.conditionalDistributions[0]
<4.300: <0.837, 0.137, 0.026>;, 4.333: <0.834, 0.140, 0.026>, 4.367: <0.830, (...)

For a nicer picture, we can print out the probabilities, copy and paste it to
some graph drawing program ... and get something like the figure below.

>>> for x, probs in bayes.conditionalDistributions[0].items():
...     print "%5.3f\t%5.3f\t%5.3f\t%5.3f" % (x, probs[0], probs[1], probs[2])
4.300   0.837   0.137   0.026
4.333   0.834   0.140   0.026
4.367   0.830   0.144   0.026
4.400   0.826   0.147   0.027
4.433   0.823   0.150   0.027
(...)

If petal lengths are shorter, the most probable class is "setosa". Irises with middle petal lengths belong to "versicolor", while longer petal lengths indicate for "virginica". Critical values where the decision would change are at about 5.4 and 6.3.

It is important to stress that the curves are relatively smooth although no fitting (either manual or automatic) of parameters took place.


.. _bayes-run.py: code/bayes-run.py
.. _iris.tab: code/iris.tab
"""

import Orange
from Orange.core import BayesClassifier as _BayesClassifier
from Orange.core import BayesLearner as _BayesLearner

class NaiveLearner(Orange.classification.Learner):
    """
    Probabilistic classifier based on applying Bayes' theorem (from Bayesian
    statistics) with strong (naive) independence assumptions.
    If data instances are provided to the constructor, the learning algorithm
    is called and the resulting classifier is returned instead of the learner.
    
    :param adjustTreshold: sets the corresponding attribute
    :type adjustTreshold: boolean
    :param m: sets the estimatorConstructor to \
    :class:`orange.ProbabilityEstimatorConstructor_m` with specified m 
    :type m: integer
    :param estimatorConstructor: sets the corresponding attribute
    :type estimatorConstructor: orange.ProbabilityEstimatorConstructor
    :param conditionalEstimatorConstructor: sets the corresponding attribute
    :type conditionalEstimatorConstructor:
            :class:`orange.ConditionalProbabilityEstimatorConstructor`
    :param conditionalEstimatorConstructorContinuous: sets the corresponding
            attribute
    :type conditionalEstimatorConstructorContinuous: 
            :class:`orange.ConditionalProbabilityEstimatorConstructor`
    :rtype: :class:`Orange.classification.bayes.NaiveBayesLearner` or
            :class:`Orange.classification.bayes.NaiveBayesClassifier` 
    
    All attributes can also be set as constructor parameters.
    
    :var adjustTreshold: If set and the class is binary, the classifier's
            threshold will be set as to optimize the classification accuracy.
            The threshold is tuned by observing the probabilities predicted on
            learning data. Setting it to True can increase the
            accuracy considerably
    :var m: m for m-estimate. If set, m-estimation of probabilities
            will be used using :class:`orange.ProbabilityEstimatorConstructor_m`
            This attribute is ignored if you also set estimatorConstructor.
    :var estimatorConstructor: Probability estimator constructor for
            prior class probabilities. Defaults to
            :class:`orange.ProbabilityEstimatorConstructor_relative`
            Setting this attribute disables the above described attribute m.
    :var conditionalEstimatorConstructor: Probability estimator constructor
            for conditional probabilities for discrete features. If omitted,
            the estimator for prior probabilities will be used.
    :var conditionalEstimatorConstructorContinuous: Probability estimator
            constructor for conditional probabilities for continuous features.
            Defaults to 
            :class:`orange.ConditionalProbabilityEstimatorConstructor_loess` 
    """
    
    def __new__(cls, examples = None, weightID = 0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if examples:
            self.__init__(**argkw)
            return self.__call__(examples, weightID)
        else:
            return self
        
    def __init__(self, adjustTreshold=False, m=0, estimatorConstructor=None,
                 conditionalEstimatorConstructor=None,
                 conditionalEstimatorConstructorContinuous=None,**argkw):
        self.adjustThreshold = adjustTreshold
        self.m = m
        self.estimatorConstructor = estimatorConstructor
        self.conditionalEstimatorConstructor = conditionalEstimatorConstructor
        self.conditionalEstimatorConstructorContinuous = conditionalEstimatorConstructorContinuous
        self.__dict__.update(argkw)

    def __call__(self, instances, weight=0):
        """Learn from the given table of data instances.
        
        :param instances: Data instances to learn from.
        :type instances: Orange.data.Table
        :param weight: Id of meta attribute with weights of instances
        :type weight: integer
        :rtype: :class:`Orange.classification.bayes.NaiveBayesClassifier`
        """
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
            bayes.conditionalEstimatorConstructor = Orange.core.ConditionalProbabilityEstimatorConstructor_ByRows()
            bayes.conditionalEstimatorConstructor.estimatorConstructor=bayes.estimatorConstructor
            
        if self.conditionalEstimatorConstructorContinuous:
            bayes.conditionalEstimatorConstructorContinuous = self.conditionalEstimatorConstructorContinuous
            
        return NaiveClassifier(bayes(instances, weight))
            
class NaiveClassifier(Orange.classification.Classifier):
    """
    Predictor based on calculated probabilities
    
    :param baseClassifier:
    :type:
    
    :var distribution: Stores probabilities of classes, i.e. p(C) for each
            class C.
    :var estimator: An object that returns a probability of class p(C) for a
            given class C.
    :var conditionalDistributions: A list of conditional probabilities.
    :var conditionalEstimators: A list of estimators for conditional
            probabilities
    :var normalize: Tells whether the returned probabilities should be
            normalized (default: True)
    :var adjustThreshold: For binary classes, this tells the learner to
            determine the optimal threshold probability according to 0-1
            loss on the training set. For multiple class problems, it has
            no effect.
    """
    
    def __init__(self, baseClassifier=None):
        if not baseClassifier: baseClassifier = _BayesClassifier()
        self.nativeBayesClassifier = baseClassifier
        for k, v in self.nativeBayesClassifier.__dict__.items():
            self.__dict__[k] = v
  
    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue,
                 *args, **kwdargs):
        """Classify a new instance
        
        :param instance: instance to be classifier
        :type instance: :class:`Orange.data.Instance`
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        return self.nativeBayesClassifier(instance, result_type, *args, **kwdargs)

    def __setattr__(self, name, value):
        if name == "nativeBayesClassifier":
            self.__dict__[name] = value
            return
        if name in self.nativeBayesClassifier.__dict__:
            self.nativeBayesClassifier.__dict__[name] = value
        self.__dict__[name] = value
    
    def p(self, class_, instance):
        """Return probability of single class
        Probability is not normalized and can be different from probability
        returned from __call__
        
        :param class_:
        :type class_:
        :param instance: instance to be classified
        :type instance: :class:`Orange.data.Instance`
        
        """
        return self.nativeBayesClassifier.p(class_, instance)
    
    def printModel(self):
        """Print classificator in human friendly format"""
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
