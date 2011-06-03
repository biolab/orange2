""" 
.. index:: naive Bayes classifier
   
.. index:: 
   single: classification; naive Bayes classifier

**********************
Naive Bayes classifier
**********************

The most primitive Bayesian classifier is :obj:`NaiveLearner`. 
`Naive Bayes classification algorithm <http://en.wikipedia.org/wiki/Naive_Bayes_classifier>`_ 
estimates conditional probabilities from training data and uses them
for classification of new data instances. The algorithm learns very fast if all features
in the training data set are discrete. If a number of features are continues, though, the 
algorithm runs slower due to time spent to estimate continuous conditional distributions.

The following example demonstrates a straightforward invocation of
this algorithm (`bayes-run.py`_, uses `titanic.tab`_):

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

:obj:`NaiveLearner` can estimate probabilities using relative frequencies or
m-estimate (`bayes-mestimate.py`_, uses `lenses.tab`_):

.. literalinclude:: code/bayes-mestimate.py
    :lines: 7-

Observing conditional probabilities in an m-estimate based classifier shows a
shift towards the second class - as compared to probabilities above, where
relative frequencies were used. Note that the change in error estimation did
not have any effect on apriori probabilities
(`bayes-thresholdAdjustment.py`_, uses `adult-sample.tab`_):

.. literalinclude:: code/bayes-thresholdAdjustment.py
    :lines: 7-
    
Setting adjustThreshold parameter can sometimes improve the results. Those are
the classification accuracies of 10-fold cross-validation of a normal naive
bayesian classifier, and one with an adjusted threshold::

    [0.7901746265516516, 0.8280138859667578]

Probabilities for continuous features are estimated with \
:class:`ProbabilityEstimatorConstructor_loess`.
(`bayes-plot-iris.py`_, uses `iris.tab`_):

.. literalinclude:: code/bayes-plot-iris.py
    :lines: 4-
    
.. image:: code/bayes-iris.png
   :scale: 50 %

If petal lengths are shorter, the most probable class is "setosa". Irises with
middle petal lengths belong to "versicolor", while longer petal lengths indicate
for "virginica". Critical values where the decision would change are at about
5.4 and 6.3.


.. _bayes-run.py: code/bayes-run.py
.. _bayes-thresholdAdjustment.py: code/bayes-thresholdAdjustment.py
.. _bayes-mestimate.py: code/bayes-mestimate.py
.. _bayes-plot-iris.py: code/bayes-plot-iris.py
.. _adult-sample.tab: code/adult-sample.tab
.. _iris.tab: code/iris.tab
.. _titanic.tab: code/iris.tab
.. _lenses.tab: code/lenses.tab

Implementation details
======================

The following two classes are implemented in C++ (*bayes.cpp*). They are not
intended to be used directly. Here we provide implementation details for those
interested.

Orange.core.BayesLearner
------------------------
Fields estimatorConstructor, conditionalEstimatorConstructor and
conditionalEstimatorConstructorContinuous are empty (None) by default.

If estimatorConstructor is left undefined, p(C) will be estimated by relative
frequencies of examples (see ProbabilityEstimatorConstructor_relative).
When conditionalEstimatorConstructor is left undefined, it will use the same
constructor as for estimating unconditional probabilities (estimatorConstructor
is used as an estimator in ConditionalProbabilityEstimatorConstructor_ByRows).
That is, by default, both will use relative frequencies. But when
estimatorConstructor is set to, for instance, estimate probabilities by
m-estimate with m=2.0, the same estimator will be used for estimation of
conditional probabilities, too.
P(c|vi) for continuous attributes are, by default, estimated with loess (a
variant of locally weighted linear regression), using
ConditionalProbabilityEstimatorConstructor_loess.
The learner first constructs an estimator for p(C). It tries to get a
precomputed distribution of probabilities; if the estimator is capable of
returning it, the distribution is stored in the classifier's field distribution
and the just constructed estimator is disposed. Otherwise, the estimator is
stored in the classifier's field estimator, while the distribution is left
empty.

The same is then done for conditional probabilities. Different constructors are
used for discrete and continuous attributes. If the constructed estimator can
return all conditional probabilities in form of Contingency, the contingency is
stored and the estimator disposed. If not, the estimator is stored. If there
are no contingencies when the learning is finished, the resulting classifier's
conditionalDistributions is None. Alternatively, if all probabilities are
stored as contingencies, the conditionalEstimators fields is None.

Field normalizePredictions is copied to the resulting classifier.

Orange.core.BayesClassifier
---------------------------
Class NaiveClassifier represents a naive bayesian classifier. Probability of
class C, knowing that values of features :math:`F_1, F_2, ..., F_n` are
:math:`v_1, v_2, ..., v_n`, is computed as :math:`p(C|v_1, v_2, ..., v_n) = \
p(C) \\cdot \\frac{p(C|v_1)}{p(C)} \\cdot \\frac{p(C|v_2)}{p(C)} \\cdot ... \
\\cdot \\frac{p(C|v_n)}{p(C)}`.

Note that when relative frequencies are used to estimate probabilities, the
more usual formula (with factors of form :math:`\\frac{p(v_i|C)}{p(v_i)}`) and
the above formula are exactly equivalent (without any additional assumptions of
independency, as one could think at a first glance). The difference becomes
important when using other ways to estimate probabilities, like, for instance,
m-estimate. In this case, the above formula is much more appropriate. 

When computing the formula, probabilities p(C) are read from distribution, which
is of type Distribution, and stores a (normalized) probability of each class.
When distribution is None, BayesClassifier calls estimator to assess the
probability. The former method is faster and is actually used by all existing
methods of probability estimation. The latter is more flexible.

Conditional probabilities are computed similarly. Field conditionalDistribution
is of type DomainContingency which is basically a list of instances of
Contingency, one for each attribute; the outer variable of the contingency is
the attribute and the inner is the class. Contingency can be seen as a list of
normalized probability distributions. For attributes for which there is no
contingency in conditionalDistribution a corresponding estimator in
conditionalEstimators is used. The estimator is given the attribute value and
returns distributions of classes.

If neither, nor pre-computed contingency nor conditional estimator exist, the
attribute is ignored without issuing any warning. The attribute is also ignored
if its value is undefined; this cannot be overriden by estimators.

Any field (distribution, estimator, conditionalDistributions,
conditionalEstimators) can be None. For instance, BayesLearner normally
constructs a classifier which has either distribution or estimator defined.
While it is not an error to have both, only distribution will be used in that
case. As for the other two fields, they can be both defined and used
complementarily; the elements which are missing in one are defined in the
other. However, if there is no need for estimators, BayesLearner will not
construct an empty list; it will not construct a list at all, but leave the
field conditionalEstimators empty.

If you only need probabilities of individual class call BayesClassifier's
method p(class, example) to compute the probability of this class only. Note
that this probability will not be normalized and will thus, in general, not
equal the probability returned by the call operator.
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
    
    ..
        :param adjustTreshold: sets the corresponding attribute
        :type adjustTreshold: boolean
        :param m: sets the :obj:`estimatorConstructor` to
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
                
    :rtype: :class:`Orange.classification.bayes.NaiveLearner` or
            :class:`Orange.classification.bayes.NaiveClassifier`
            
    Constructor parameters set the corresponding attributes.
    
    .. attribute:: adjustTreshold
    
        If set and the class is binary, the classifier's
        threshold will be set as to optimize the classification accuracy.
        The threshold is tuned by observing the probabilities predicted on
        learning data. Setting it to True can increase the
        accuracy considerably
        
    .. attribute:: m
    
        m for m-estimate. If set, m-estimation of probabilities
        will be used using :class:`orange.ProbabilityEstimatorConstructor_m`.
        This attribute is ignored if you also set estimatorConstructor.
        
    .. attribute:: estimatorConstructor
    
        Probability estimator constructor for
        prior class probabilities. Defaults to
        :class:`orange.ProbabilityEstimatorConstructor_relative`.
        Setting this attribute disables the above described attribute m.
        
    .. attribute:: conditionalEstimatorConstructor
    
        Probability estimator constructor
        for conditional probabilities for discrete features. If omitted,
        the estimator for prior probabilities will be used.
        
    .. attribute:: conditionalEstimatorConstructorContinuous
    
        Probability estimator constructor for conditional probabilities for
        continuous features. Defaults to 
        :class:`orange.ConditionalProbabilityEstimatorConstructor_loess`.
    """
    
    def __new__(cls, instances = None, weight_id = 0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances, weight_id)
        else:
            return self
        
    def __init__(self, adjust_threshold=False, m=0, estimator_constructor=None,
                 conditional_estimator_constructor=None,
                 conditional_estimator_constructor_continuous=None,**argkw):
        self.adjust_threshold = adjust_threshold
        self.m = m
        self.estimator_constructor = estimator_constructor
        self.conditional_estimator_constructor = conditional_estimator_constructor
        self.conditional_estimator_constructor_continuous = conditional_estimator_constructor_continuous
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
        if self.estimator_constructor:
            bayes.estimator_constructor = self.estimator_constructor
            if self.m:
                if not hasattr(bayes.estimator_constructor, "m"):
                    raise AttributeError, "invalid combination of attributes: 'estimator_constructor' does not expect 'm'"
                else:
                    self.estimator_constructor.m = self.m
        elif self.m:
            bayes.estimator_constructor = Orange.core.ProbabilityEstimatorConstructor_m(m = self.m)
        if self.conditional_estimator_constructor:
            bayes.conditional_estimator_constructor = self.conditional_estimator_constructor
        elif bayes.estimator_constructor:
            bayes.conditional_estimator_constructor = Orange.core.ConditionalProbabilityEstimatorConstructor_ByRows()
            bayes.conditional_estimator_constructor.estimator_constructor=bayes.estimator_constructor
        if self.conditional_estimator_constructor_continuous:
            bayes.conditional_estimator_constructor_continuous = self.conditional_estimator_constructor_continuous
        if self.adjust_threshold:
            bayes.adjust_threshold = self.adjust_threshold
        return NaiveClassifier(bayes(instances, weight))
NaiveLearner = Orange.misc.deprecated_members(
{     "adjustThreshold": "adjust_threshold",
      "estimatorConstructor": "estimator_constructor",
      "conditionalEstimatorConstructor": "conditional_estimator_constructor",
      "conditionalEstimatorConstructorContinuous":"conditional_estimator_constructor_continuous",
      "weightID": "weight_id"
}, in_place=False)(NaiveLearner)


class NaiveClassifier(Orange.classification.Classifier):
    """
    Predictor based on calculated probabilities. It wraps an
    :class:`Orange.core.BayesClassifier` that does the actual classification.
    
    :param baseClassifier: an :class:`Orange.core.BayesLearner` to wrap. If
            not set, a new :class:`Orange.core.BayesLearner` is created.
    :type baseClassifier: :class:`Orange.core.BayesLearner`
    
    .. attribute:: distribution
    
        Stores probabilities of classes, i.e. p(C) for each class C.
        
    .. attribute:: estimator
    
        An object that returns a probability of class p(C) for a given class C.
        
    .. attribute:: conditionalDistributions
    
        A list of conditional probabilities.
        
    .. attribute:: conditionalEstimators
    
        A list of estimators for conditional probabilities.
        
    .. attribute:: adjustThreshold
    
        For binary classes, this tells the learner to
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
        """Classify a new instance.
        
        :param instance: instance to be classified.
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
        """
        Return probability of a single class.
        Probability is not normalized and can be different from probability
        returned from __call__.
        
        :param class_: class variable for which the probability should be
                output.
        :type class_: :class:`Orange.data.Variable`
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        """
        return self.nativeBayesClassifier.p(class_, instance)
    
    def __str__(self):
        """return classifier in human friendly format."""
        nValues=len(self.classVar.values)
        frmtStr=' %10.3f'*nValues
        classes=" "*20+ ((' %10s'*nValues) % tuple([i[:10] for i in self.classVar.values]))
        
        return "\n".join([
            classes,
            "class probabilities "+(frmtStr % tuple(self.distribution)),
            "",
            "\n".join(["\n".join([
                "Attribute " + i.variable.name,
                classes,
                "\n".join(
                    ("%20s" % i.variable.values[v][:20]) + (frmtStr % tuple(i[v]))
                    for v in xrange(len(i.variable.values)))]
                ) for i in self.conditionalDistributions])])
            

def printModel(model):
    print NaiveClassifier(model)
