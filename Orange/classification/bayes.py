import Orange
import Orange.core
from Orange.core import BayesClassifier as _BayesClassifier
from Orange.core import BayesLearner as _BayesLearner


class NaiveLearner(Orange.classification.Learner):
    """
    Probabilistic classifier based on applying Bayes' theorem (from Bayesian
    statistics) with strong (naive) independence assumptions. Constructor parameters
    set the corresponding attributes.
    
    .. attribute:: adjust_threshold
    
        If set and the class is binary, the classifier's
        threshold will be set as to optimize the classification accuracy.
        The threshold is tuned by observing the probabilities predicted on
        learning data. Setting it to True can increase the
        accuracy considerably
        
    .. attribute:: m
    
        m for m-estimate. If set, m-estimation of probabilities
        will be used using :class:`~Orange.statistics.estimate.M`.
        This attribute is ignored if you also set :obj:`estimator_constructor`.
        
    .. attribute:: estimator_constructor
    
        Probability estimator constructor for
        prior class probabilities. Defaults to
        :class:`~Orange.statistics.estimate.RelativeFrequency`.
        Setting this attribute disables the above described attribute :obj:`m`.
        
    .. attribute:: conditional_estimator_constructor
    
        Probability estimator constructor
        for conditional probabilities for discrete features. If omitted,
        the estimator for prior probabilities will be used.
        
    .. attribute:: conditional_estimator_constructor_continuous
    
        Probability estimator constructor for conditional probabilities for
        continuous features. Defaults to 
        :class:`~Orange.statistics.estimate.Loess`.
    """
    
    def __new__(cls, data = None, weight_id = 0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if data:
            self.__init__(**argkw)
            return self.__call__(data, weight_id)
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

    def __call__(self, data, weight=0):
        """Learn from the given table of data instances.
        
        :param data: Data instances to learn from.
        :type data: :class:`~Orange.data.Table`
        :param weight: Id of meta attribute with weights of instances
        :type weight: int
        :rtype: :class:`~Orange.classification.bayes.NaiveClassifier`
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
        return NaiveClassifier(bayes(data, weight))
NaiveLearner = Orange.misc.deprecated_members(
{     "adjustThreshold": "adjust_threshold",
      "estimatorConstructor": "estimator_constructor",
      "conditionalEstimatorConstructor": "conditional_estimator_constructor",
      "conditionalEstimatorConstructorContinuous":"conditional_estimator_constructor_continuous",
      "weightID": "weight_id"
})(NaiveLearner)


class NaiveClassifier(Orange.classification.Classifier):
    """
    Predictor based on calculated probabilities.
    
    .. attribute:: distribution
    
        Stores probabilities of classes, i.e. p(C) for each class C.
        
    .. attribute:: estimator
    
        An object that returns a probability of class p(C) for a given class C.
        
    .. attribute:: conditional_distributions
    
        A list of conditional probabilities.
        
    .. attribute:: conditional_estimators
    
        A list of estimators for conditional probabilities.
        
    .. attribute:: adjust_threshold
    
        For binary classes, this tells the learner to
        determine the optimal threshold probability according to 0-1
        loss on the training set. For multiple class problems, it has
        no effect.
    """
    
    def __init__(self, base_classifier=None):
        if not base_classifier: base_classifier = _BayesClassifier()
        self.native_bayes_classifier = base_classifier
        for k, v in self.native_bayes_classifier.__dict__.items():
            self.__dict__[k] = v
  
    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue,
                 *args, **kwdargs):
        """Classify a new instance.
        
        :param instance: instance to be classified.
        :type instance: :class:`~Orange.data.Instance`
        :param result_type: :class:`~Orange.classification.Classifier.GetValue` or
              :class:`~Orange.classification.Classifier.GetProbabilities` or
              :class:`~Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`~Orange.data.Value`,
              :class:`~Orange.statistics.distribution.Distribution` or a
              tuple with both
        """
        return self.native_bayes_classifier(instance, result_type, *args, **kwdargs)

    def __setattr__(self, name, value):
        if name == "native_bayes_classifier":
            self.__dict__[name] = value
            return
        if name in self.native_bayes_classifier.__dict__:
            self.native_bayes_classifier.__dict__[name] = value
        self.__dict__[name] = value
    
    def p(self, class_, instance):
        """
        Return probability of a single class.
        Probability is not normalized and can be different from probability
        returned from __call__.
        
        :param class_: class value for which the probability should be
                output.
        :type class_: :class:`~Orange.data.Value`
        :param instance: instance to be classified.
        :type instance: :class:`~Orange.data.Instance`
        
        """
        return self.native_bayes_classifier.p(class_, instance)
    
    def __str__(self):
        """Return classifier in human friendly format."""
        nvalues=len(self.class_var.values)
        frmtStr=' %10.3f'*nvalues
        classes=" "*20+ ((' %10s'*nvalues) % tuple([i[:10] for i in self.class_var.values]))
        
        return "\n".join([
            classes,
            "class probabilities "+(frmtStr % tuple(self.distribution)),
            "",
            "\n\n".join(["\n".join([
                "Attribute " + i.variable.name,
                classes,
                "\n".join(
                    ("%20s" % i.variable.values[v][:20]) + (frmtStr % tuple(i[v]))
                    for v in xrange(len(i.variable.values)))]
                ) for i in self.conditional_distributions
                        if i.variable.var_type == i.variable.Discrete])])
            

def printModel(model):
    print NaiveClassifier(model)
