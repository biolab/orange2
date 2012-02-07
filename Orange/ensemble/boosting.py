import Orange
import Orange.core as orange

_inf = 100000

class BoostedLearner(orange.Learner):
    """
    Instead of drawing a series of bootstrap samples from the training set,
    bootstrap maintains a weight for each instance. When a classifier is 
    trained from the training set, the weights for misclassified instances 
    are increased. Just like in a bagged learner, the class is decided based 
    on voting of classifiers, but in boosting votes are weighted by accuracy 
    obtained on training set.

    BoostedLearner is an implementation of AdaBoost.M1 (Freund and Shapire, 
    1996). From user's viewpoint, the use of the BoostedLearner is similar to 
    that of BaggedLearner. The learner passed as an argument needs to deal 
    with instance weights.
    
    :param learner: learner to be boosted.
    :type learner: :class:`Orange.core.Learner`
    :param t: number of boosted classifiers created from the instance set.
    :type t: int
    :param name: name of the resulting learner.
    :type name: str
    :rtype: :class:`Orange.ensemble.boosting.BoostedClassifier` or 
            :class:`Orange.ensemble.boosting.BoostedLearner`
    """
    def __new__(cls, learner, instances=None, weight_id=None, **kwargs):
        self = orange.Learner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(self, learner, **kwargs)
            return self.__call__(instances, weight_id)
        else:
            return self

    def __init__(self, learner, t=10, name='AdaBoost.M1'):
        self.t = t
        self.name = name
        self.learner = learner

    def __call__(self, instances, orig_weight = 0):
        """
        Learn from the given table of data instances.
        
        :param instances: data instances to learn from.
        :type instances: Orange.data.Table
        :param orig_weight: weight.
        :type orig_weight: int
        :rtype: :class:`Orange.ensemble.boosting.BoostedClassifier`
        
        """
        import math
        weight = Orange.feature.new_meta_id()
        if orig_weight:
            for i in instances:
                i.setweight(weight, i.getweight(orig_weight))
        else:
            instances.addMetaAttribute(weight, 1.0)
            
        n = len(instances)
        classifiers = []
        for i in range(self.t):
            epsilon = 0.0
            classifier = self.learner(instances, weight)
            corr = []
            for ex in instances:
                if classifier(ex) != ex.getclass():
                    epsilon += ex.getweight(weight)
                    corr.append(0)
                else:
                    corr.append(1)
            epsilon = epsilon / float(reduce(lambda x,y:x+y.getweight(weight), 
                instances, 0))
            classifiers.append((classifier, epsilon and math.log(
                (1-epsilon)/epsilon) or _inf))
            if epsilon==0 or epsilon >= 0.499:
                if epsilon >= 0.499 and len(classifiers)>1:
                    del classifiers[-1]
                instances.removeMetaAttribute(weight)
                return BoostedClassifier(classifiers = classifiers, 
                    name=self.name, class_var=instances.domain.class_var)
            beta = epsilon/(1-epsilon)
            for e in range(n):
                if corr[e]:
                    instances[e].setweight(weight, instances[e].getweight(weight)*beta)
            f = 1/float(sum([e.getweight(weight) for e in instances]))
            for e in range(n):
                instances[e].setweight(weight, instances[e].getweight(weight)*f)

        instances.removeMetaAttribute(weight)
        return BoostedClassifier(classifiers = classifiers, name=self.name, 
            class_var=instances.domain.class_var)
BoostedLearner = Orange.misc.deprecated_members({"examples":"instances", "classVar":"class_var", "weightId":"weigth_id", "origWeight":"orig_weight"})(BoostedLearner)

class BoostedClassifier(orange.Classifier):
    """
    A classifier that uses a boosting technique. Usually the learner
    (:class:`Orange.ensemble.boosting.BoostedLearner`) is used to construct the
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
        votes = Orange.statistics.distribution.Discrete(self.class_var)
        for c, e in self.classifiers:
            votes[int(c(instance))] += e
        index = Orange.misc.selection.select_best_index(votes)
        # TODO
        value = Orange.data.Value(self.class_var, index)
        if result_type == orange.GetValue:
            return value
        sv = sum(votes)
        for i in range(len(votes)):
            votes[i] = votes[i]/sv
        if result_type == orange.GetProbabilities:
            return votes
        elif result_type == orange.GetBoth:
            return (value, votes)
        else:
            return value
        
    def __reduce__(self):
        return type(self), (self.classifiers, self.name, self.class_var), dict(self.__dict__)

BoostedClassifier = Orange.misc.deprecated_members({"classVar":"class_var", "resultType":"result_type"})(BoostedClassifier)
