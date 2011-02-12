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
    def __new__(cls, learner, instances=None, weightId=None, **kwargs):
        self = orange.Learner.__new__(cls, **kwargs)
        if instances is not None:
            self.__init__(self, learner, **kwargs)
            return self.__call__(instances, weightId)
        else:
            return self

    def __init__(self, learner, t=10, name='AdaBoost.M1'):
        self.t = t
        self.name = name
        self.learner = learner

    def __call__(self, instances, origWeight = 0):
        """
        Learn from the given table of data instances.
        
        :param instances: data instances to learn from.
        :type instances: Orange.data.Table
        :param origWeight: weight.
        :type origWeight: int
        :rtype: :class:`Orange.ensemble.boosting.BoostedClassifier`
        
        """
        import math
        weight = orange.newmetaid()
        if origWeight:
            for i in instances:
                i.setweight(weight, i.getweight(origWeight))
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
                    name=self.name, classVar=instances.domain.classVar)
            beta = epsilon/(1-epsilon)
            for e in range(n):
                if corr[e]:
                    instances[e].setweight(weight, instances[e].getweight(weight)*beta)
            f = 1/float(sum([e.getweight(weight) for e in instances]))
            for e in range(n):
                instances[e].setweight(weight, instances[e].getweight(weight)*f)

        instances.removeMetaAttribute(weight)
        return BoostedClassifier(classifiers = classifiers, name=self.name, 
            classVar=instances.domain.classVar)

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
        votes = [0.] * len(self.classVar.values)
        for c, e in self.classifiers:
            votes[int(c(instance))] += e
        index = Orange.misc.selection.selectBestIndex(votes)
        # TODO
        value = Orange.data.Value(self.classVar, index)
        if resultType == orange.GetValue:
            return value
        sv = sum(votes)
        for i in range(len(votes)):
            votes[i] = votes[i]/sv
        if resultType == orange.GetProbabilities:
            return votes
        else:
            return (value, votes)