import Orange
import Orange.core as orange
import orngMisc

_inf = 100000

#def BoostedLearner(learner, examples=None, t=10, name='AdaBoost.M1'):
#    learner = BoostedLearnerClass(learner, t, name)
#    if examples:
#        return learner(examples)
#    else:
#        return learner

class BoostedLearner(orange.Learner):
    """
    Instead of drawing a series of bootstrap samples from the training set,
    bootstrap maintains a weight for each instance. When classifier is 
    trained from the training set, the weights for misclassified instances 
    are increased. Just like in bagged learner, the class is decided based 
    on voting of classifiers, but in boosting votes are weighted by accuracy 
    obtained on training set.

    BoostedLearner is an implementation of AdaBoost.M1 (Freund and Shapire, 
    1996). From user's viewpoint, the use of the BoostedLearner is similar to 
    that of BaggedLearner. The learner passed as an argument needs to deal 
    with example weights.
    """
    def __new__(cls, learner, examples=None, weightId=None, **kwargs):
        self = orange.Learner.__new__(cls, **kwargs)
        if examples is not None:
            self.__init__(self, learner, **kwargs)
            return self.__call__(examples, weightId)
        else:
            return self

    def __init__(self, learner, t=10, name='AdaBoost.M1'):
        """:param learner: A learner to be bagged.
        :type learner: :class:`Orange.core.Learner`
        :param examples: If examples are passed to BoostedLearner,
            this returns a BoostedClassifier, that is, creates t 
            classifiers using learner and a subset of examples, 
            as appropriate for AdaBoost.M1 (default: None).
        :type examples: :class:`Orange.data.Table`
        :param t: Number of boosted classifiers created from the example set.
        :type t: int
        :param name: The name of the learner.
        :type name: string"""
        self.t = t
        self.name = name
        self.learner = learner

    def __call__(self, instances, origWeight = 0):
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
            epsilon = epsilon / float(reduce(lambda x,y:x+y.getweight(weight), instances, 0))
            classifiers.append((classifier, epsilon and math.log((1-epsilon)/epsilon) or _inf))
            if epsilon==0 or epsilon >= 0.499:
                if epsilon >= 0.499 and len(classifiers)>1:
                    del classifiers[-1]
                instances.removeMetaAttribute(weight)
                return BoostedClassifier(classifiers = classifiers, name=self.name, classVar=instances.domain.classVar)
            beta = epsilon/(1-epsilon)
            for e in range(n):
                if corr[e]:
                    instances[e].setweight(weight, instances[e].getweight(weight)*beta)
            f = 1/float(sum([e.getweight(weight) for e in instances]))
            for e in range(n):
                instances[e].setweight(weight, instances[e].getweight(weight)*f)

        instances.removeMetaAttribute(weight)
        return BoostedClassifier(classifiers = classifiers, name=self.name, classVar=instances.domain.classVar)

class BoostedClassifier(orange.Classifier):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, example, resultType = orange.GetValue):
        votes = [0.] * len(self.classVar.values)
        for c, e in self.classifiers:
            votes[int(c(example))] += e
        index = orngMisc.selectBestIndex(votes)
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