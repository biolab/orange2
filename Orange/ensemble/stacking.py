import Orange

class StackedClassificationLearner(Orange.classification.Learner):
    """Stacking by inference of meta classifier from class probability estimates
    on cross-validation held-out data for level-0 classifiers developed on held-in data sets.

    :param learners: level-0 learners.
    :type learners: list

    :param meta_learner: meta learner (default: :class:`~Orange.classification.bayes.NaiveLearner`).
    :type meta_learner: :class:`~Orange.classification.Learner`

    :param folds: number of iterations (folds) of cross-validation to assemble class probability data for meta learner.

    :param name: learner name (default: stacking).
    :type name: string

    :rtype: :class:`~Orange.ensemble.stacking.StackedClassificationLearner` or
        :class:`~Orange.ensemble.stacking.StackedClassifier`
    """
    def __new__(cls, learners, data=None, weight=0, **kwds):
        if data is None:
            self = Orange.classification.Learner.__new__(cls)
            return self
        else:
            self = cls(learners, **kwds)
            return self(data, weight)

    def __init__(self, learners, meta_learner=Orange.classification.bayes.NaiveLearner(), folds=10, name='stacking'):
        self.learners = learners
        self.meta_learner = meta_learner
        self.name = name
        self.folds = folds

    def __call__(self, data, weight=0):
        res = Orange.evaluation.testing.cross_validation(self.learners, data, self.folds)
        features = [Orange.feature.Continuous("%d" % i) for i in range(len(self.learners) * (len(data.domain.class_var.values) - 1))]
        domain = Orange.data.Domain(features + [data.domain.class_var])
        p_data = Orange.data.Table(domain)
        for r in res.results:
            p_data.append([p for ps in r.probabilities for p in list(ps)[:-1]] + [r.actual_class])
        meta_classifier = self.meta_learner(p_data)

        classifiers = [l(data, weight) for l in self.learners]
        feature_domain = Orange.data.Domain(features)
        return StackedClassifier(classifiers, meta_classifier, name=self.name)

class StackedClassifier:
    """
    A classifier for stacking. Uses a set of level-0 classifiers to induce class probabilities, which
    are an input to a meta-classifier to predict class probability for a given data instance.

    :param classifiers: a list of level-0 classifiers.
    :type classifiers: list

    :param meta_classifier: meta-classifier.
    :type meta_classifier: :class:`~Orange.classification.Classifier`
    """
    def __init__(self, classifiers, meta_classifier, **kwds):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.domain = Orange.data.Domain(self.meta_classifier.domain.features, False)
        self.__dict__.update(kwds)

    def __call__(self, instance, resultType=Orange.core.GetValue):
        ps = Orange.data.Instance(self.domain, [p for cl in self.classifiers for p in list(cl(instance, Orange.core.GetProbabilities))[:-1]])
        return self.meta_classifier(ps, resultType)
