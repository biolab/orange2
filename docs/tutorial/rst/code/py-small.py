import Orange

class SmallLearner(Orange.classification.PyLearner):
    def __init__(self, base_learner=Orange.classification.bayes.NaiveLearner,
                 name='small', m=5):
        self.name = name
        self.m   = m
        self.base_learner = base_learner

    def __call__(self, data, weight=None):
        gain = Orange.feature.scoring.InfoGain()
        m = min(self.m, len(data.domain.features))
        best = [f for _, f in sorted((gain(x, data), x) for x in data.domain.features)[-m:]]
        domain = Orange.data.Domain(best + [data.domain.class_var])

        model = self.base_learner(Orange.data.Table(domain, data), weight)
        return Orange.classification.PyClassifier(classifier=model, name=self.name)

class OptimizedSmallLearner(Orange.classification.PyLearner):
    def __init__(self, name="opt_small", ms=range(1,30,3)):
        self.ms = ms
        self.name = name

    def __call__(self, data, weight=None):
        scores = []
        for m in self.ms:
            res = Orange.evaluation.testing.cross_validation([SmallLearner(m=m)], data, folds=5)
            scores.append((Orange.evaluation.scoring.AUC(res)[0], m))
        _, best_m = max(scores)

        return SmallLearner(data, m=best_m)

data = Orange.data.Table("promoters")
s_learner = SmallLearner(m=3)
classifier = s_learner(data)
print classifier(data[20])
print classifier(data[20], Orange.classification.Classifier.GetProbabilities)

nbc = Orange.classification.bayes.NaiveLearner(name="nbc")
s_learner = SmallLearner(m=3)
o_learner = OptimizedSmallLearner()

learners = [o_learner, s_learner, nbc]
res = Orange.evaluation.testing.cross_validation(learners, data, folds=10)
print ", ".join("%s: %.3f" % (l.name, s) for l, s in zip(learners, Orange.evaluation.scoring.AUC(res)))

