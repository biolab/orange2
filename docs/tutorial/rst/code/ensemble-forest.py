import Orange

data = Orange.data.Table("promoters")

bayes = Orange.classification.bayes.NaiveLearner(name="bayes")
knn = Orange.classification.knn.kNNLearner(name="knn")
forest = Orange.ensemble.forest.RandomForestLearner(name="forest")

learners = [forest, bayes, knn]
res = Orange.evaluation.testing.cross_validation(learners, data, 5)
print "\n".join(["%6s: %5.3f" % (l.name, r) for r, l in zip(Orange.evaluation.scoring.AUC(res), learners)])