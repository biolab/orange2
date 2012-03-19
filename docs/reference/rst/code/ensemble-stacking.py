import Orange

data = Orange.data.Table("promoters")

bayes = Orange.classification.bayes.NaiveLearner(name="bayes")
tree = Orange.classification.tree.SimpleTreeLearner(name="tree")
lin = Orange.classification.svm.LinearLearner(name="lr")
knn = Orange.classification.knn.kNNLearner(name="knn")

base_learners = [bayes, tree, lin, knn]
stack = Orange.ensemble.stacking.StackedClassificationLearner(base_learners)

learners = [stack, bayes, tree, lin, knn]
res = Orange.evaluation.testing.cross_validation(learners, data, 3)
print "\n".join(["%8s: %5.3f" % (l.name, r) for r, l in zip(Orange.evaluation.scoring.CA(res), learners)])
