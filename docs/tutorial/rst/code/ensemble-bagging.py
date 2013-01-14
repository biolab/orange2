import Orange

data = Orange.data.Table("promoters")

tree = Orange.classification.tree.TreeLearner(m_pruning=2, name="tree")
boost = Orange.ensemble.boosting.BoostedLearner(tree, name="boost")
bagg = Orange.ensemble.bagging.BaggedLearner(tree, name="bagg")

learners = [tree, boost, bagg]
results = Orange.evaluation.testing.cross_validation(learners, data, folds=10)
for l, s in zip(learners, Orange.evaluation.scoring.AUC(results)):
    print "%5s: %.2f" % (l.name, s)
