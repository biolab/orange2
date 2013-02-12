import Orange

data = Orange.data.Table("housing.tab")

lin = Orange.regression.linear.LinearRegressionLearner()
lin.name = "lin"
earth = Orange.regression.earth.EarthLearner()
earth.name = "mars"
tree = Orange.regression.tree.TreeLearner(m_pruning = 2)
tree.name = "tree"

learners = [lin, earth, tree]

res = Orange.evaluation.testing.cross_validation(learners, data, folds=5)
rmse = Orange.evaluation.scoring.RMSE(res)

print "Learner  RMSE"
for i in range(len(learners)):
    print "{0:8}".format(learners[i].name),
    print "%.2f" % rmse[i]