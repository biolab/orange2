import Orange

housing = Orange.data.Table("housing.tab")
learner = Orange.regression.tree.SimpleTreeLearner
res = Orange.evaluation.testing.cross_validation([learner], housing)
print Orange.evaluation.scoring.MSE(res)[0]
