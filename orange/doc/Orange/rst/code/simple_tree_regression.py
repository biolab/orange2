import Orange

table = Orange.data.Table("housing.tab")
learner = Orange.regression.tree.SimpleTreeLearner
res = Orange.evaluation.testing.cross_validation([learner], table)
print Orange.evaluation.scoring.MSE(res)[0]
