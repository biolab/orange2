import Orange

learners = [Orange.classification.bayes.NaiveLearner(name="bayes"),
            Orange.classification.tree.TreeLearner(name="tree"),
            Orange.classification.majority.MajorityLearner(name="majority")]

voting = Orange.data.Table("voting")
res = Orange.evaluation.testing.cross_validation(learners, voting)

print "CA =", Orange.evaluation.scoring.CA(res)
print "AUC = ", Orange.evaluation.scoring.AUC(res)