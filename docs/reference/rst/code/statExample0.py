import Orange

learners = [Orange.classification.bayes.NaiveLearner(name="bayes"),
            Orange.classification.tree.TreeLearner(name="tree"),
            Orange.classification.majority.MajorityLearner(name="majority")]

voting = Orange.data.Table("voting")
res = Orange.evaluation.testing.cross_validation(learners, voting)

print "CA =", ["%.6f" % r for r in Orange.evaluation.scoring.CA(res)]
print "AUC = ", ["%.6f" % r for r in Orange.evaluation.scoring.AUC(res)]
