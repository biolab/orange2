import Orange

learners = [Orange.classification.bayes.NaiveLearner(name="bayes"),
            Orange.classification.tree.TreeLearner(name="tree"),
            Orange.classification.majority.MajorityLearner(name="majority")]

voting = Orange.data.Table("voting")
res = Orange.evaluation.testing.cross_validation(learners, voting)
CAs = Orange.evaluation.scoring.CA(res)
AUCs = Orange.evaluation.scoring.AUC(res)

print "%10s  %5s %5s" % ("Learner", "AUC", "CA")
for l, _ in enumerate(learners):
    print "%10s: %5.3f %5.3f" % (learners[l].name, AUCs[l], CAs[l])