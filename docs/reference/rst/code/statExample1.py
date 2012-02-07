import Orange

learners = [ Orange.classification.bayes.NaiveLearner(name = "bayes"),
             Orange.classification.tree.TreeLearner(name="tree"),
             Orange.classification.majority.MajorityLearner(name="majrty")]

voting = Orange.data.Table("voting")
res = Orange.evaluation.testing.cross_validation(learners, voting)

vehicle = Orange.data.Table("vehicle")
resVeh = Orange.evaluation.testing.cross_validation(learners, vehicle)

import orngStat

CAs = Orange.evaluation.scoring.CA(res)
APs = Orange.evaluation.scoring.AP(res)
Briers = Orange.evaluation.scoring.Brier_score(res)
ISs = Orange.evaluation.scoring.IS(res)

print
print "method\tCA\tAP\tBrier\tIS"
for l in range(len(learners)):
    print "%s\t%5.3f\t%5.3f\t%5.3f\t%6.3f" % (learners[l].name, CAs[l], APs[l], Briers[l], ISs[l])
