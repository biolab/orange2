# Description: Demostrates the use of classification scores
# Category:    evaluation
# Uses:        voting.tab
# Referenced:  orngStat.htm

import Orange

learners = [ Orange.classification.bayes.NaiveLearner(name = "bayes"),
             Orange.classification.tree.TreeLearner(name="tree"),
             Orange.classification.majority.MajorityLearner(name="majrty")]

voting = Orange.data.Table("voting")
res = Orange.evaluation.testing.cross_validation(learners, voting)

vehicle = Orange.data.Table("vehicle")
resVeh = Orange.evaluation.testing.cross_validation(learners, vehicle)

import Orange.evaluation.scoring

CAs = Orange.evaluation.scoring.CA(res)
APs = Orange.evaluation.scoring.AP(res)
Briers = Orange.evaluation.scoring.Brier_score(res)
ISs = Orange.evaluation.scoring.IS(res)

print
print "method\tCA\tAP\tBrier\tIS"
for l in range(len(learners)):
    print "%s\t%5.3f\t%5.3f\t%5.3f\t%6.3f" % (learners[l].name, CAs[l], APs[l], Briers[l], ISs[l])


CAs = Orange.evaluation.scoring.CA(res, reportSE=True)
APs = Orange.evaluation.scoring.AP(res, reportSE=True)
Briers = Orange.evaluation.scoring.Brier_score(res, reportSE=True)
ISs = Orange.evaluation.scoring.IS(res, reportSE=True)

print
print "method\tCA\tAP\tBrier\tIS"
for l in range(len(learners)):
    print "%s\t%5.3f+-%5.3f\t%5.3f+-%5.3f\t%5.3f+-%5.3f\t%6.3f+-%5.3f" % ((learners[l].name, ) + CAs[l] + APs[l] + Briers[l] + ISs[l])


print
cm = Orange.evaluation.scoring.confusion_matrices(res)[0]
print "Confusion matrix for naive Bayes:"
print "TP: %i, FP: %i, FN: %s, TN: %i" % (cm.TP, cm.FP, cm.FN, cm.TN)

print
cm = Orange.evaluation.scoring.confusion_matrices(res, cutoff=0.2)[0]
print "Confusion matrix for naive Bayes:"
print "TP: %i, FP: %i, FN: %s, TN: %i" % (cm.TP, cm.FP, cm.FN, cm.TN)

print
cm = Orange.evaluation.scoring.confusion_matrices(resVeh, vehicle.domain.class_var.values.index("van"))[0]
print "Confusion matrix for naive Bayes for 'van':"
print "TP: %i, FP: %i, FN: %s, TN: %i" % (cm.TP, cm.FP, cm.FN, cm.TN)

print
cm = Orange.evaluation.scoring.confusion_matrices(resVeh, vehicle.domain.class_var.values.index("opel"))[0]
print "Confusion matrix for naive Bayes for 'opel':"
print "TP: %i, FP: %i, FN: %s, TN: %i" % (cm.TP, cm.FP, cm.FN, cm.TN)

print
cm = Orange.evaluation.scoring.confusion_matrices(resVeh)[0]
classes = vehicle.domain.class_var.values
print "\t"+"\t".join(classes)
for className, classConfusions in zip(classes, cm):
    print ("%s" + ("\t%i" * len(classes))) % ((className, ) + tuple(classConfusions))

cm = Orange.evaluation.scoring.confusion_matrices(res)
print
print "Sensitivity and specificity for 'voting'"
print "method\tsens\tspec"
for l in range(len(learners)):
    print "%s\t%5.3f\t%5.3f" % (learners[l].name, Orange.evaluation.scoring.sens(cm[l]), Orange.evaluation.scoring.spec(cm[l]))

cm = Orange.evaluation.scoring.confusion_matrices(resVeh, vehicle.domain.class_var.values.index("van"))
print
print "Sensitivity and specificity for 'vehicle=van'"
print "method\tsens\tspec"
for l in range(len(learners)):
    print "%s\t%5.3f\t%5.3f" % (learners[l].name, Orange.evaluation.scoring.sens(cm[l]), Orange.evaluation.scoring.spec(cm[l]))

print
print "AUC (voting)"

AUCs = Orange.evaluation.scoring.AUC(res)
for l in range(len(learners)):
    print "%10s: %5.3f" % (learners[l].name, AUCs[l])


print
print "AUC for vehicle using weighted single-out method"
print "bayes\ttree\tmajority"
AUCs = Orange.evaluation.scoring.AUC(resVeh, Orange.evaluation.scoring.AUC.WeightedOneAgainstAll)
print "%5.3f\t%5.3f\t%5.3f" % tuple(AUCs)

print
print "AUC for vehicle, using different methods"
methods = ["by pairs, weighted", "by pairs", "one vs. all, weighted", "one vs. all"]
print " " *25 + "  \tbayes\ttree\tmajority"
for i in range(4):
    AUCs = Orange.evaluation.scoring.AUC(resVeh, i)
    print "%25s: \t%5.3f\t%5.3f\t%5.3f" % ((methods[i], ) + tuple(AUCs))


classes = vehicle.domain.class_var.values
classDist = Orange.statistics.distribution.Distribution(vehicle.domain.class_var, vehicle)

print
print "AUC for detecting class 'van' in 'vehicle'"
AUCs = Orange.evaluation.scoring.AUC_single(resVeh, classIndex = vehicle.domain.class_var.values.index("van"))
print "%5.3f\t%5.3f\t%5.3f" % tuple(AUCs)

print
print "AUCs for detecting various classes in 'vehicle'"
for c,s in enumerate(classes):
    print "%s (%5.3f) vs others: \t%5.3f\t%5.3f\t%5.3f" % ((s, classDist[c] ) + tuple(Orange.evaluation.scoring.AUC_single(resVeh, c)))

print
classes = vehicle.domain.class_var.values
AUCmatrix = Orange.evaluation.scoring.AUC_matrix(resVeh)[0]
print "\t"+"\t".join(classes[:-1])
for className, AUCrow in zip(classes[1:], AUCmatrix[1:]):
    print ("%s" + ("\t%5.3f" * len(AUCrow))) % ((className, ) + tuple(AUCrow))

print
print "AUCs for detecting various pairs of classes in 'vehicle'"
for c1, s1 in enumerate(classes):
    for c2 in range(c1):
        print "%s vs %s: \t%5.3f\t%5.3f\t%5.3f" % ((s1, classes[c2]) + tuple(Orange.evaluation.scoring.AUC_pair(resVeh, c1, c2)))


ri2 = Orange.data.sample.SubsetIndices2(voting, 0.6)
train = voting.selectref(ri2, 0)
test = voting.selectref(ri2, 1)
res1 = Orange.evaluation.testing.learn_and_test_on_test_data(learners, train, test)

print
print "AUC and SE for voting"
AUCs = Orange.evaluation.scoring.AUCWilcoxon(res1)
for li, lrn in enumerate(learners):
    print "%s: %5.3f+-%5.3f" % (lrn.name, AUCs[li][0], AUCs[li][1])

print
print "Difference between naive Bayes and tree: %5.3f+-%5.3f" % tuple(Orange.evaluation.scoring.compare_2_AUCs(res1, 0, 1)[2])

print
print "ROC (first 20 points) for bayes on 'voting'"
ROC_bayes = Orange.evaluation.scoring.compute_ROC(res1)[0]
for t in ROC_bayes[:20]:
    print "%5.3f\t%5.3f" % t
