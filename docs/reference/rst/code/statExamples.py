# Description: Demostrates the use of classification scores
# Category:    evaluation
# Uses:        voting.tab
# Referenced:  orngStat.htm

import orange, orngTest, orngTree

learners = [orange.BayesLearner(name = "bayes"),
            orngTree.TreeLearner(name="tree"),
            orange.MajorityLearner(name="majrty")]

voting = orange.ExampleTable("voting")
res = orngTest.crossValidation(learners, voting)

vehicle = orange.ExampleTable("vehicle")
resVeh = orngTest.crossValidation(learners, vehicle)

import orngStat

CAs = orngStat.CA(res)
APs = orngStat.AP(res)
Briers = orngStat.BrierScore(res)
ISs = orngStat.IS(res)

print
print "method\tCA\tAP\tBrier\tIS"
for l in range(len(learners)):
    print "%s\t%5.3f\t%5.3f\t%5.3f\t%6.3f" % (learners[l].name, CAs[l], APs[l], Briers[l], ISs[l])


CAs = orngStat.CA(res, reportSE=True)
APs = orngStat.AP(res, reportSE=True)
Briers = orngStat.BrierScore(res, reportSE=True)
ISs = orngStat.IS(res, reportSE=True)

print
print "method\tCA\tAP\tBrier\tIS"
for l in range(len(learners)):
    print "%s\t%5.3f+-%5.3f\t%5.3f+-%5.3f\t%5.3f+-%5.3f\t%6.3f+-%5.3f" % ((learners[l].name, ) + CAs[l] + APs[l] + Briers[l] + ISs[l])


print
cm = orngStat.confusionMatrices(res)[0]
print "Confusion matrix for naive Bayes:"
print "TP: %i, FP: %i, FN: %s, TN: %i" % (cm.TP, cm.FP, cm.FN, cm.TN)

print
cm = orngStat.confusionMatrices(res, cutoff=0.2)[0]
print "Confusion matrix for naive Bayes:"
print "TP: %i, FP: %i, FN: %s, TN: %i" % (cm.TP, cm.FP, cm.FN, cm.TN)

print
cm = orngStat.confusionMatrices(resVeh, vehicle.domain.classVar.values.index("van"))[0]
print "Confusion matrix for naive Bayes for 'van':"
print "TP: %i, FP: %i, FN: %s, TN: %i" % (cm.TP, cm.FP, cm.FN, cm.TN)

print
cm = orngStat.confusionMatrices(resVeh, vehicle.domain.classVar.values.index("opel"))[0]
print "Confusion matrix for naive Bayes for 'opel':"
print "TP: %i, FP: %i, FN: %s, TN: %i" % (cm.TP, cm.FP, cm.FN, cm.TN)

print
cm = orngStat.confusionMatrices(resVeh)[0]
classes = vehicle.domain.classVar.values
print "\t"+"\t".join(classes)
for className, classConfusions in zip(classes, cm):
    print ("%s" + ("\t%i" * len(classes))) % ((className, ) + tuple(classConfusions))

cm = orngStat.confusionMatrices(res)
print
print "Sensitivity and specificity for 'voting'"
print "method\tsens\tspec"
for l in range(len(learners)):
    print "%s\t%5.3f\t%5.3f" % (learners[l].name, orngStat.sens(cm[l]), orngStat.spec(cm[l]))

cm = orngStat.confusionMatrices(resVeh, vehicle.domain.classVar.values.index("van"))
print
print "Sensitivity and specificity for 'vehicle=van'"
print "method\tsens\tspec"
for l in range(len(learners)):
    print "%s\t%5.3f\t%5.3f" % (learners[l].name, orngStat.sens(cm[l]), orngStat.spec(cm[l]))

print
print "AUC (voting)"

AUCs = orngStat.AUC(res)
for l in range(len(learners)):
    print "%10s: %5.3f" % (learners[l].name, AUCs[l])


print
print "AUC for vehicle using weighted single-out method"
print "bayes\ttree\tmajority"
AUCs = orngStat.AUC(resVeh, orngStat.AUC.WeightedOneAgainstAll)
print "%5.3f\t%5.3f\t%5.3f" % tuple(AUCs)

print
print "AUC for vehicle, using different methods"
methods = ["by pairs, weighted", "by pairs", "one vs. all, weighted", "one vs. all"]
print " " *25 + "  \tbayes\ttree\tmajority"
for i in range(4):
    AUCs = orngStat.AUC(resVeh, i)
    print "%25s: \t%5.3f\t%5.3f\t%5.3f" % ((methods[i], ) + tuple(AUCs))


classes = vehicle.domain.classVar.values
classDist = orange.Distribution(vehicle.domain.classVar, vehicle)

print
print "AUC for detecting class 'van' in 'vehicle'"
AUCs = orngStat.AUC_single(resVeh, classIndex = vehicle.domain.classVar.values.index("van"))
print "%5.3f\t%5.3f\t%5.3f" % tuple(AUCs)

print
print "AUCs for detecting various classes in 'vehicle'"
for c,s in enumerate(classes):
    print "%s (%5.3f) vs others: \t%5.3f\t%5.3f\t%5.3f" % ((s, classDist[c] ) + tuple(orngStat.AUC_single(resVeh, c)))

print
classes = vehicle.domain.classVar.values
AUCmatrix = orngStat.AUC_matrix(resVeh)[0]
print "\t"+"\t".join(classes[:-1])
for className, AUCrow in zip(classes[1:], AUCmatrix[1:]):
    print ("%s" + ("\t%5.3f" * len(AUCrow))) % ((className, ) + tuple(AUCrow))

print
print "AUCs for detecting various pairs of classes in 'vehicle'"
for c1, s1 in enumerate(classes):
    for c2 in range(c1):
        print "%s vs %s: \t%5.3f\t%5.3f\t%5.3f" % ((s1, classes[c2]) + tuple(orngStat.AUC_pair(resVeh, c1, c2)))


ri2 = orange.MakeRandomIndices2(voting, 0.6)
train = voting.selectref(ri2, 0)
test = voting.selectref(ri2, 1)
res1 = orngTest.learnAndTestOnTestData(learners, train, test)

print
print "AUC and SE for voting"
AUCs = orngStat.AUCWilcoxon(res1)
for li, lrn in enumerate(learners):
    print "%s: %5.3f+-%5.3f" % (lrn.name, AUCs[li][0], AUCs[li][1])

print
print "Difference between naive Bayes and tree: %5.3f+-%5.3f" % tuple(orngStat.compare2AUCs(res1, 0, 1)[2])

print
print "ROC (first 20 points) for bayes on 'voting'"
ROC_bayes = orngStat.computeROC(res1)[0]
for t in ROC_bayes[:20]:
    print "%5.3f\t%5.3f" % t
