from Orange import data
from Orange.classification import svm

table = data.Table("vehicle.tab")

svm_easy = svm.SVMLearnerEasy(name="svm easy", folds=3)
svm_normal = svm.SVMLearner(name="svm")
learners = [svm_easy, svm_normal]

import orngStat, orngTest

results = orngTest.crossValidation(learners, table, folds=5)
print "Name     CA        AUC"
for learner, CA, AUC in zip(learners, orngStat.CA(results), orngStat.AUC(results)):
    print "%-8s %.2f      %.2f" % (learner.name, CA, AUC)

