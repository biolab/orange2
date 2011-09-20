from Orange import data
from Orange.classification import svm

table = data.Table("vehicle.tab")

svm_easy = svm.SVMLearnerEasy(name="svm easy", folds=3)
svm_normal = svm.SVMLearner(name="svm")
learners = [svm_easy, svm_normal]

from Orange import evaluation

results = evaluation.testing.cross_validation(learners, table, folds=5)
print "Name     CA        AUC"
for learner, CA, AUC in zip(learners, evaluation.scoring.CA(results), evaluation.scoring.AUC(results)):
    print "%-8s %.2f      %.2f" % (learner.name, CA, AUC)
