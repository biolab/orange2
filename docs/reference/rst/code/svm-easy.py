from Orange import data
from Orange.classification import svm

vehicle = data.Table("vehicle.tab")

svm_easy = svm.SVMLearnerEasy(name="svm easy", folds=3)
svm_normal = svm.SVMLearner(name="svm")
learners = [svm_easy, svm_normal]

from Orange.evaluation import testing, scoring

results = testing.cross_validation(learners, vehicle, folds=5)
print "Name     CA        AUC"
for learner,CA,AUC in zip(learners, scoring.CA(results), scoring.AUC(results)):
    print "%-8s %.2f      %.2f" % (learner.name, CA, AUC)
