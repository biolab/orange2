# Description: Test for bagging as defined in bagging.py
# Category:    modelling
# Uses:        adult_sample.tab, bagging.py
# Referenced:  c_bagging.htm
# Classes:     orngTest.crossValidation

import bagging
import Orange
data = Orange.data.Table("adult_sample.tab")

tree = Orange.classification.tree.TreeLearner(mForPrunning=10, minExamples=30)
tree.name = "tree"
baggedTree = bagging.Learner(learner=tree, t=5)

learners = [tree, baggedTree]

results = Orange.evaluation.testing.cross_validation(learners, data, folds=5)
for i in range(len(learners)):
    print "%s: %5.3f" % (learners[i].name, Orange.evaluation.scoring.CA(results)[i])
