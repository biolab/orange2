# Description: Demonstrates the use of random forests from Orange.ensemble.forest module
# Category:    classification, ensembles
# Classes:     RandomForestLearner
# Uses:        bupa.tab
# Referenced:  orngEnsemble.htm

import Orange, orngTree

table = Orange.data.Table('bupa.tab')
forest = Orange.ensemble.forest.RandomForestLearner(trees=50, name="forest")
tree = orngTree.TreeLearner(minExamples=2, mForPrunning=2, \
                            sameMajorityPruning=True, name='tree')
learners = [tree, forest]

import orngTest, orngStat
results = orngTest.crossValidation(learners, table, folds=3)
print "Learner  CA     Brier  AUC"
for i in range(len(learners)):
    print "%-8s %5.3f  %5.3f  %5.3f" % (learners[i].name, \
        orngStat.CA(results)[i], 
        orngStat.BrierScore(results)[i],
        orngStat.AUC(results)[i])
