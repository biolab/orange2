# Description: Demonstrates the use of random forests from orngEnsemble module
# Category:    classification, ensembles
# Classes:     RandomForestLearner
# Uses:        bupa.tab
# Referenced:  orngEnsemble.htm

import orange, orngTree, orngEnsemble

data = orange.ExampleTable('bupa.tab')
import random
forest = orngEnsemble.RandomForestLearner(trees=50, name="forest")
tree = orngTree.TreeLearner(min_instances=2, m_for_prunning=2, \
                            same_majority_pruning=True, name='tree')
learners = [tree, forest]

import orngTest, orngStat
results = orngTest.crossValidation(learners, data, folds=3)
print "Learner  CA     Brier  AUC"
for i in range(len(learners)):
    print "%-8s %5.3f  %5.3f  %5.3f" % (learners[i].name, \
        orngStat.CA(results)[i], 
        orngStat.BrierScore(results)[i],
        orngStat.AUC(results)[i])

