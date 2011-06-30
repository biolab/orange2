# Description: Demonstrates the use of boosting and bagging from orngEnsemble module
# Category:    classification, ensembles
# Classes:     BoostedLearner, BaggedLearner
# Uses:        lymphography.tab
# Referenced:  orngEnsemble.htm

import orange, orngEnsemble, orngTree
import orngTest, orngStat

tree = orngTree.TreeLearner(mForPruning=2, name="tree")
bs = orngEnsemble.BoostedLearner(tree, name="boosted tree")
bg = orngEnsemble.BaggedLearner(tree, name="bagged tree")

data = orange.ExampleTable("lymphography.tab")

learners = [tree, bs, bg]
results = orngTest.crossValidation(learners, data, folds=3)
print "Classification Accuracy:"
for i in range(len(learners)):
    print ("%15s: %5.3f") % (learners[i].name, orngStat.CA(results)[i])
