# Author:      J Zabkar
# Version:     1.0
# Description: Demostration of use of orngEnsemble module
# Category:    modelling
# Uses:        iris.tab

import orange, orngEnsemble, orngTree
import orngTest, orngStat

tree = orngTree.TreeLearner(mForPruning=2)
tree.name = "tree"
bs = orngEnsemble.BoostedLearner(tree)
bs.name = "boosted tree"
bg = orngEnsemble.BaggedLearner(tree)
bg.name = "bagged tree"

data = orange.ExampleTable("iris.tab")

learners = [tree, bs, bg]
results = orngTest.crossValidation(learners, data)
print "Classification Accuracy:"
for i in range(len(learners)):
    print ("%15s: %5.3f") % (learners[i].name, orngStat.CA(results)[i])
