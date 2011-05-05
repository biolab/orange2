# Author:      J Zabkar
# Version:     1.0
# Description: Pre-prunning of classification tree using orngTree module
# Category:    modelling
# Uses:        iris.tab
# Referenced:   orngTree.htm

import Orange
data = Orange.data.Table("iris.tab")

print "BIG TREE:"
tree1 = Orange.classification.tree.TreeLearner(data)
print tree1.dump(leafStr="%m", nodeStr=".")

print "\nPRE-PRUNED TREE:"
tree2 = Orange.classification.tree.TreeLearner(data, maxMajority=0.7)
print tree2.dump(leafStr="%m", nodeStr=".")
