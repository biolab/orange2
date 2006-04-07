# Author:      J Zabkar
# Version:     1.0
# Description: Pre-prunning of classification tree using orngTree module
# Category:    modelling
# Uses:        iris.tab
# Referenced:   orngTree.htm

import orange, orngTree
data = orange.ExampleTable("iris.tab")

print "BIG TREE:"
tree1 = orngTree.TreeLearner(data)
orngTree.printTree(tree1, leafStr="%m", nodeStr=".")

print "\nPRE-PRUNED TREE:"
tree2 = orngTree.TreeLearner(data, maxMajority=0.7)
orngTree.printTree(tree2, leafStr="%m", nodeStr=".")
