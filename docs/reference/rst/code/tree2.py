# Author:      J Zabkar
# Version:     1.0
# Description: Pre-prunning of classification tree using orngTree module
# Category:    modelling
# Uses:        iris.tab
# Referenced:   orngTree.htm

import Orange
iris = Orange.data.Table("iris.tab")

print "BIG TREE:"
tree1 = Orange.classification.tree.TreeLearner(iris)
print tree1.to_string(leaf_str="%m", node_str=".")

print "\nPRE-PRUNED TREE:"
tree2 = Orange.classification.tree.TreeLearner(iris, max_majority=0.7)
print tree2.to_string(leaf_str="%m", node_str=".")

