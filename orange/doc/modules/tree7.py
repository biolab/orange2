# Author:      J Zabkar
# Version:     1.0
# Description: Printing out a prunned tree with continuous class values
# Category:    evaluation
# Uses:        housing.tab
# Referenced:   orngTree.htm

import orange, orngTree

train = orange.ExampleTable("../datasets/housing.tab")

l = orngTree.TreeLearner(train, measure="retis", mForPruning=2, minExamples=20)

orngTree.printTxt(l, leafFields=["average","confidenceInterval"],decimalPlaces=1, confidenceLevel=0.85)
orngTree.printDot(l, leafFields=["average","confidenceInterval"], fileName="tree7.dot", decimalPlaces=1, confidenceLevel=0.85)
print
print "Number of nodes:",orngTree.countNodes(l)
print "Number of leaves:",orngTree.countLeaves(l)
