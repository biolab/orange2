# Author:      J Zabkar
# Version:     1.0
# Description: Prepruning of classification tree using a worstAcceptable attribute in orngTree module
# Category:    modelling
# Uses:        iris.tab

import orange, orngTree
data = orange.ExampleTable("iris.tab")

tree = orngTree.TreeLearner(data, worstAcceptable=0.6)
orngTree.printTxt(tree)
orngTree.printDot(tree, fileName="tree5.dot")
