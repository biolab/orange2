# Author:      J Zabkar
# Version:     1.0
# Description: Shows the use of some attributes in printing-out the classification trees with orngTree module
# Category:    modelling
# Uses:        iris.tab
# Referenced:   orngTree.htm

import orange, orngTree
data = orange.ExampleTable("iris.tab")

tree = orange.TreeLearner(data)
orngTree.printTxt(tree, internalNodeFields=['contingency','major'], depthLimit=3)
orngTree.printDot(tree, fileName="tree6.dot", internalNodeFields=['contingency','major'], depthLimit=3)
