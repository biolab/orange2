# Author:      B Zupan
# Version:     1.0
# Description: Learn a decision tree from data and print it out
# Category:    modelling
# Uses:        voting.tab

import orange, orngTree
data = orange.ExampleTable("voting")
tree = orngTree.TreeLearner(data, sameMajorityPruning=1, mForPruning=2)
orngTree.printTxt(tree)
