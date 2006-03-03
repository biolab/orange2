# Description: Learn a decision tree from data and print it out
# Category:    modelling
# Uses:        voting.tab
# Classes:     orngTree.TreeLearner
# Referenced:  c_otherclass.htm

import orange, orngTree
data = orange.ExampleTable("voting")
tree = orngTree.TreeLearner(data, sameMajorityPruning=1, mForPruning=2)
orngTree.printTxt(tree)
