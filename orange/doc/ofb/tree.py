# Author:      B Zupan
# Version:     1.0
# Description: Learn decision tree from data and output class probabilities for first few instances
# Category:    modelling
# Uses:        voting.tab

import orange, orngTree
data = orange.ExampleTable("voting")

tree = orngTree.TreeLearner(data, sameMajorityPruning=1, mForPruning=2)
print "Possible classes:", data.domain.classVar.values
print "Probabilities for democrats:"
for i in range(5):
    p = tree(data[i], orange.GetProbabilities)
    print "%d: %5.3f (originally %s)" % (i+1, p[1], data[i].getclass())

orngTree.printTxt(tree)
#orngTree.printDot(tree, fileName='tree.dot', internalNodeShape="ellipse", leafShape="box")

