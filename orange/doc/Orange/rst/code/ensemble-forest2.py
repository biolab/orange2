# Description: Defines a tree learner (trunks of depth less than 5) and uses them in forest tree, prints out the number of nodes in each tree
# Category:    classification, ensembles
# Classes:     RandomForestLearner
# Uses:        bupa.tab
# Referenced:  orngEnsemble.htm

import Orange, orngTree

table = Orange.data.Table('bupa.tab')

tree = orngTree.TreeLearner(storeNodeClassifier = 0, storeContingencies=0, \
  storeDistributions=1, minExamples=5, ).instance()
gini = Orange.feature.scoring.Gini()
tree.split.discreteSplitConstructor.measure = \
  tree.split.continuousSplitConstructor.measure = gini
tree.maxDepth = 5
tree.split = Orange.ensemble.forest.SplitConstructor_AttributeSubset(tree.split, 3)

forestLearner = Orange.ensemble.forest.RandomForestLearner(learner=tree, trees=50)
forest = forestLearner(table)

for c in forest.classifiers:
    print orngTree.countNodes(c),
print
