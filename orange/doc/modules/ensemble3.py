# Description: Defines a tree learner (trunks of depth less than 5) and uses them in forest tree, prints out the number of nodes in each tree
# Category:    classification, ensembles
# Classes:     RandomForestLearner
# Uses:        bupa.tab
# Referenced:  orngEnsemble.htm

import orange, orngTree, orngEnsemble

data = orange.ExampleTable('bupa.tab')

tree = orngTree.TreeLearner(storeNodeClassifier = 0, storeContingencies=0, \
  storeDistributions=1, minExamples=5, ).instance()
gini = orange.MeasureAttribute_gini()
tree.split.discreteSplitConstructor.measure = \
  tree.split.continuousSplitConstructor.measure = gini
tree.maxDepth = 5
tree.split = orngEnsemble.SplitConstructor_AttributeSubset(tree.split, 3)

forestLearner = orngEnsemble.RandomForestLearner(learner=tree, trees=50)
forest = forestLearner(data)

for c in forest.classifiers:
    print orngTree.countNodes(c),
print
