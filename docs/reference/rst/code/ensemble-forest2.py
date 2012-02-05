# Description: Defines a tree learner (trunks of depth less than 5) and uses them in forest tree, prints out the number of nodes in each tree
# Category:    classification, ensembles
# Classes:     RandomForestLearner
# Uses:        bupa.tab
# Referenced:  orngEnsemble.htm

import Orange

bupa = Orange.data.Table('bupa.tab')

tree = Orange.classification.tree.TreeLearner()
tree.minExamples = 5
tree.maxDepth = 5

forest_learner = Orange.ensemble.forest.RandomForestLearner(base_learner=tree, trees=50, attributes=3)
forest = forest_learner(bupa)

for c in forest.classifiers:
    print c.countNodes(),
print
