import Orange
import Orange.core as orange

data = Orange.data.Table('bupa.tab')

tree = Orange.classification.tree.TreeLearner(storeNodeClassifier = 0, 
    storeContingencies=0, storeDistributions=1, minExamples=5, ).instance()
gini = orange.MeasureAttribute_gini()
tree.split.discreteSplitConstructor.measure = \
  tree.split.continuousSplitConstructor.measure = gini
tree.maxDepth = 5
tree.split = Orange.ensemble.SplitConstructor_AttributeSubset(tree.split, 3)

forestLearner = Orange.ensemble.forest.RandomForestLearner(learner=tree, trees=50)
forest = forestLearner(data)

for c in forest.classifiers:
    print Orange.classification.tree.countNodes(c),
print