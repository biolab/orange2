import Orange
import orngTest, orngStat

tree = Orange.classification.tree.TreeLearner(name="tree") # mForPruning=2
bs = Orange.ensemble.boosting.BoostedLearner(tree, name="boosted tree")
bg = Orange.ensemble.bagging.BaggedLearner(tree, name="bagged tree")

data = Orange.data.Table("lymphography.tab")

learners = [tree, bs, bg]
results = orngTest.crossValidation(learners, data)
print "Classification Accuracy:"
for i in range(len(learners)):
    print ("%15s: %5.3f") % (learners[i].name, orngStat.CA(results)[i])