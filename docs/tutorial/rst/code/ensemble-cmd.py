import Orange
data = Orange.data.Table("housing")
tree = Orange.classification.tree.TreeLearner()
# btree = Orange.ensemble.boosting.BoostedLearner(tree)
btree = Orange.ensemble.bagging.BaggedLearner(tree)
#btree
#btree(data)
model = btree(data)
print model(data[0])