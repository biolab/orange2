import Orange

data = Orange.data.Table("housing.tab")
tree = Orange.regression.tree.TreeLearner(data, m_pruning=2., min_instances=20)
print tree.to_string()
