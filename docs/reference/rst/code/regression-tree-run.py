# Description: Regression trees on servo dataset
# Category:    regression
# Uses:        servo
# Referenced:  Orange.regression.tree
# Classes:     Orange.regression.tree.TreeLearner

import Orange
table = Orange.data.Table("servo.tab")
tree = Orange.regression.tree.TreeLearner(table)
print tree