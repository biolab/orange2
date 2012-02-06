# Description: Regression trees on servo dataset
# Category:    regression
# Uses:        servo
# Referenced:  Orange.regression.tree
# Classes:     Orange.regression.tree.TreeLearner

import Orange
servo = Orange.data.Table("servo.tab")
tree = Orange.regression.tree.TreeLearner(servo)
print tree