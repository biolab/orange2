# Description: Builds a regression tree and prints it out
# Category:    modelling
# Uses:        housing
# Classes:     orngTree.TreeLearner
# Referenced:  regression.htm

import orange, orngTree

data = orange.ExampleTable("../datasets/housing.tab")
rt = orngTree.TreeLearner(data, measure="retis", mForPruning=2, minExamples=20)
orngTree.printTxt(rt, leafStr="%V %I")
