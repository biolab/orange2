# Author:      B Zupan
# Version:     1.0
# Description: Builds a regression tree and prints it out
# Category:    modelling
# Uses:        housing

import orange, orngTree

data = orange.ExampleTable("../datasets/housing.tab")
rt = orngTree.TreeLearner(data, measure="retis", mForPruning=2, minExamples=20)
orngTree.printTxt(rt, leafFields=["average","confidenceInterval"], decimalPlaces=1, confidenceLevel=0.85)
