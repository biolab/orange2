# Description: FreeViz classifier
# Category:    projection
# Uses:        zoo
# Referenced:  Orange.projection.linear
# Classes:     Orange.projection.linear.FreeViz, Orange.projection.linear.FreeVizLearner, Orange.projection.linear.FreeVizClassifier

import Orange
zoo = Orange.data.Table('zoo')

ind = Orange.data.sample.SubsetIndices2(p0=0.9)(zoo)
train, test = zoo.select(ind, 0), zoo.select(ind, 1)

l = Orange.projection.linear.FreeVizLearner()
c = l(train)
for e in test:
    print c(e, Orange.classification.Classifier.GetBoth)