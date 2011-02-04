# Description: Shows how to "learn" the mean class and compare other classifiers to the default classification
# Category:    default classification accuracy, statistics
# Classes:     MeanLearner, Orange.evaluate.crossValidation
# Uses:        monks-1
# Referenced:  mean.htm

import Orange
import orngTest, orngStat

table = Orange.data.Table("housing")

treeLearner = Orange.classification.tree.TreeLearner() #Orange.regression.TreeLearner()
meanLearner = Orange.regression.mean.MeanLearner()
learners = [treeLearner, meanLearner]

res = orngTest.crossValidation(learners, table)
MSEs = orngStat.MSE(res)

print "Tree:    %5.3f" % MSEs[0]
print "Default: %5.3f" % MSEs[1]
