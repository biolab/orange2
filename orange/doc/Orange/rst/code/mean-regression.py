# Description: Shows how to "learn" the mean class and compare other classifiers to the default classification
# Category:    default classification accuracy, statistics
# Classes:     MeanLearner, Orange.evaluation.testing.crossValidation
# Uses:        monks-1
# Referenced:  mean.htm

import Orange

table = Orange.data.Table("housing")

treeLearner = Orange.classification.tree.TreeLearner() #Orange.regression.TreeLearner()
meanLearner = Orange.regression.mean.MeanLearner()
learners = [treeLearner, meanLearner]

res = Orange.evaluation.testing.crossValidation(learners, table)
MSEs = Orange.evaluation.scoring.MSE(res)

print "Tree:    %5.3f" % MSEs[0]
print "Default: %5.3f" % MSEs[1]
