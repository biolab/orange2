# Description: Shows how to "learn" the majority class and compare other classifiers to the default classification
# Category:    default classification accuracy, statistics
# Classes:     MajorityLearner, Orange.evaluate.crossValidation
# Uses:        monks-1
# Referenced:  majority.htm

import Orange
import orngTest, orngStat

table = Orange.data.Table("monks-1")

treeLearner = Orange.classification.tree.TreeLearner()
bayesLearner = Orange.classification.bayes.NaiveLearner()
majorityLearner = Orange.classification.majority.MajorityLearner()
learners = [treeLearner, bayesLearner, majorityLearner]

res = orngTest.crossValidation(learners, table)
CAs = orngStat.CA(res, reportSE = 1)

print "Tree:    %5.3f+-%5.3f" % CAs[0]
print "Bayes:   %5.3f+-%5.3f" % CAs[1]
print "Default: %5.3f+-%5.3f" % CAs[2]
