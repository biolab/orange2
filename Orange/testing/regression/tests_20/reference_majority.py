# Description: Shows how to "learn" the majority class and compare other classifiers to the default classification
# Category:    default classification accuracy, statistics
# Classes:     MajorityLearner, DefaultClassifier, orngTest.crossValidation
# Uses:        monk1
# Referenced:  majority.htm

import orange, orngTest, orngStat

data = orange.ExampleTable("monk1")

treeLearner = orange.TreeLearner()
bayesLearner = orange.BayesLearner()
majorityLearner = orange.MajorityLearner()
learners = [treeLearner, bayesLearner, majorityLearner]

res = orngTest.crossValidation(learners, data)
CAs = orngStat.CA(res, reportSE = 1)

print "Tree:    %5.3f+-%5.3f" % CAs[0]
print "Bayes:   %5.3f+-%5.3f" % CAs[1]
print "Default: %5.3f+-%5.3f" % CAs[2]
