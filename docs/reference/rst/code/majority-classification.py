# Description: Shows how to "learn" the majority class and compare other classifiers to the default classification
# Category:    default classification accuracy, statistics
# Classes:     MajorityLearner, Orange.evaluation.testing.cross_validation
# Uses:        monks-1
# Referenced:  majority.htm

import Orange

table = Orange.data.Table("monks-1")

treeLearner = Orange.classification.tree.TreeLearner()
bayesLearner = Orange.classification.bayes.NaiveLearner()
majorityLearner = Orange.classification.majority.MajorityLearner()
learners = [treeLearner, bayesLearner, majorityLearner]

res = Orange.evaluation.testing.cross_validation(learners, table)
CAs = Orange.evaluation.scoring.CA(res, reportSE=True)

print "Tree:    %5.3f+-%5.3f" % CAs[0]
print "Bayes:   %5.3f+-%5.3f" % CAs[1]
print "Default: %5.3f+-%5.3f" % CAs[2]
