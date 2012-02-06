# Description: Shows how to build a learner that imputes prior to learning.
# Category:    imputation
# Uses:        bridges
# Referenced:  Orange.feature.html#imputation
# Classes:     Orange.feature.imputation.ImputeLearner, Orange.feature.imputation.ImputerConstructor_minimal

import Orange

lr = Orange.classification.logreg.LogRegLearner()
imputer = Orange.feature.imputation.ImputerConstructor_minimal

imlr = Orange.feature.imputation.ImputeLearner(base_learner=lr,
    imputer_constructor=imputer)

voting = Orange.data.Table("voting")
res = Orange.evaluation.testing.cross_validation([lr, imlr], voting)
CAs = Orange.evaluation.scoring.CA(res)

print "Without imputation: %5.3f" % \
      CAs[0]
print "With imputation: %5.3f" % CAs[1]
