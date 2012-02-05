# Description: Shows how to impute missing values using the 
# Category:    imputation
# Uses:        voting
# Referenced:  Orange.feature.html#imputation
# Classes:     Orange.feature.imputation.ImputeLearner, Orange.feature.imputation.ImputerConstructor_minimal

import Orange

ba = Orange.classification.bayes.NaiveLearner()
imba = Orange.feature.imputation.ImputeLearner(base_learner=ba, 
       imputer_constructor=Orange.feature.imputation.ImputerConstructor_minimal)

voting = Orange.data.Table("voting")
res = Orange.evaluation.testing.cross_validation([ba, imba], voting)
CAs = Orange.evaluation.scoring.CA(res)

print "Without imputation: %5.3f" % CAs[0]
print "With imputation: %5.3f" % CAs[1]
