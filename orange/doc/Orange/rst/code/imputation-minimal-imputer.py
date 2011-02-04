# Description: Shows how to impute missing values using the 
# Category:    imputation
# Uses:        voting
# Referenced:  Orange.feature.html#imputation
# Classes:     Orange.feature.imputation.ImputeLearner, Orange.feature.imputation.ImputerConstructor_minimal

import Orange

ba = Orange.classification.bayes.NaiveLearner()
imba = Orange.feature.imputation.ImputeLearner(baseLearner=ba, 
       imputerConstructor=Orange.feature.imputation.ImputerConstructor_minimal)

table = Orange.data.Table("voting")
res = Orange.evaluation.testing.crossValidation([ba, imba], table)
CAs = Orange.evaluation.scoring.CA(res)

print "Without imputation: %5.3f" % CAs[0]
print "With imputation: %5.3f" % CAs[1]