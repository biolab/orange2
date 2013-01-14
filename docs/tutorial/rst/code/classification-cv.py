import Orange

data = Orange.data.Table("voting")
bayes = Orange.classification.bayes.NaiveLearner()
res = Orange.evaluation.testing.cross_validation([bayes], data, folds=5)
print "Accuracy: %.2f" % Orange.evaluation.scoring.CA(res)[0]
print "AUC:      %.2f" % Orange.evaluation.scoring.AUC(res)[0]
