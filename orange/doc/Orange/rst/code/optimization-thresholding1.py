import Orange

table = Orange.data.Table("bupa")

learner = Orange.classification.bayes.NaiveLearner()
thresh = Orange.optimization.ThresholdLearner(learner=learner)
thresh80 = Orange.optimization.ThresholdLearner_fixed(learner=learner, 
                                                      threshold=0.8)
res = Orange.evaluation.testing.crossValidation([learner, thresh, thresh80], table)
CAs = Orange.evaluation.scoring.CA(res)

print "W/out threshold adjustement: %5.3f" % CAs[0]
print "With adjusted thredhold: %5.3f" % CAs[1]
print "With threshold at 0.80: %5.3f" % CAs[2]
