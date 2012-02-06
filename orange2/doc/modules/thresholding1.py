import orange, orngWrap, orngTest, orngStat

data = orange.ExampleTable("bupa")

learner = orange.BayesLearner()
thresh = orngWrap.ThresholdLearner(learner = learner)
thresh80 = orngWrap.ThresholdLearner_fixed(learner = learner, threshold = .8)
res = orngTest.crossValidation([learner, thresh, thresh80], data)
CAs = orngStat.CA(res)

print "W/out threshold adjustement: %5.3f" % CAs[0]
print "With adjusted thredhold: %5.3f" % CAs[1]
print "With threshold at 0.80: %5.3f" % CAs[2]
