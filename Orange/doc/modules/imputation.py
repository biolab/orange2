import orange, orngImpute, orngTest, orngStat

data = orange.ExampleTable("voting")
ba = orange.BayesLearner()
imba = orngImpute.ImputeLearner(baseLearner = ba, imputerConstructor=orange.ImputerConstructor_minimal)
res = orngTest.crossValidation([ba, imba], data)
CAs = orngStat.CA(res)

print "Without imputation: %5.3f" % CAs[0]
print "With imputation: %5.3f" % CAs[1]