import orange, orngBayes, orngTest, orngStat

data = orange.ExampleTable("lung-cancer")

bayes = orngBayes.BayesLearner()
bayes_m = orngBayes.BayesLearner(m=2)

res = orngTest.crossValidation([bayes, bayes_m], data)
CAs = orngStat.CA(res)
print
print "Without m: %5.3f" % CAs[0]
print "With m=2: %5.3f" % CAs[1]

data = orange.ExampleTable("voting")
model = orngBayes.BayesLearner(data)
orngBayes.printModel(model)