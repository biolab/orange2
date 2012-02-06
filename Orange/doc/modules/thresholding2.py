import orange, orngWrap, orngTest, orngStat

data = orange.ExampleTable("bupa")
ri2 = orange.MakeRandomIndices2(data, 0.7)
train = data.select(ri2, 0)
test = data.select(ri2, 1)

bayes = orange.BayesLearner(train)

thresholds = [.2, .5, .8]
models = [orngWrap.ThresholdClassifier(bayes, thr) for thr in thresholds]

res = orngTest.testOnData(models, test)
cm = orngStat.confusionMatrices(res)

print
for i, thr in enumerate(thresholds):
    print "%1.2f: TP %5.3f, TN %5.3f" % (thr, cm[i].TP, cm[i].TN)
