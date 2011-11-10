import Orange

table = Orange.data.Table("bupa")
ri2 = Orange.core.MakeRandomIndices2(table, 0.7)
train = table.select(ri2, 0)
test = table.select(ri2, 1)

bayes = Orange.classification.bayes.NaiveLearner(train)

thresholds = [.2, .5, .8]
models = [Orange.optimization.ThresholdClassifier(bayes, thr) for thr in thresholds]

res = Orange.evaluation.testing.test_on_data(models, test)
cm = Orange.evaluation.scoring.confusion_matrices(res)

print
for i, thr in enumerate(thresholds):
    print "%1.2f: TP %5.3f, TN %5.3f" % (thr, cm[i].TP, cm[i].TN)
