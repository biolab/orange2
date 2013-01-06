import Orange

data = Orange.data.Table("voting")
classifier = Orange.classification.bayes.NaiveLearner(data)
target = 1
print "Probabilities for %s:" % data.domain.class_var.values[target]
for d in data[:5]:
    ps = classifier(d, Orange.classification.Classifier.GetProbabilities)
    print "%5.3f; originally %s" % (ps[target], d.getclass())
