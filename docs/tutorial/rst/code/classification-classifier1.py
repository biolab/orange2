import Orange

data = Orange.data.Table("voting")
classifier = Orange.classification.bayes.NaiveLearner(data)
for d in data[:5]:
    c = classifier(d)
    print "%10s; originally %s" % (classifier(d), d.getclass())
