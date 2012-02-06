import Orange

adult = Orange.data.Table("adult_sample")
lr = Orange.classification.logreg.LogRegLearner(adult, removeSingular=1)

for ex in adult[:5]:
    print ex.getclass(), lr(ex)
Orange.classification.logreg.dump(lr)
