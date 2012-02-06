import Orange

table = Orange.data.Table("adult_sample")
lr = Orange.classification.logreg.LogRegLearner(table, remove_singular=1)

for ex in table[:5]:
    print ex.getclass(), lr(ex)
Orange.classification.logreg.dump(lr)
