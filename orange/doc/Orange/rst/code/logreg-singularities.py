from Orange import *

table = data.Table("adult_sample")
lr = classification.logreg.LogRegLearner(table, removeSingular=1)

for ex in table[:5]:
    print ex.getclass(), lr(ex)
classification.logreg.printOUT(lr)
