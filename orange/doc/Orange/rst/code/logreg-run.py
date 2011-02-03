from Orange import *

table = data.Table("titanic")
lr = classification.logreg.LogRegLearner(table)

# compute classification accuracy
correct = 0.0
for ex in table:
    if lr(ex) == ex.getclass():
        correct += 1
print "Classification accuracy:", correct / len(table)
classification.logreg.printOUT(lr)
