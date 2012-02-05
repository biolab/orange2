import Orange

titanic = Orange.data.Table("titanic")
lr = Orange.classification.logreg.LogRegLearner(titanic)

# compute classification accuracy
correct = 0.0
for ex in titanic:
    if lr(ex) == ex.getclass():
        correct += 1
print "Classification accuracy:", correct / len(titanic)
Orange.classification.logreg.dump(lr)
