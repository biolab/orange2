import Orange
import random

data = Orange.data.Table("voting")
test = Orange.data.Table(random.sample(data, 5))
train = Orange.data.Table([d for d in data if d not in test])

tree = Orange.regression.tree.TreeLearner(train, same_majority_pruning=1, m_pruning=2)
tree.name = "tree"
knn = Orange.classification.knn.kNNLearner(train, k=21)
knn.name = "k-NN"
lr = Orange.classification.logreg.LogRegLearner(train)
lr.name = "lr"

classifiers = [tree, knn, lr]

target = 0
print "Probabilities for %s:" % data.domain.class_var.values[target]
print "original class ",
print " ".join("%-9s" % l.name for l in classifiers)

return_type = Orange.classification.Classifier.GetProbabilities
for d in test:
    print "%-15s" % (d.getclass()),
    print "     ".join("%5.3f" % c(d, return_type)[target] for c in classifiers)
