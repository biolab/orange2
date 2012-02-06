# Description: Shows how to use C4.5 learner
# Category:    learning
# Classes:     C45Learner, C45Classifier
# Uses:        iris
# Referenced:  TODO

import Orange

iris = Orange.data.Table("iris")
tree = Orange.classification.tree.C45Learner(iris)

print "\n\nC4.5 with default arguments"
for i in iris[:5]:
    print tree(i), i.getclass()

print "\n\nC4.5 with m=100"
tree = Orange.classification.tree.C45Learner(iris, m=100)
for i in iris[:5]:
    print tree(i), i.getclass()

print "\n\nC4.5 with minObjs=100"
tree = Orange.classification.tree.C45Learner(iris, minObjs=100)
for i in iris[:5]:
    print tree(i), i.getclass()

print "\n\nC4.5 with -m 1 and -s"
lrn = Orange.classification.tree.C45Learner()
lrn.commandline("-m 1 -s")
tree = lrn(iris)
for i in iris:
    if i.getclass() != tree(i):
        print i, tree(i)

tree = Orange.classification.tree.C45Learner(iris)
print tree
print


res = Orange.evaluation.testing.cross_validation([Orange.classification.tree.C45Learner(), 
    Orange.classification.tree.C45Learner(convertToOrange=1)], iris)
print "Classification accuracy: %5.3f (converted to tree: %5.3f)" % tuple(Orange.evaluation.scoring.CA(res))
print "Brier score: %5.3f (converted to tree: %5.3f)" % tuple(Orange.evaluation.scoring.Brier_score(res))
