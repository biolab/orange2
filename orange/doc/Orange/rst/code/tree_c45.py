# Description: Shows how to use C4.5 learner
# Category:    learning
# Classes:     C45Learner, C45Classifier
# Uses:        iris
# Referenced:  TODO

import Orange

data = Orange.data.Table("iris")
tree = Orange.classification.tree.C45Learner(data)

print "\n\nC4.5 with default arguments"
for i in data[:5]:
    print tree(i), i.getclass()

print "\n\nC4.5 with m=100"
tree = Orange.classification.tree.C45Learner(data, m=100)
for i in data[:5]:
    print tree(i), i.getclass()

print "\n\nC4.5 with minObjs=100"
tree = Orange.classification.tree.C45Learner(data, minObjs=100)
for i in data[:5]:
    print tree(i), i.getclass()

print "\n\nC4.5 with -m 1 and -s"
lrn = Orange.classification.tree.C45Learner()
lrn.commandline("-m 1 -s")
tree = lrn(data)
for i in data:
    if i.getclass() != tree(i):
        print i, tree(i)

tree = Orange.classification.tree.C45Learner(data)
Orange.classification.tree.printTreeC45(tree)
print

import orngStat, orngTest
res = orngTest.crossValidation([Orange.classification.tree.C45Learner(), 
    Orange.classification.tree.C45Learner(convertToOrange=1)], data)
print "Classification accuracy: %5.3f (converted to tree: %5.3f)" % tuple(orngStat.CA(res))
print "Brier score: %5.3f (converted to tree: %5.3f)" % tuple(orngStat.BrierScore(res))
