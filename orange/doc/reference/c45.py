# Description: Shows how to use C4.5 learner
# Category:    learning
# Classes:     C45Learner, C45Classifier
# Uses:        iris
# Referenced:  C45Learner.htm

import orange

#data = orange.ExampleTable("lenses.tab")
data = orange.ExampleTable("iris")
tree = orange.C45Learner(data)

print "\n\nC4.5 with default arguments"
for i in data[:5]:
    print tree(i), i.getclass()

print "\n\nC4.5 with m=100"
tree = orange.C45Learner(data, m=100)
for i in data[:5]:
    print tree(i), i.getclass()

print "\n\nC4.5 with minObjs=100"
tree = orange.C45Learner(data, minObjs=100)
for i in data[:5]:
    print tree(i), i.getclass()

print "\n\nC4.5 with -m 1 and -s"
lrn = orange.C45Learner()
lrn.commandline("-m 1 -s")
tree = lrn(data)
for i in data:
    if i.getclass() != tree(i):
        print i, tree(i)


import orngC45
orngC45.printTree(tree)

lrn = orange.C45Learner()
lrn.commandline("-m 1 -s")
lrn.convertToOrange = 1
tree = lrn(data)
for i in data:
    if i.getclass() != tree(i):
        print i, tree(i)

import orngTree
reload(orngTree)
orngTree.printTxt(tree, leafFields = ["major", "distribution"])