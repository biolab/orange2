# Description: Storing and printing out the examples in classification tree
# Category:    modelling
# Uses:        iris.tab
# Referenced:  orngTree.htm

import orange, orngTree

def printExamples(node):
    if node:
        if node.branches:
            for b in node.branches:
                printExamples(b)
        else:
            print "----------------- NEW NODE -----------------"
            for ex in node.examples:
                print ex
    else:
        print "null node"


data = orange.ExampleTable("iris.tab")
print len(data)

tree = orngTree.TreeLearner(data, storeExamples=1, storeContingencies=1)
print 'CLASSIFICATION TREE:'
orngTree.printTxt(tree)

print '\nEXAMPLES IN NODES:'
printExamples(tree.tree)

print '\nCONTINGENCY:'
print tree.tree.contingency.classes
