# Description: Shows how to construct trees
# Category:    learning, decision trees, classification
# Classes:     TreeLearner, TreeClassifier, TreeStopCriteria, TreeStopCriteria_common
# Uses:        lenses
# Referenced:  TreeLearner.htm

import orange

data = orange.ExampleTable("lenses")
learner = orange.TreeLearner()

def printTree0(node, level):
    if not node:
        print " "*level + "<null node>"
        return

    if node.branchSelector:
        nodeDesc = node.branchSelector.classVar.name
        nodeCont = node.distribution
        print "\n" + "   "*level + "%s (%s)" % (nodeDesc, nodeCont),
        for i in range(len(node.branches)):
            print "\n" + "   "*level + ": %s" % node.branchDescriptions[i],
            printTree0(node.branches[i], level+1)
    else:
        nodeCont = node.distribution
        majorClass = node.nodeClassifier.defaultValue
        print "--> %s (%s) " % (majorClass, nodeCont),

def printTree(x):
    if type(x) == orange.TreeClassifier:
        printTree0(x.tree, 0)
    elif type(x) == orange.TreeNode:
        printTree0(x, 0)
    else:
        raise TypeError, "invalid parameter"

print learner.split
learner(data)
print learner.split

learner.stop = orange.TreeStopCriteria_common()
print learner.stop.maxMajority, learner.stop.minExamples

print "\n\nTree with minExamples = 5.0"
learner.stop.minExamples = 5.0
tree = learner(data)
printTree(tree)

print "\n\nTree with maxMajority = 0.5"
learner.stop.maxMajority = 0.5
tree = learner(data)
printTree(tree)
