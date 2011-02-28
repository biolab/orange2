# Description: Shows the structure that represents decision trees in Orange
# Category:    learning, decision trees, classification
# Classes:     TreeLearner, TreeClassifire, TreeNode, 
# Uses:        lenses
# Referenced:  TreeLearner.htm

import Orange

data = Orange.data.Table("lenses")
treeClassifier = Orange.classification.tree.TreeLearner(data)

def treeSize(node):
    if not node:
        return 0

    size = 1
    if node.branchSelector:
        for branch in node.branches:
            size += treeSize(branch)

    return size

print "Tree size:", treeSize(treeClassifier.tree)


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
    if isinstance(x, Orange.classification.tree.TreeClassifier):
        printTree0(x.tree, 0)
    elif isinstance(x, Orange.classification.tree.Node):
        printTree0(x, 0)
    else:
        raise TypeError, "invalid parameter"

print "\n\nUnpruned tree"
print treeClassifier.dump()

def cutTree(node, level):
    if node and node.branchSelector:
        if level:
            for branch in node.branches:
                cutTree(branch, level-1)
        else:
            node.branchSelector = None
            node.branches = None
            node.branchDescriptions = None

print "\n\nPruned tree"
cutTree(treeClassifier.tree, 2)
print treeClassifier.dump()

