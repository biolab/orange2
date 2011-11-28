# Description: Shows how to construct trees
# Category:    learning, decision trees, classification
# Classes:     TreeLearner, TreeClassifier, TreeStopCriteria, TreeStopCriteria_common
# Uses:        lenses
# Referenced:  TreeLearner.htm

import Orange

data = Orange.data.Table("lenses")
learner = Orange.classification.tree.TreeLearner()

def printTree0(node, level):
    if not node:
        print " "*level + "<null node>"
        return

    if node.branch_selector:
        node_desc = node.branch_selector.class_var.name
        node_cont = node.distribution
        print "\n" + "   "*level + "%s (%s)" % (node_desc, node_cont),
        for i in range(len(node.branches)):
            print "\n" + "   "*level + ": %s" % node.branch_descriptions[i],
            printTree0(node.branches[i], level+1)
    else:
        node_cont = node.distribution
        major_class = node.node_classifier.default_value
        print "--> %s (%s) " % (major_class, node_cont),

def printTree(x):
    if isinstance(x, Orange.classification.tree.TreeClassifier):
        printTree0(x.tree, 0)
    elif isinstance(x, Orange.classification.tree.Node):
        printTree0(x, 0)
    else:
        raise TypeError, "invalid parameter"

learner.stop = Orange.classification.tree.StopCriteria_common()
print learner.stop.max_majority, learner.stop.min_examples

print "\n\nTree with minExamples = 5.0"
learner.stop.min_examples = 5.0
tree = learner(data)
print tree

print "\n\nTree with maxMajority = 0.5"
learner.stop.max_majority = 0.5
tree = learner(data)
print tree
