# Description: Builds a classification tree, and prunes it using minimal error
#              prunning with different values of parameter m. Prints
#              out m and the size of the tree.
#              a tree in text and dot format
# Category:    modelling
# Uses:        iris
# Referenced:  orngTree.htm

import orange, orngTree
data = orange.ExampleTable("../datasets/adult_sample.tab")

tree = orange.TreeLearner(data)
prunner = orange.TreePruner_m()
trees = [(0, tree.tree)]
for m in [0.0, 0.1, 0.5, 1, 5, 10, 50, 100]:
    prunner.m = m
    trees.append((m, prunner(tree)))

for m, t in trees:
    print "m = %5.3f: %i nodes, %i leaves" % (m, t.treesize(), orngTree.countLeaves(t))
