import orange, orngTree
data = orange.ExampleTable("../datasets/adult_sample.tab")

tree = orange.TreeLearner(data)
prunner = orange.TreePruner_m()
trees = [(0, tree.tree)] + [(m, prunner(tree, m=m)) for m in [0.0, 0.1, 0.5, 1, 5, 10, 50, 100]]

for m, t in trees:
    print "m = %5.3f: %i nodes, %i leaves" % (m, t.treesize(), orngTree.countLeaves(t))
