import orange, orngTree
reload(orngTree)

data = orange.ExampleTable("iris")
tree = orngTree.TreeLearner(data, maxDepth=3)

formats = ["", "%V (%M out of %N)", "%V (%^MbA%, %^MbP%)",
           '%C="Iris-versicolor" (%^c="Iris-versicolor"% of node, %^CbA="Iris-versicolor"% of versicolors)',
           "%D", "%.2d"]

for format in formats:
    print '\n\n*** FORMAT: "%s"\n' % format
    orngTree.printTree(tree, leafStr=format)

formats2 = [("%V", "."), ('%^.1CbA="Iris-virginica"% (%^.1CbP="Iris-virginica"%)', '.'), ("%V   %D %.2DbP %.2dbP", "%D %.2DbP %.2dbP")]
for fl, fn in formats2:
    orngTree.printTree(tree, leafStr=fl, nodeStr=fn)


data = orange.ExampleTable("housing")
tree = orngTree.TreeLearner(data, maxDepth=3)
formats = ["", "%V"]
for format in formats:
    print '\n\n*** FORMAT: "%s"\n' % format
    orngTree.printTree(tree, leafStr=format)

formats2 = [("[SE: %E]\t %V %I(90)", "[SE: %E]"), ("%C<22 (%cbP<22)", "."), ("%C![20,22] (%^cbP![20,22]%)", ".")]
for fl, fn in formats2:
    orngTree.printTree(tree, leafStr=fl, nodeStr=fn)
