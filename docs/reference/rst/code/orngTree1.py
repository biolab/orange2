import Orange

iris = Orange.data.Table("iris")
tree = Orange.classification.tree.TreeLearner(iris, max_depth=3)

formats = ["", "%V (%M out of %N)", "%V (%^MbA%, %^MbP%)",
           '%C="Iris-versicolor" (%^c="Iris-versicolor"% of node, %^CbA="Iris-versicolor"% of versicolors)',
           "%D", "%.2d"]

for format in formats:
    print '\n\n*** FORMAT: "%s"\n' % format
    print tree.to_string(leaf_str=format)

formats2 = [("%V", "."), ('%^.1CbA="Iris-virginica"% (%^.1CbP="Iris-virginica"%)', '.'), ("%V   %D %.2DbP %.2dbP", "%D %.2DbP %.2dbP")]
for fl, fn in formats2:
    print tree.to_string(leaf_str=fl, node_str=fn)


housing = Orange.data.Table("housing")
tree = Orange.classification.tree.TreeLearner(housing, max_depth=3)
formats = ["", "%V"]
for format in formats:
    print '\n\n*** FORMAT: "%s"\n' % format
    print tree.to_string(leaf_str=format)

formats2 = [("[SE: %E]\t %V %I(90)", "[SE: %E]"), ("%C<22 (%cbP<22)", "."), ("%C![20,22] (%^cbP![20,22]%)", ".")]
for fl, fn in formats2:
    print tree.to_string(leaf_str=fl, node_str=fn)
