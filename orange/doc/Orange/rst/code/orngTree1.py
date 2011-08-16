import Orange

data = Orange.data.Table("iris")
tree = Orange.classification.tree.TreeLearner(data, max_depth=3)

formats = ["", "%V (%M out of %N)", "%V (%^MbA%, %^MbP%)",
           '%C="Iris-versicolor" (%^c="Iris-versicolor"% of node, %^CbA="Iris-versicolor"% of versicolors)',
           "%D", "%.2d"]

for format in formats:
    print '\n\n*** FORMAT: "%s"\n' % format
    print tree.dump(leaf_str=format)

formats2 = [("%V", "."), ('%^.1CbA="Iris-virginica"% (%^.1CbP="Iris-virginica"%)', '.'), ("%V   %D %.2DbP %.2dbP", "%D %.2DbP %.2dbP")]
for fl, fn in formats2:
    print tree.dump(leaf_str=fl, node_str=fn)


data = Orange.data.Table("housing")
tree = Orange.classification.tree.TreeLearner(data, max_depth=3)
formats = ["", "%V"]
for format in formats:
    print '\n\n*** FORMAT: "%s"\n' % format
    print tree.dump(leaf_str=format)

formats2 = [("[SE: %E]\t %V %I(90)", "[SE: %E]"), ("%C<22 (%cbP<22)", "."), ("%C![20,22] (%^cbP![20,22]%)", ".")]
for fl, fn in formats2:
    print tree.dump(leaf_str=fl, node_str=fn)
