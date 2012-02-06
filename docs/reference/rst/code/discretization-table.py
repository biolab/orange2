import Orange
iris = Orange.data.Table("iris.tab")
disc_iris = Orange.feature.discretization.DiscretizeTable(iris,
    method=Orange.feature.discretization.EquiNDiscretization(n_intervals=3))

print "Original data set:"
for e in iris[:3]:
    print e

print "Discretized data set:"
for e in disc_iris[:3]:
    print e
