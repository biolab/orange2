import Orange

cm = Orange.misc.CostMatrix(3)
print "classVar =", cm.classVar
for pred in range(3):
    for corr in range(3):
        print cm.getcost(pred, corr),
    print

data = Orange.data.Table("iris")
cm = Orange.misc.CostMatrix(data.domain.classVar, 2)

cm = Orange.misc.CostMatrix(data.domain.classVar, [(0, 2, 1), (2, 0, 1), (2, 2, 0)])

cm = Orange.misc.CostMatrix(data.domain.classVar, 2)
cm.setcost("Iris-setosa", "Iris-virginica", 1)
cm.setcost("Iris-versicolor", "Iris-virginica", 1)

data = Orange.data.Table("lenses")
meas = Orange.feature.scoring.Cost()
meas.cost = ((0, 2, 1), (2, 0, 1), (2, 2, 0))
for attr in data.domain.attributes:
    print "%s: %5.3f" % (attr.name, meas(attr, data))

meas.cost = data.domain.classVar
