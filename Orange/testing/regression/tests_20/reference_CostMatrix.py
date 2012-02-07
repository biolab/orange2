# Description: Shows how to assess the quality of attributes not in the dataset
# Category:    attribute quality
# Classes:     EntropyDiscretization, MeasureAttribute, MeasureAttribute_info
# Uses:        iris
# Referenced:  MeasureAttribute.htm

import orange

print
print "Default matrix of size 3"
cm = orange.CostMatrix(3)
print "classVar =", cm.classVar
for pred in range(3):
    for corr in range(3):
        print cm.getcost(pred, corr),
    print

print
print "Matrix for Iris, with default element 2 and several modified elements"
data = orange.ExampleTable("iris")
cm = orange.CostMatrix(data.domain.classVar, 2)
cm.setcost("Iris-setosa", "Iris-virginica", 1)
cm.setcost("Iris-versicolor", "Iris-virginica", 1)

print "classVar = %s, values = %s" % (cm.classVar.name, cm.classVar.values)
for pred in range(3):
    for corr in range(3):
        print cm.getcost(pred, corr),
    print

print
print "Manually initialized matrix"
cm = orange.CostMatrix(data.domain.classVar, [(0, 2, 1), (2, 0, 1), (2, 2, 0)])
for pred in range(3):
    for corr in range(3):
        print `cm.getcost(pred, corr)`,
    print

data = orange.ExampleTable("lenses")
print
print "Cost-sensitive attribute quality"
meas = orange.MeasureAttribute_cost()
meas.cost = ((0, 2, 1), (2, 0, 1), (2, 2, 0))
for attr in data.domain.attributes:
    print "%s: %5.3f" % (attr.name, meas(attr, data))
print

data = orange.ExampleTable("lenses")
print
print "Cost-sensitive attribute quality"
meas = orange.MeasureAttribute_cost()
meas.cost = data.domain.classVar
for attr in data.domain.attributes:
    print "%s: %5.3f" % (attr.name, meas(attr, data))
print