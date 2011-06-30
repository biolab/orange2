# Description: Attribute-based discretization. Shows how different attributes may be discretized with different categorization methods. Also shows how the resulting domain is put together using orange.select.
# Category:    preprocessing
# Uses:        iris
# Classes:     EquiNDiscretization, EntropyDiscretization
# Referenced:  o_categorization.htm

def printexamples(data, inxs, msg="First %i examples"):
  print msg % len(inxs)
  for i in inxs:
    print i, data[i]
  print

import orange
iris = orange.ExampleTable("iris")

equiN = orange.EquiNDiscretization(numberOfIntervals=4)
entropy = orange.EntropyDiscretization()

pl = equiN("petal length", iris)
sl = equiN("sepal length", iris)
sl_ent = entropy("sepal length", iris)

inxs = [0, 15, 35, 50, 98]
d_iris = iris.select(["sepal width", pl, "sepal length",sl, sl_ent, iris.domain.classVar])
printexamples(iris, inxs, "%i examples before discretization")
printexamples(d_iris, inxs, "%i examples before discretization")
