# Description: Shows how to round-off the cut-off points used for categorization.
# Category:    preprocessing
# Uses:        iris
# Classes:     EquiNDiscretization, EntropyDiscretization
# Referenced:  o_categorization.htm

import orange
iris = orange.ExampleTable("iris")

equiN = orange.EquiNDiscretization(numberOfIntervals=4)
entropy = orange.EntropyDiscretization()

pl = equiN("petal length", iris)
sl = equiN("sepal length", iris)
sl_ent = entropy("sepal length", iris)

points = pl.getValueFrom.transformer.points
points2 = map(lambda x:round(x), points)
pl.getValueFrom.transformer.points = points2

for attribute in [pl, sl, sl_ent]:
  print "Cut-off points for", attribute.name, \
    "are", attribute.getValueFrom.transformer.points
