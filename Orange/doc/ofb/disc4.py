# Description: Shows how to find out which are the cut-off points introduced by Orange's automatic categorization rutines.
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

for attribute in [pl, sl, sl_ent]:
  print "Cut-off points for", attribute.name, \
    "are", attribute.getValueFrom.transformer.points
