# Description: Creates a list of association rules, selects five rules and prints them out
# Category:    description
# Uses:        imports-85
# Classes:     orngAssoc.build, Preprocessor_discretize, EquiNDiscretization
# Referenced:  assoc.htm

import orange, orngAssoc

data = orange.ExampleTable("imports-85")
data = orange.Preprocessor_discretize(data, \
  method=orange.EquiNDiscretization(numberOfIntervals=3))
data = data.select(range(10))

minSupport = 0.4
rules = orngAssoc.build(data, minSupport)

print "%i rules with support higher than or equal to %5.3f found.\n" % (len(rules), minSupport)

subset = rules[0:5]
subset.printMeasures(['support','confidence'])

print
del subset[0:2]
subset.printMeasures(['support','confidence'])
