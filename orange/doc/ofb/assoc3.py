# Author:      B Zupan
# Version:     1.0
# Description: Cloning of association rules, filtering
# Category:    description
# Uses:        imports-85

import orange, orngAssoc

data = orange.ExampleTable("imports-85")
data = orange.Preprocessor_discretize(data, \
  method=orange.EquiNDiscretization(numberOfIntervals=3))
data = data.select(range(10))

minSupport = 0.2
rules = orngAssoc.build(data, minSupport)
print "%i rules with support higher than or equal to %5.3f found.\n" % (len(rules), minSupport)

rules2 = rules.clone()
rules2.sortByConfidence()

n = 5
print "Best %i rules:" % n
subset = rules[:n]
subset.printMeasures(['support','confidence'])