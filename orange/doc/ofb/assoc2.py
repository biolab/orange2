# Author:      B Zupan
# Version:     1.0
# Description: Association rule sorting and filtering
# Category:    description
# Uses:        imports-85

import orange, orngAssoc

data = orange.ExampleTable("imports-85")
data = orange.Preprocessor_discretize(data, \
  method=orange.EquiNDiscretization(numberOfIntervals=3))
data = data.select(range(10))

minSupport = 0.4
rules = orngAssoc.build(data, minSupport)

n = 5
print "%i most confident rules:" % (n)
rules.sortByConfidence()
rules[0:n].printMeasures(['confidence','support','lift'])

supp = 0.8; lift = 1.1
print "\nRules with support>%5.3f and lift>%5.3f" % (supp, lift)
rulesC=rules.filter(lambda x,supp=supp,lift=lift: x.support>supp and x.lift>lift)
rulesC.sortByFields(['confidence'])
rulesC.printMeasures(['confidence','support','lift'])
