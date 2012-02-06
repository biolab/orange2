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

rules = orange.AssociationRulesInducer(data, support=0.4)

print "%i rules with support higher than or equal to %5.3f found.\n" % (len(rules), 0.4)

orngAssoc.sort(rules, ["support", "confidence"])

orngAssoc.printRules(rules[:5], ["support", "confidence"])
print

del rules[:3]
orngAssoc.printRules(rules[:5], ["support", "confidence"])
print
