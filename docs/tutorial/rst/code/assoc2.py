# Description: Association rule sorting and filtering
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

n = 5
print "%i most confident rules:" % (n)
orngAssoc.sort(rules, ["confidence", "support"])
orngAssoc.printRules(rules[0:n], ['confidence', 'support', 'lift'])

conf = 0.8; lift = 1.1
print "\nRules with confidence>%5.3f and lift>%5.3f" % (conf, lift)
rulesC = rules.filter(lambda x: x.confidence > conf and x.lift > lift)
orngAssoc.sort(rulesC, ['confidence'])
orngAssoc.printRules(rulesC, ['confidence', 'support', 'lift'])
