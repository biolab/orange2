# Description: Shows how to construct and use classifiers by lookup table to construct new features from the existing
# Category:    classification, lookup classifiers, constructive induction, feature construction
# Classes:     ClassifierByLookupTable, ClassifierByLookupTable1, ClassifierByLookupTable2, ClassifierByLookupTable3
# Uses:        monks-1
# Referenced:  lookup.htm

import Orange

monks = Orange.data.Table("monks-1")

a, b, e = monks.domain["a"], monks.domain["b"], monks.domain["e"]

ab = Orange.feature.Discrete("a==b", values = ["no", "yes"])
ab.get_value_from = Orange.classification.lookup.ClassifierByLookupTable(ab, a, b,
                    ["yes", "no", "no",  "no", "yes", "no",  "no", "no", "yes"])

e1 = Orange.feature.Discrete("e==1", values = ["no", "yes"])
e1.get_value_from = Orange.classification.lookup.ClassifierByLookupTable(e1, e,
                    ["yes", "no", "no", "no", "?"])

monks2 = monks.select([a, b, ab, e, e1, monks.domain.class_var])

for i in range(5):
    print monks2.random_example()

for i in range(5):
    ex = monks.random_example()
    print "%s: ab %i, e1 %i " % (ex, ab.get_value_from.get_index(ex),
                                 e1.get_value_from.get_index(ex))
    
# What follows is only for testing Orange...

ab_c = ab.get_value_from
print ab_c.variable1.name, ab_c.variable2.name, ab_c.class_var.name
print ab_c.no_of_values1, ab_c.no_of_values2
print [x.name for x in ab_c.variables]

e1_c = e1.get_value_from
print e1_c.variable1.name, e1_c.class_var.name
print [x.name for x in e1_c.variables]
