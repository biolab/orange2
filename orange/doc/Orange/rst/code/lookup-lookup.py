# Description: Shows how to construct and use classifiers by lookup table to construct new features from the existing
# Category:    classification, lookup classifiers, constructive induction, feature construction
# Classes:     ClassifierByLookupTable, ClassifierByLookupTable1, ClassifierByLookupTable2, ClassifierByLookupTable3
# Uses:        monks-1
# Referenced:  lookup.htm

import Orange

table = Orange.data.Table("monks-1")

a, b, e = table.domain["a"], table.domain["b"], table.domain["e"]

ab = Orange.data.feature.Discrete("a==b", values = ["no", "yes"])
ab.getValueFrom = Orange.classification.lookup.ClassifierByLookupTable(ab, a, b,
                    ["yes", "no", "no",  "no", "yes", "no",  "no", "no", "yes"])

e1 = Orange.data.feature.Discrete("e==1", values = ["no", "yes"])
e1.getValueFrom = Orange.classification.lookup.ClassifierByLookupTable(e1, e,
                    ["yes", "no", "no", "no", "?"])

table2 = table.select([a, b, ab, e, e1, table.domain.classVar])

for i in range(5):
    print table2.randomexample()

for i in range(5):
    ex = table.randomexample()
    print "%s: ab %i, e1 %i " % (ex, ab.getValueFrom.getindex(ex),
                                 e1.getValueFrom.getindex(ex))
    
# What follows is only for testing Orange...

ab_c = ab.getValueFrom
print ab_c.variable1.name, ab_c.variable2.name, ab_c.classVar.name
print ab_c.noOfValues1, ab_c.noOfValues2
print [x.name for x in ab_c.variables]

e1_c = e1.getValueFrom
print e1_c.variable1.name, e1_c.classVar.name
print [x.name for x in e1_c.variables]
