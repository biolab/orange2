# Description: Shows how to construct and use classifiers by lookup table to construct new features from the existing
# Category:    classification, lookup classifiers, constructive induction, feature construction
# Classes:     ClassifierByLookupTable, ClassifierByLookupTable1, ClassifierByLookupTable2, ClassifierByLookupTable3
# Uses:        monk1
# Referenced:  lookup.htm

import orange

data = orange.ExampleTable("monk1")

a, b, e = data.domain["a"], data.domain["b"], data.domain["e"]

ab = orange.EnumVariable("a==b", values = ["no", "yes"])
ab.getValueFrom = orange.ClassifierByLookupTable(ab, a, b,
                    ["yes", "no", "no",  "no", "yes", "no",  "no", "no", "yes"])

e1 = orange.EnumVariable("e==1", values = ["no", "yes"])
e1.getValueFrom = orange.ClassifierByLookupTable(e1, e,
                    ["yes", "no", "no", "no", "?"])

data2 = data.select([a, b, ab, e, e1, data.domain.classVar])

for i in range(5):
    print data2.randomexample()

for i in range(5):
    ex = data.randomexample()
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
