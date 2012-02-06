# Description: Shows how to use value transformers
# Category:    preprocessing
# Classes:     TransformValue, Continuous2Discrete, Discrete2Continuous, MapIntValue
# Uses:        
# Referenced:

import orange
print

def printExample(ex):
    for val in ex:
        print "%16s: %s" % (val.variable.name, val)

data = orange.ExampleTable("bridges")

for attr in data.domain:
    if attr.varType == orange.VarTypes.Continuous:
        print "%20s: continuous" % attr.name
    else:
        print "%20s: %s" % (attr.name, attr.values)

print
print "Original 15th example:"
printExample(data[15])

continuizer = orange.DomainContinuizer()

continuizer.multinomialTreatment = continuizer.LowestIsBase
domain0 = continuizer(data)
data0 = data.translate(domain0)
print
print "Lowest is base"
printExample(data0[15])

continuizer.multinomialTreatment = continuizer.FrequentIsBase
domain0 = continuizer(data)
data0 = data.translate(domain0)
print
print "Frequent is base"
printExample(data0[15])


continuizer.multinomialTreatment = continuizer.NValues
domain0 = continuizer(data)
data0 = data.translate(domain0)
print
print "NValues"
printExample(data0[15])


continuizer.multinomialTreatment = continuizer.Ignore
domain0 = continuizer(data)
data0 = data.translate(domain0)
print
print "Ignore"
printExample(data0[15])


continuizer.multinomialTreatment = continuizer.AsOrdinal
domain0 = continuizer(data)
data0 = data.translate(domain0)
print
print "As ordinal"
printExample(data0[15])


continuizer.multinomialTreatment = continuizer.AsNormalizedOrdinal
domain0 = continuizer(data)
data0 = data.translate(domain0)
print
print "As normalized ordinal"
printExample(data0[15])


