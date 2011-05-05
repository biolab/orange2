# Description: Tests various classes for attribute quality assessment
# Category:    attribute quality
# Classes:     MeasureAttribute, MeasureAttribute_info, MeasureAttribute_gainRatio, MeasureAttribute_gini, MeasureAttribute_relevance, MeasureAttribute_cost, MeasureAttribute_Relief
# Uses:        measure
# Referenced:  MeasureAttribute.htm

import orange, random
data = orange.ExampleTable("measure")

data2 = orange.ExampleTable(data)
nulls = [(0, 1, 24, 25), (24, 25), range(24, 34), (24, 25)]
for attr in range(len(nulls)):
    for e in nulls[attr]:
        data2[e][attr]="?"

names = [a.name for a in data.domain.attributes]
attrs = len(names)
print
print ("%30s"+"%15s"*attrs) % (("",) + tuple(names))
fstr = "%30s" + "%15.4f"*attrs

def printVariants(meas):
    print fstr % (("- no unknowns:",) + tuple([meas(i, data) for i in range(attrs)]))

    meas.unknownsTreatment = meas.IgnoreUnknowns
    print fstr % (("- ignore unknowns:",) + tuple([meas(i, data2) for i in range(attrs)]))

    meas.unknownsTreatment = meas.ReduceByUnknowns
    print fstr % (("- reduce unknowns:",) + tuple([meas(i, data2) for i in range(attrs)]))

    meas.unknownsTreatment = meas.UnknownsToCommon
    print fstr % (("- unknowns to common:",) + tuple([meas(i, data2) for i in range(attrs)]))

    meas.unknownsTreatment = meas.UnknownsAsValue
    print fstr % (("- unknowns as value:",) + tuple([meas(i, data2) for i in range(attrs)]))
    print
    
print "Information gain"
printVariants(orange.MeasureAttribute_info())

print "Gain ratio"
printVariants(orange.MeasureAttribute_gainRatio())

print "Gini index"
printVariants(orange.MeasureAttribute_gini())

print "Relief"
meas = orange.MeasureAttribute_relief()
print fstr % (("- no unknowns:",) + tuple([meas(i, data) for i in range(attrs)]))
print fstr % (("- with unknowns:",) + tuple([meas(i, data2) for i in range(attrs)]))
print

print "Cost matrix ((0, 5), (1, 0))"
meas = orange.MeasureAttribute_cost()
meas.cost = ((0, 5), (1, 0))
printVariants(meas)

print "Relevance"
printVariants(orange.MeasureAttribute_relevance())
