# Description: Tests various classes for attribute quality assessment
# Category:    attribute quality
# Classes:     MeasureAttribute, MeasureAttribute_info, MeasureAttribute_gainRatio, MeasureAttribute_gini, MeasureAttribute_relevance, MeasureAttribute_cost, MeasureAttribute_Relief
# Uses:        measure
# Referenced:  MeasureAttribute.htm

import Orange
import random
table = Orange.data.Table("measure")

table2 = Orange.data.Table(table)
nulls = [(0, 1, 24, 25), (24, 25), range(24, 34), (24, 25)]
for attr in range(len(nulls)):
    for e in nulls[attr]:
        table2[e][attr]="?"

names = [a.name for a in table.domain.attributes]
attrs = len(names)
print
print ("%30s"+"%15s"*attrs) % (("",) + tuple(names))
fstr = "%30s" + "%15.4f"*attrs

def printVariants(meas):
    print fstr % (("- no unknowns:",) + tuple([meas(i, table) for i in range(attrs)]))

    meas.unknownsTreatment = meas.IgnoreUnknowns
    print fstr % (("- ignore unknowns:",) + tuple([meas(i, table2) for i in range(attrs)]))

    meas.unknownsTreatment = meas.ReduceByUnknowns
    print fstr % (("- reduce unknowns:",) + tuple([meas(i, table2) for i in range(attrs)]))

    meas.unknownsTreatment = meas.UnknownsToCommon
    print fstr % (("- unknowns to common:",) + tuple([meas(i, table2) for i in range(attrs)]))

    meas.unknownsTreatment = meas.UnknownsAsValue
    print fstr % (("- unknowns as value:",) + tuple([meas(i, table2) for i in range(attrs)]))
    print

print "Information gain"
printVariants(Orange.feature.scoring.InfoGain())

print "Gain ratio"
printVariants(Orange.feature.scoring.GainRatio())

print "Gini index"
printVariants(Orange.feature.scoring.Gini())

print "Relief"
meas = Orange.feature.scoring.Relief()
print fstr % (("- no unknowns:",) + tuple([meas(i, table) for i in range(attrs)]))
print fstr % (("- with unknowns:",) + tuple([meas(i, table2) for i in range(attrs)]))
print

print "Cost matrix ((0, 5), (1, 0))"
meas = Orange.feature.scoring.Cost()
meas.cost = ((0, 5), (1, 0))
printVariants(meas)

print "Relevance"
printVariants(Orange.feature.scoring.Relevance())
