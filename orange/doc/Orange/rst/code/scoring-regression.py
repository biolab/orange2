# Description: Shows how to measure the attribute quality in regression problems.
# Category:    feature scoring
# Uses:        measure-c
# Referenced:  Orange.feature.html#scoring
# Classes:     Orange.feature.scoring.MSE

import Orange
import random
data = Orange.data.Table("measure-c")

data2 = Orange.data.Table(data)
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
    print
    
print "MSE"
printVariants(Orange.feature.scoring.MSE())
