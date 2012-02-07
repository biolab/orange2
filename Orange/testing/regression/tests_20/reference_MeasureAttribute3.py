# Description: Shows how to measure the attribute quality in regression problems
# Category:    statistics
# Classes:     MeasureAttribute, MeasureAttribute_MSE
# Uses:        measure-c
# Referenced:  MeasureAttribute.htm

import orange, random
data = orange.ExampleTable("measure-c")

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
    print
    
print "MSE"
printVariants(orange.MeasureAttribute_MSE())

print "Relief"
meas = orange.MeasureAttribute_relief()
print fstr % (("- no unknowns:",) + tuple([meas(i, data) for i in range(attrs)]))
print fstr % (("- with unknowns:",) + tuple([meas(i, data2) for i in range(attrs)]))
print
