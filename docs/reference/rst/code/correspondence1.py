# Description: Demonstrates the use of correspondence analysis
# Category:    correspondence, projection
# Classes:     CA
# Uses:        bridges.tab

import Orange
import Orange.projection.correspondence as corr
import Orange.statistics.contingency as cont

data = Orange.data.Table("bridges")
cm = cont.VarVar("PURPOSE", "MATERIAL", data)
ca = corr.CA(cm)

def report(coors, labels):
    for coor, label in zip(coors, labels):
        print "  %-10s (%.3f, %.3f)" % (label + ":", coor[0, 0], coor[0, 1])
        
print "PURPOSE"
report(ca.column_factors(), data.domain["PURPOSE"].values)
print 

print "MATERIAL"
report(ca.row_factors(), data.domain["PURPOSE"].values)
print 