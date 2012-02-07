# Description: Demonstrates the use of correspondence analysis
# Category:    correspondence, projection
# Classes:     CA
# Uses:        bridges.tab

import orange
import orngCA

data = orange.ExampleTable("bridges")
cm = orange.ContingencyAttrAttr("PURPOSE", "MATERIAL", data)
ca = orngCA.CA([list(col) for col in cm])

def report(coors, labels):
    for coor, label in zip(coors, labels):
        print "  %-10s (%.3f, %.3f)" % (label + ":", coor[0], coor[1])
        
print "PURPOSE"
report(ca.getPrincipalColProfilesCoordinates(), data.domain["PURPOSE"].values)
print 

print "MATERIAL"
report(ca.getPrincipalRowProfilesCoordinates(), data.domain["PURPOSE"].values)
print 