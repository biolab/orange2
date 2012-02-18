# Description: Shows the limitations of contingencies with continuous outer attributes
# Category:    statistics
# Classes:     Contingency
# Uses:        iris
# Referenced:  contingency.htm

import orange
data = orange.ExampleTable("iris")
cont = orange.ContingencyAttrClass(0, data)

print "Contingency items:"
for val, dist in cont.items()[:5]:
    print val, dist
print

print "Contingency keys: ", ["%.8f" % key for key in cont.keys()[:3]]
print "Contingency values: ", cont.values()[:3]
print "Contingency items: ", ["%.8f, %s" % (key, val) for key, val in cont.items()[:3]]
print
