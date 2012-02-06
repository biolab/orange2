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

print "Contingency keys: ", cont.keys()[:3]
print "Contingency values: ", cont.values()[:3]
print "Contingency items: ", cont.items()[:3]
print

try:
    midkey = (cont.keys()[0] + cont.keys()[1])/2.0
    print "cont[%5.3f] =" % (midkey, cont[midkey])
except Exception, v:
    print "Error: ", v
