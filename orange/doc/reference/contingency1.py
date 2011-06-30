# Description: Shows what the contingency matrix looks like and which are its common methods
# Category:    statistics
# Classes:     Contingency, ContingencyAttrClass
# Uses:        monk1
# Referenced:  contingency.htm

import orange
data = orange.ExampleTable("monk1")
cont = orange.ContingencyAttrClass("e", data)

print "Contingency items:"
for val, dist in cont.items():
    print val, dist
print

print "Contingency keys: ", cont.keys()
print "Contingency values: ", cont.values()
print "Contingency items: ", cont.items()
print

print "cont[0] =",cont[0]
print 'cont[\"1\"] =', cont["1"]
print 'cont[orange.Value(data.domain["e"], "1")] =', cont[orange.Value(data.domain["e"], "1")]
print

print "Iteration through contingency:"
for i in cont:
    print i
print

cont.normalize()
print "Contingency items after normalization:"
for val, dist in cont.items():
    print val, dist
print
