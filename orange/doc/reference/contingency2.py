# Description: Shows how to work with base class Contingency
# Category:    statistics
# Classes:     Contingency
# Uses:        monk1
# Referenced:  contingency.htm

import orange
data = orange.ExampleTable("monk1")

cont = orange.Contingency(data.domain["e"], data.domain.classVar)
for ex in data:
    cont [ex["e"]] [ex.getclass()] += 1
    
print "Contingency items:"
for val, dist in cont.items():
    print val, dist
print "Outer distribution: ", cont.outerDistribution
print "Inner distribution: ", cont.outerDistribution
print

cont = orange.Contingency(data.domain["e"], data.domain.classVar)
for ex in data:
    cont.add(ex["e"], ex.getclass())
    
print "Contingency items (with add):"
for val, dist in cont.items():
    print val, dist
print "Outer distribution: ", cont.outerDistribution
print "Inner distribution: ", cont.outerDistribution
print
