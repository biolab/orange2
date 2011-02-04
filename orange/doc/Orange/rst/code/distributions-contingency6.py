import Orange
import distributions

table = Orange.data.Table("iris.tab")
cont = distributions.ContingencyAttrClass(0, table)

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
