# Description: Demonstrates the use of ContingencyClassAttr
# Category:    statistics
# Classes:     Contingency, ContingencyClassAttr
# Uses:        monk1
# Referenced:  contingency.htm

import orange
data = orange.ExampleTable("bridges")
cont = orange.ContingencyAttrAttr("SPAN", "MATERIAL", data)

print "Distributions:"
for val in cont.outerVariable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print

cont.normalize()
for val in cont.outerVariable:
    print "%s:" % val.native()
    for inval, p in cont[val].items():
        if p:
            print "   %s (%i%%)" % (inval, int(100*p+0.5))
    print

cont = orange.ContingencyAttrAttr(data.domain["SPAN"], data.domain["MATERIAL"])
for ex in data:
    cont.add(ex["SPAN"], ex["MATERIAL"])

print "Distributions from a matrix computed manually:"
for val in cont.outerVariable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print
