import Orange
import distributions

table = Orange.data.Table("bridges.tab")
cont = distributions.ContingencyAttrAttr("SPAN", "MATERIAL", table)

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

cont = distributions.ContingencyAttrAttr(table.domain["SPAN"], table.domain["MATERIAL"])
for ins in table:
    cont.add(ins["SPAN"], ins["MATERIAL"])

print "Distributions from a matrix computed manually:"
for val in cont.outerVariable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print