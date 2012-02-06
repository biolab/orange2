import Orange

bridges = Orange.data.Table("bridges.tab")
cont = Orange.statistics.contingency.VarVar("SPAN", "MATERIAL", bridges)

print "Distributions:"
for val in cont.outer_variable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print

cont.normalize()
for val in cont.outer_variable:
    print "%s:" % val.native()
    for inval, p in cont[val].items():
        if p:
            print "   %s (%i%%)" % (inval, int(100*p+0.5))
    print

cont = Orange.statistics.contingency.VarVar(bridges.domain["SPAN"], bridges.domain["MATERIAL"])
for ins in bridges:
    cont.add(ins["SPAN"], ins["MATERIAL"])

print "Distributions from a matrix computed manually:"
for val in cont.outer_variable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print