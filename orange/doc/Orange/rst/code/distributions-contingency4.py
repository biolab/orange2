import Orange
import distributions

table = Orange.data.Table("monks-1.tab")
cont = distributions.ContingencyClassAttr("e", table)

print "Inner variable: ", cont.innerVariable.name
print "Outer variable: ", cont.outerVariable.name
print
print "Class variable: ", cont.classVar.name
print "Attribute:      ", cont.variable.name
print

print "Distributions:"
for val in cont.classVar:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print

firstvalue = Orange.data.Value(cont.variable, 0)
firstnative = firstvalue.native()
print "Probabilities for e='%s'" % firstnative
for val in cont.classVar:
    print "  p(%s|%s) = %5.3f" % (firstnative, val.native(), cont.p_attr(firstvalue, val))
print

cont = distributions.ContingencyClassAttr(table.domain["e"], table.domain.classVar)
for ins in table:
    cont.add_attrclass(ins["e"], ins.getclass())

print "Distributions from a matrix computed manually:"
for val in cont.classVar:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print
