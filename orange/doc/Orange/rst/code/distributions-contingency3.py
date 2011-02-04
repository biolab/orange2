import Orange
import distributions

table = Orange.data.Table("monks-1.tab")
cont = distributions.ContingencyAttrClass("e", table)

print "Inner variable: ", cont.innerVariable.name
print "Outer variable: ", cont.outerVariable.name
print
print "Class variable: ", cont.classVar.name
print "Feature:      ", cont.variable.name
print

print "Distributions:"
for val in cont.variable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_class(val))
print

firstclass = Orange.data.Value(cont.classVar, 1)
firstnative = firstclass.native()
print "Probabilities of class '%s'" % firstnative
for val in cont.variable:
    print "  p(%s|%s) = %5.3f" % (firstnative, val.native(), cont.p_class(val, firstclass))
print

cont = distributions.ContingencyAttrClass(table.domain["e"], table.domain.classVar)
for ins in table:
    cont.add_attrclass(ins["e"], ins.getclass())

print "Distributions from a matrix computed manually:"
for val in cont.variable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_class(val))
print
