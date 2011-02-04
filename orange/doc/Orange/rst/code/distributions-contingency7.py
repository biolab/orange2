import Orange
import distributions

table = Orange.data.Table("iris")
cont = distributions.ContingencyClassAttr("sepal length", table)

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

print "Estimated for e=5.5"
for val in cont.classVar:
    print "  f(%s|%s) = %5.3f" % (5.5, val.native(), cont.p_attr(5.5, val))
print

cont = distributions.ContingencyClassAttr(table.domain["sepal length"], table.domain.classVar)
for ins in table:
    cont.add_attrclass(ins["sepal length"], ins.getclass())

print "Distributions from a matrix computed manually:"
for val in cont.classVar:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print
