# Description: Demonstrates the use of ContingencyClassAttr
# Category:    statistics
# Classes:     Contingency, ContingencyClassAttr
# Uses:        monk1
# Referenced:  contingency.htm

import orange
data = orange.ExampleTable("monk1")
cont = orange.ContingencyClassAttr("e", data)

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

firstvalue = orange.Value(cont.variable, 0)
firstnative = firstvalue.native()
print "Probabilities for e='%s'" % firstnative
for val in cont.classVar:
    print "  p(%s|%s) = %5.3f" % (firstnative, val.native(), cont.p_attr(firstvalue, val))
print

cont = orange.ContingencyClassAttr(data.domain["e"], data.domain.classVar)
for ex in data:
    cont.add_attrclass(ex["e"], ex.getclass())

print "Distributions from a matrix computed manually:"
for val in cont.classVar:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print
