# Description: Demonstrates the use of ContingencyAttrClass
# Category:    statistics
# Classes:     Contingency, ContingencyAttrClass
# Uses:        monk1
# Referenced:  contingency.htm

import orange
data = orange.ExampleTable("monk1")
cont = orange.ContingencyAttrClass("e", data)

print "Inner variable: ", cont.innerVariable.name
print "Outer variable: ", cont.outerVariable.name
print
print "Class variable: ", cont.classVar.name
print "Attribute:      ", cont.variable.name
print

print "Distributions:"
for val in cont.variable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_class(val))
print

firstclass = orange.Value(cont.classVar, 1)
firstnative = firstclass.native()
print "Probabilities of class '%s'" % firstnative
for val in cont.variable:
    print "  p(%s|%s) = %5.3f" % (firstnative, val.native(), cont.p_class(val, firstclass))
print

cont = orange.ContingencyAttrClass(data.domain["e"], data.domain.classVar)
for ex in data:
    cont.add_attrclass(ex["e"], ex.getclass())

print "Distributions from a matrix computed manually:"
for val in cont.variable:
    print "  p(.|%s) = %s" % (val.native(), cont.p_class(val))
print
print