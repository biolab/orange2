# Description: Demonstrates the use of ContingencyClassAttr
# Category:    statistics
# Classes:     Contingency, ContingencyClassAttr
# Uses:        monk1
# Referenced:  contingency.htm

import orange
data = orange.ExampleTable("iris")
cont = orange.ContingencyClassAttr("sepal length", data)

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

print "Probabilities for e=5.5"
for val in cont.classVar:
    print "  p(%s|%s) = %5.3f" % (5.5, val.native(), cont.p_attr(5.5, val))
print

cont = orange.ContingencyClassAttr(data.domain["sepal length"], data.domain.classVar)
for ex in data:
    cont.add_attrclass(ex["sepal length"], ex.getclass())

print "Distributions from a matrix computed manually:"
for val in cont.classVar:
    print "  p(.|%s) = %s" % (val.native(), cont.p_attr(val))
print
