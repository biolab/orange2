# Description: Demonstrates the use of DomainContingency
# Category:    statistics
# Classes:     DomainContingency, ContingencyClassAttr, ContingencyAttrClass
# Uses:        monk1
# Referenced:  contingency.htm

import orange
data = orange.ExampleTable("monk1")

print "Distributions of classes given the attribute value"
dc = orange.DomainContingency(data)
print "a: ", dc["a"]
print "b: ", dc["b"]
print "c: ", dc["e"]
print

print "Distributions of attribute values given the class value"
dc = orange.DomainContingency(data, classIsOuter = 1)
print "a: ", dc["a"]
print "b: ", dc["b"]
print "c: ", dc["e"]
print
