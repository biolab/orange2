import Orange
import distributions

table = Orange.data.Table("monks-1.tab")

print "Distributions of classes given the feature value"
dc = distributions.DomainContingency(table)
print "a: ", dc["a"]
print "b: ", dc["b"]
print "c: ", dc["e"]
print

print "Distributions of feature values given the class value"
dc = distributions.DomainContingency(table, classIsOuter = 1)
print "a: ", dc["a"]
print "b: ", dc["b"]
print "c: ", dc["e"]
print