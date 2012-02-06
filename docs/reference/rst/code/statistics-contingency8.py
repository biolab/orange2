import Orange

monks = Orange.data.Table("monks-1.tab")

print "Distributions of classes given the feature value"
dc = Orange.statistics.contingency.Domain(monks)
print "a: ", dc["a"]
print "b: ", dc["b"]
print "c: ", dc["e"]
print

print "Distributions of feature values given the class value"
dc = Orange.statistics.contingency.Domain(monks, classIsOuter = 1)
print "a: ", dc["a"]
print "b: ", dc["b"]
print "c: ", dc["e"]
print