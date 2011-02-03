import Orange
import random

table = Orange.data.Table("iris.tab")

measure = Orange.ensemble.forest.MeasureAttribute_randomForests(trees=100)

#call by attribute index
imp0 = measure(0, table) 
#call by orange.Variable
imp1 = measure(table.domain.attributes[1], table)
print "first: %0.2f, second: %0.2f\n" % (imp0, imp1)

print "different random seed"
measure = Orange.ensemble.forest.MeasureAttribute_randomForests(trees=100, 
        rand=random.Random(10))

imp0 = measure(0, table)
imp1 = measure(table.domain.attributes[1], table)
print "first: %0.2f, second: %0.2f\n" % (imp0, imp1)

print "All importances:"
imps = measure.importances(table)
for i,imp in enumerate(imps):
  print "%15s: %6.2f" % (table.domain.attributes[i].name, imp)