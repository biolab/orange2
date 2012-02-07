import orange, orngEnsemble, random

data = orange.ExampleTable("iris.tab")

measure = orngEnsemble.MeasureAttribute_randomForests(trees=100)

#call by attribute index
imp0 = measure(0, data) 
#call by orange.Variable
imp1 = measure(data.domain.attributes[1], data)
print "first: %0.2f, second: %0.2f\n" % (imp0, imp1)

print "different random seed"
measure = orngEnsemble.MeasureAttribute_randomForests(trees=100, rand=random.Random(10))

imp0 = measure(0, data)
imp1 = measure(data.domain.attributes[1], data)
print "first: %0.2f, second: %0.2f\n" % (imp0, imp1)

print "All importances:"
imps = measure.importances(data)
for i,imp in enumerate(imps):
  print "%15s: %6.2f" % (data.domain.attributes[i].name, imp)
