# Description: Shows how to remove or select examples with missing values
# Category:    preprocessing, missing values
# Classes:     Preprocessor, Preprocessor_addMissing, Preprocessor_addMissingClasses, Preprocessor_dropMissing, Preprocessor_dropMissingClasses, Preprocessor_takeMissing, Preprocessor_takeMissingClasses
# Uses:        lenses
# Referenced:  preprocessing.htm

import orange
data = orange.ExampleTable("lenses")
age, prescr, astigm, tears, y = data.domain.variables

pp = orange.Preprocessor_addMissingClasses()
pp.proportion = 0.5
pp.specialType = orange.ValueTypes.DK
data2 = pp(data)

print "Removing 50% of class values:",
for ex in data2:
    print ex.getclass(),
print

data2 = orange.Preprocessor_dropMissingClasses(data2)
print "Removing examples with unknown class values:",
for ex in data2:
    print ex.getclass(),
print

print "\n\nRemoving 20% of values of 'age' and 50% of astigmatism:"
pp = orange.Preprocessor_addMissing()
pp.proportions = {age: 0.2, astigm: 0.5}
pp.specialType = orange.ValueTypes.DC
data2 = pp(data)
for ex in data2:
    print ex

print "\n\nRemoving examples with unknown values"
data3 = orange.Preprocessor_dropMissing(data2)
for ex in data3:
    print ex

print "\n\nSelecting examples with unknown values"
data3 = orange.Preprocessor_takeMissing(data2)
for ex in data3:
    print ex

    