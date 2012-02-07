# Description: Shows how to reweight the examples to modify the class distribution
# Category:    preprocessing, weighting
# Classes:     Preprocessor, Preprocessor_addClassWeight
# Uses:        lenses
# Referenced:  preprocessing.htm

import orange
data = orange.ExampleTable("lenses")
age, prescr, astigm, tears, y = data.domain.variables

pp = orange.Preprocessor_addClassWeight()
pp.classWeights = [2.0, 1.0, 1.0]
data2, weightID = pp(data)
# we add a meta attribute so that output is always the same
# (else, the meta id would depend upon the number of meta attributes
# constructed, which would trigger suspicions about randomness in testing scripts
data2.domain.addmeta(weightID, orange.FloatVariable("W"))

print "Assigning weight 2.0 to examples from the first class"
print "  - original class distribution: ", orange.Distribution(y, data2)
print "  - weighted class distribution: ", orange.Distribution(y, data2, weightID)

pp.classWeights = None
pp.equalize = 1
data2, weightID = pp(data)
data2.domain.addmeta(weightID, orange.FloatVariable("W"))

print "\nEqualizing class distribution"
print "  - original class distribution: ", orange.Distribution(y, data2)
print "  - weighted class distribution: ", orange.Distribution(y, data2, weightID)


pp.classWeights = [0.5, 0.25, 0.25]
pp.equalize = 1
data2, weightID = pp(data)
data2.domain.addmeta(weightID, orange.FloatVariable("W"))

print "\nEqualizing class distribution and weighting by [0.5, 0.25, 0.25]"
print "  - original class distribution: ", orange.Distribution(y, data2)
print "  - weighted class distribution: ", orange.Distribution(y, data2, weightID)
