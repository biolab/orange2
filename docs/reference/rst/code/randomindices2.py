# Description: Shows how to sample by random divisions into two groups
# Category:    sampling
# Classes:     SubsetIndices2, RandomGenerator
# Uses:        lenses
# Referenced:  RandomIndices.htm

import Orange

lenses = Orange.data.Table("lenses")

indices2 = Orange.data.sample.SubsetIndices2(p0=6)

ind = indices2(lenses)
print ind
lenses0 = lenses.select(ind, 0)
lenses1 = lenses.select(ind, 1)
print len(lenses0), len(lenses1)

print "\nIndices without playing with random generator"
for i in range(5):
    print indices2(lenses)

print "\nIndices with random generator"
indices2.random_generator = Orange.misc.Random(42)    
for i in range(5):
    print indices2(lenses)

print "\nIndices with randseed"
indices2.random_generator = None
indices2.randseed = 42
for i in range(5):
    print indices2(lenses)


print "\nIndices with p0 set as probability (not 'a number of')"
indices2.p0 = 0.25
print indices2(lenses)

print "\n... with stratification"
indices2.stratified = indices2.Stratified
ind = indices2(lenses)
print ind
lenses2 = lenses.select(ind)
od = Orange.core.getClassDistribution(lenses)
sd = Orange.core.getClassDistribution(lenses2)
od.normalize()
sd.normalize()
print od
print sd

print "\n... and without stratification"
indices2.stratified = indices2.NotStratified
print indices2(lenses)
ind = indices2(lenses)
print ind
lenses2 = lenses.select(ind)
od = Orange.core.getClassDistribution(lenses)
sd = Orange.core.getClassDistribution(lenses2)
od.normalize()
sd.normalize()
print od
print sd

print "\n... stratified 'if possible'"
indices2.stratified = indices2.StratifiedIfPossible
print indices2(lenses)

print "\n... stratified 'if possible', after removing the first instance's class"
lenses[0].setclass("?")
print indices2(lenses)
