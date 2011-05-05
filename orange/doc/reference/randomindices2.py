# Description: Shows how to sample example by random divisions into two groups
# Category:    sampling
# Classes:     MakeRandomIndices, MakeRandomIndices2, RandomGenerator
# Uses:        lenses
# Referenced:  RandomIndices.htm

import orange

data = orange.ExampleTable("lenses")

indices2 = orange.MakeRandomIndices2(p0=6)

ind = indices2(data)
print ind
data0 = data.select(ind, 0)
data1 = data.select(ind, 1)
print len(data0), len(data1)

print "\nIndices without playing with random generator"
for i in range(5):
    print indices2(data)

print "\nIndices with random generator"
indices2.randomGenerator = orange.RandomGenerator(42)    
for i in range(5):
    print indices2(data)

print "\nIndices with randseed"
indices2.randomGenerator = None
indices2.randseed = 42
for i in range(5):
    print indices2(data)


print "\nIndices with p0 set as probability (not 'a number of')"
indices2.p0 = 0.25
print indices2(data)

print "\n... with stratification"
indices2.stratified = indices2.Stratified
ind = indices2(data)
print ind
data2 = data.select(ind)
od = orange.getClassDistribution(data)
sd = orange.getClassDistribution(data2)
od.normalize()
sd.normalize()
print od
print sd

print "\n... and without stratification"
indices2.stratified = indices2.NotStratified
print indices2(data)
ind = indices2(data)
print ind
data2 = data.select(ind)
od = orange.getClassDistribution(data)
sd = orange.getClassDistribution(data2)
od.normalize()
sd.normalize()
print od
print sd

print "\n... stratified 'if possible'"
indices2.stratified = indices2.StratifiedIfPossible
print indices2(data)

print "\n... stratified 'if possible', after removing the first example's class"
data[0].setclass("?")
print indices2(data)
