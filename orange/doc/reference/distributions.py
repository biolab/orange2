# Description: Shows how to compute and print attribute distributions. Also shows how to approximate a continuous distribution by Gaussian distribution
# Category:    statistics, distributions
# Classes:     Distribution, DiscDistribution, ContDistribution, DomainDistributions, GaussianDistribution
# Uses:        adult_sample
# Referenced:  distributions.htm

import orange

data = orange.ExampleTable("../datasets/adult_sample")

disc = orange.Distribution("workclass", data)
print type(disc)
print disc

print "Distribution of 'workclass' (through values)"
workclass = data.domain["workclass"]
for i in range(len(workclass.values)):
    print "%20s: %5.3f" % (workclass.values[i], disc[i])
print

print "Distribution of 'workclass' (through items)"
for val, num in disc.items():
    print "%20s: %5.3f" % (val, num)
print

disc[0] = disc[1] = 1000
for i in range(20):
 	print disc.modus(),
print

disc[0] = disc[1] = 1000
for i in range(20):
 	disc[2]=i
 	print disc.modus(),
print

disc = orange.Distribution("workclass", data)
print "Private: ", disc["Private"]
print "Private: ", disc[0]
print "Private: ", disc[orange.Value(workclass, "Private")]

print "length of distribution:", len(disc)
print "no. of values:", len(workclass.values)

print orange.Distribution(1, data)
print orange.Distribution(data.domain["workclass"], data)

cont = orange.Distribution("education-num", data)
print type(cont)
print cont

dist = orange.DomainDistributions(data)

for d in dist:
    if d.variable.varType == orange.VarTypes.Discrete:
        print "%30s: %s" % (d.variable.name, d)
    else:
        print "%30s: avg. %5.3f" % (d.variable.name, d.average())

print "*** AGE ***"
dage = dist["age"]
print "Native representation:", dage.native()
print "Keys:", dage.keys()
print "Values:", dage.values()
print "Items: ", dage.items()
print "Average: %5.3f" % dage.average()
print "Var/Dev/Err: %5.3f/%5.3f/%5.3f" % (dage.var(), dage.dev(), dage.error())
print "Quartiles: %5.3f - %5.3f - %5.3f" % (dage.percentile(25), dage.percentile(50), dage.percentile(75))
print

for x in range(170, 190):
    print "dens(%4.1f)=%5.3f," % (x/10.0, dage.density(x/10.0)),
    

print "*** WORKCLASS ***"
dwcl = dist["workclass"]
print "Native representation:", dwcl.native()
print "Keys:", dwcl.keys()
print "Values:", dwcl.values()
print "Items: ", dwcl.items()
print


disc = orange.DiscDistribution([0.5, 0.3, 0.2])
for i in range(20):
    print disc.random(),
print

v = orange.EnumVariable(values = ["red", "green", "blue"])
disc.variable = v
for i in range(20):
    print disc.random(),
print

print
cont = orange.ContDistribution({0.1: 12, 0.3: 3, 0.7: 3})
print "Manually constructed continuous distibution: ", cont
print


cont = orange.ContDistribution(data.domain["age"])


gauss = orange.GaussianDistribution(10, 2)
print "*** Gauss(10, 2) ***"
print "Average: %5.3f" % gauss.average()
print "Var/Dev/Err: %5.3f/%5.3f/%5.3f" % (gauss.var(), gauss.dev(), gauss.error())
print

for i in range(20):
    print "%5.3f" % gauss.random(),
print

for i in range(60, 140, 5):
    print "dens(%4.1f)=%5.3f" % (i/10.0, gauss.density(i/10.0)),
print

#dage.normalize()
gage = orange.GaussianDistribution(dage)
for x in range(17, 80):
    print "%i\t%5.3f\t%5.3f" % (x, dage.density(x), gage.density(x))