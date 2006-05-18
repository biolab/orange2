# Description: Shows how usage of different classes for discretization, including manual discretization
# Category:    discretization, categorization, preprocessing
# Classes:     EntropyDiscretization, EquiDistDiscretization, BiModalDiscretization, Discretization, IntervalDiscretizer, Discretizer, BiModalDiscretizer
# Uses:        iris
# Referenced:  discretization.htm

import orange

data = orange.ExampleTable("iris")

print "\nEntropy discretization, first 10 examples"
sep_w = orange.EntropyDiscretization("sepal width", data)

data2 = data.select([data.domain["sepal width"], sep_w, data.domain.classVar])
for ex in data2[:10]:
    print ex

print "\nDiscretized attribute:", sep_w
print "Continuous attribute:", sep_w.getValueFrom.whichVar
print "Cut-off points:", sep_w.getValueFrom.transformer.points


print "\nManual construction of IntervalDiscretizer - single attribute"
idisc = orange.IntervalDiscretizer(points = [3.0, 5.0])
sep_l = idisc.constructVariable(data.domain["sepal length"])
data2 = data.select([data.domain["sepal length"], sep_l, data.domain.classVar])
for ex in data2[:10]:
    print ex


print "\nManual construction of IntervalDiscretizer - all attributes"
idisc = orange.IntervalDiscretizer(points = [3.0, 5.0])
newattrs = [idisc.constructVariable(attr) for attr in data.domain.attributes]
data2 = data.select(newattrs + [data.domain.classVar])
for ex in data2[:10]:
    print ex


print "\n\nEqual interval size discretization"
disc = orange.EquiDistDiscretization(numberOfIntervals = 6)
newattrs = [disc(attr, data) for attr in data.domain.attributes]
data2 = data.select(newattrs + [data.domain.classVar])

for attr in newattrs:
    print "%s: %s" % (attr.name, attr.values)
print

for attr in newattrs:
    print "%15s: first interval at %5.3f, step %5.3f" % (attr.name, attr.getValueFrom.transformer.firstCut, attr.getValueFrom.transformer.step)
    print " "*17 + "cutoffs at " + ", ".join(["%5.3f" % x for x in attr.getValueFrom.transformer.points])
print



print "\n\nQuartile discretization"
disc = orange.EquiNDiscretization(numberOfIntervals = 6)
newattrs = [disc(attr, data) for attr in data.domain.attributes]
data2 = data.select(newattrs + [data.domain.classVar])

for attr in newattrs:
    print "%s: %s" % (attr.name, attr.values)
print

for attr in newattrs:
    print " "*17 + "cutoffs at " + ", ".join(["%5.3f" % x for x in attr.getValueFrom.transformer.points])
print



print "\nManual construction of EquiDistDiscretizer - all attributes"
edisc = orange.EquiDistDiscretizer(firstCut = 2.0, step = 1.0, numberOfIntervals = 5)
newattrs = [edisc.constructVariable(attr) for attr in data.domain.attributes]
data2 = data.select(newattrs + [data.domain.classVar])
for ex in data2[:10]:
    print ex


print "\nFayyad-Irani discretization"
entro = orange.EntropyDiscretization()
for attr in data.domain.attributes:
    disc = entro(attr, data)
    print "%s: %s" % (attr.name, disc.getValueFrom.transformer.points)
print


newclass = orange.EnumVariable("is versicolor", values = ["no", "yes"])
newclass.getValueFrom = lambda ex, w: ex["iris"]=="Iris-versicolor"
newdomain = orange.Domain(data.domain.attributes, newclass)
data_v = orange.ExampleTable(newdomain, data)

print "\nBi-Modal discretization on binary problem"
bimod = orange.BiModalDiscretization(splitInTwo = 0)
for attr in data_v.domain.attributes:
    disc = bimod(attr, data_v)
    print "%s: %s" % (attr.name, disc.getValueFrom.transformer.points)
print

print "\nBi-Modal discretization on binary problem"
bimod = orange.BiModalDiscretization()
for attr in data_v.domain.attributes:
    disc = bimod(attr, data_v)
    print "%s: (%5.3f, %5.3f]" % (attr.name, disc.getValueFrom.transformer.low, disc.getValueFrom.transformer.high)
print


print "\nEntropy discretization on binary problem"
for attr in data_v.domain.attributes:
    disc = entro(attr, data_v)
    print "%s: %s" % (attr.name, disc.getValueFrom.transformer.points)
