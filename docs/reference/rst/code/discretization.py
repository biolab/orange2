# Description: Shows how usage of different classes for discretization, including manual discretization
# Category:    discretization, categorization, preprocessing
# Classes:     EntropyDiscretization, EquiDistDiscretization, BiModalDiscretization, Discretization, IntervalDiscretizer, Discretizer, BiModalDiscretizer
# Uses:        iris
# Referenced:  discretization.htm

import Orange
data = Orange.data.Table("iris")

print "\nEntropy discretization, first 10 examples"
sep_w = Orange.feature.discretization.Entropy("sepal width", data)

data2 = data.select([data.domain["sepal width"], sep_w, data.domain.class_var])
for ex in data2[:10]:
    print ex

print "\nDiscretized attribute:", sep_w
print "Continuous attribute:", sep_w.get_value_from.whichVar #FIXME not which_var
print "Cut-off points:", sep_w.get_value_from.transformer.points

print "\nManual construction of Interval discretizer - single attribute"
idisc = Orange.feature.discretization.Interval(points = [3.0, 5.0])
sep_l = idisc.construct_variable(data.domain["sepal length"])
data2 = data.select([data.domain["sepal length"], sep_l, data.domain.classVar])
for ex in data2[:10]:
    print ex


print "\nManual construction of Interval discretizer - all attributes"
idisc = Orange.feature.discretization.Interval(points = [3.0, 5.0])
newattrs = [idisc.construct_variable(attr) for attr in data.domain.attributes]
data2 = data.select(newattrs + [data.domain.class_var])
for ex in data2[:10]:
    print ex


print "\n\nDiscretization with equal width intervals"
disc = Orange.feature.discretization.EqualWidth(numberOfIntervals = 6)
newattrs = [disc(attr, data) for attr in data.domain.attributes]
data2 = data.select(newattrs + [data.domain.classVar])

for attr in newattrs:
    print "%s: %s" % (attr.name, attr.values)
print

for attr in newattrs:
    print "%15s: first interval at %5.3f, step %5.3f" % (attr.name, attr.get_value_from.transformer.first_cut, attr.get_value_from.transformer.step)
    print " "*17 + "cutoffs at " + ", ".join(["%5.3f" % x for x in attr.get_value_from.transformer.points])
print



print "\n\nQuartile (equal frequency) discretization"
disc = Orange.feature.discretization.EqualFreq(numberOfIntervals = 6)
newattrs = [disc(attr, data) for attr in data.domain.attributes]
data2 = data.select(newattrs + [data.domain.classVar])

for attr in newattrs:
    print "%s: %s" % (attr.name, attr.values)
print

for attr in newattrs:
    print " "*17 + "cutoffs at " + ", ".join(["%5.3f" % x for x in attr.get_value_from.transformer.points])
print



print "\nManual construction of EqualWidth - all attributes"
edisc = Orange.feature.discretization.EqualWidth(first_cut = 2.0, step = 1.0, number_of_intervals = 5)
newattrs = [edisc.constructVariable(attr) for attr in data.domain.attributes]
data2 = data.select(newattrs + [data.domain.classVar])
for ex in data2[:10]:
    print ex


print "\nFayyad-Irani entropy-based discretization"
entro = Orange.feature.discretization.Entropy()
for attr in data.domain.attributes:
    disc = entro(attr, data)
    print "%s: %s" % (attr.name, disc.get_value_from.transformer.points)
print


newclass = Orange.data.variable.Discrete("is versicolor", values = ["no", "yes"])
newclass.get_value_from = lambda ex, w: ex["iris"]=="Iris-versicolor"
newdomain = Orange.data.Domain(data.domain.attributes, newclass)
data_v = Orange.data.Table(newdomain, data)

print "\nBi-modal discretization on a binary problem"
bimod = Orange.feature.discretization.BiModal(split_in_two = 0)
for attr in data_v.domain.attributes:
    disc = bimod(attr, data_v)
    print "%s: %s" % (attr.name, disc.get_value_from.transformer.points)
print

print "\nBi-modal discretization on a binary problem"
bimod = Orange.feature.discretization.BiModal()
for attr in data_v.domain.attributes:
    disc = bimod(attr, data_v)
    print "%s: (%5.3f, %5.3f]" % (attr.name, disc.get_value_from.transformer.low, disc.get_value_from.transformer.high)
print


print "\nEntropy-based discretization on a binary problem"
for attr in data_v.domain.attributes:
    disc = entro(attr, data_v)
    print "%s: %s" % (attr.name, disc.getValueFrom.transformer.points)
