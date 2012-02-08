# Description: Shows how to filter examples using various classes derived from Orange.data.filter.Filter
# Category:    filtering
# Classes:     Filter, Random, HasSpecial, HasClassValue, SameValue, Values, ValueFilterDiscrete, ValueFilterContinuous, ValueFilterString
# Uses:        lenses
# Referenced:  filters.htm

import Orange

data = Orange.data.Table("lenses")
instance = data[0]

randomfilter = Orange.data.filter.Random(prob = 0.7, randomGenerator = 24)
for i in range(10):
    print randomfilter(instance),
print

data70 = randomfilter(data)
print len(data), len(data70)

data2 = data[:5]
data2[0]["age"] = "?"
data2[1].setclass("?")
print "First five intances"
for ex in data2:
    print ex

print "\nInstances without unknown values"
f = Orange.data.filter.IsDefined(domain = data.domain)
for ex in f(data2):
    print ex

print "\nInstances without unknown values, ignoring 'age'"
f.check["age"] = 0
for ex in f(data2):
    print ex

print "\nInstances with unknown values (ignoring age)"
for ex in f(data2, negate=1):
    print ex

print "\nInstances with unknown values (HasSpecial)"
for ex in Orange.data.filter.HasSpecial(data2):
    print ex

print "\nInstances with no unknown values (HasSpecial)"
for ex in Orange.data.filter.HasSpecial(data2, negate=1):
    print ex

print "\nInstances with defined class (HasClassValue)"
for ex in Orange.data.filter.HasClassValue(data2):
    print ex

print "\nInstances with undefined class (HasClassValue)"
for ex in Orange.data.filter.HasClassValue(data2, negate=1):
    print ex


filteryoung = Orange.data.filter.SameValue()
age = data.domain["age"]
filteryoung.value = Orange.data.Value(age, "young")
filteryoung.position = data.domain.features.index(age)
print "\nYoung instances"
for ex in filteryoung(data):
    print ex


print "\nYoung or presbyopic with astigmatism"
fya = Orange.data.filter.Values()
age, astigm = data.domain["age"], data.domain["astigmatic"]
fya.domain = data.domain
fya.conditions.append(
    Orange.data.filter.ValueFilterDiscrete(
        position=data.domain.features.index(age),
        values=[Orange.data.Value(age, "young"),
                Orange.data.Value(age, "presbyopic")])
)
fya.conditions.append(
    Orange.data.filter.ValueFilterDiscrete(
        position = data.domain.features.index(astigm),
        values=[Orange.data.Value(astigm, "yes")]))
for ex in fya(data):
    print ex

print "\nYoung or presbyopic with astigmatism"
fya = Orange.data.filter.Values(domain=data.domain, conditions=
    [
    Orange.data.filter.ValueFilterDiscrete(
        position=data.domain.features.index(age),
        values=[Orange.data.Value(age, "young"),
                Orange.data.Value(age, "presbyopic")]),
    Orange.data.filter.ValueFilterDiscrete(
        position=data.domain.features.index(astigm),
        values=[Orange.data.Value(astigm, "yes")])
    ])
for ex in fya(data):
    print ex


print "\nYoung or presbyopic with astigmatism"
fya = Orange.data.filter.Values(domain=data.domain, conditions=
    [
    Orange.data.filter.ValueFilterDiscrete(
        position=data.domain.features.index(age),
        values=[Orange.data.Value(age, "young"),
                Orange.data.Value(age, "presbyopic")], acceptSpecial = 0),
    Orange.data.filter.ValueFilterDiscrete(
        position=data.domain.features.index(astigm),
        values=[Orange.data.Value(astigm, "yes")])
    ])
for ex in fya(data):
    print ex

print "\nYoung or presbyopic with astigmatism"
fya = Orange.data.filter.Values(domain=data.domain, conditions=
    [
    Orange.data.filter.ValueFilterDiscrete(
        position=data.domain.features.index(age),
        values=[Orange.data.Value(age, "young"),
                Orange.data.Value(age, "presbyopic")
                ], acceptSpecial = 1),
    Orange.data.filter.ValueFilterDiscrete(
        position=data.domain.features.index(astigm),
        values=[Orange.data.Value(astigm, "yes")])
    ])
for ex in fya(data):
    print ex

print "\nYoung or presbyopic or astigmatic"
fya = Orange.data.filter.Values(domain=data.domain, conditions=
    [
    Orange.data.filter.ValueFilterDiscrete(
        position=data.domain.features.index(age),
        values=[Orange.data.Value(age, "young"),
                Orange.data.Value(age, "presbyopic")
                ], acceptSpecial = 1),
    Orange.data.filter.ValueFilterDiscrete(
        position=data.domain.features.index(astigm),
        values=[Orange.data.Value(astigm, "yes")])
    ],
    conjunction = 0
)
for ex in fya(data):
    print ex
