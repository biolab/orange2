# Description: Shows how to filter examples using various classes derived from orange.Filter
# Category:    filtering, preprocessing
# Classes:     Filter, Filter_random, Filter_hasSpecial, Filter_hasClassValue, Filter_sameValue, Filter_values
# Uses:        lenses
# Referenced:  filters.htm

import orange

data = orange.ExampleTable("lenses")

example = data[0]

randomfilter = orange.Filter_random(prob = 0.7, randomGenerator = 24)
for i in range(10):
    print randomfilter(example),
print

data70 = randomfilter(data)
print len(data), len(data70)

data2 = data[:5]
data2[0]["age"] = "?"
data2[1].setclass("?")
print "First five examples"
for ex in data2:
    print ex

print "\nExamples without unknown values"
f = orange.Filter_isDefined(domain = data.domain)
for ex in f(data2):
    print ex

print "\nExamples without unknown values, ignoring 'age'"
f.check["age"] = 0
for ex in f(data2):
    print ex

print "\nExamples with unknown values (ignoring age)"
for ex in f(data2, negate=1):
    print ex


print "\nExamples with unknown values (Filter_hasSpecial)"
for ex in orange.Filter_hasSpecial(data2):
    print ex

print "\nExamples with no unknown values (Filter_hasSpecial)"
for ex in orange.Filter_hasSpecial(data2, negate=1):
    print ex

print "\nExamples with defined class"
for ex in orange.Filter_hasClassValue(data2):
    print ex

print "\nExamples with undefined class"
for ex in orange.Filter_hasClassValue(data2, negate=1):
    print ex


filteryoung = orange.Filter_sameValue()
age = data.domain["age"]
filteryoung.value = orange.Value(age, "young")
filteryoung.position = data.domain.attributes.index(age)
print "\nYoung examples"
for ex in filteryoung(data):
    print ex


print "\nYoung or presbyopic with astigmatism"
fya = orange.Filter_values()
age, astigm = data.domain["age"], data.domain["astigmatic"]
fya.domain = data.domain
fya.conditions.append(orange.ValueFilter_discrete(position = data.domain.attributes.index(age), values=[orange.Value(age, "young"), orange.Value(age, "presbyopic")]))
fya.conditions.append(orange.ValueFilter_discrete(position = data.domain.attributes.index(astigm), values=[orange.Value(astigm, "yes")]))
for ex in fya(data):
    print ex

print "\nYoung or presbyopic with astigmatism"
fya = orange.Filter_values(domain = data.domain,
                           conditions = [orange.ValueFilter_discrete(position = data.domain.attributes.index(age), values=[orange.Value(age, "young"), orange.Value(age, "presbyopic")]),
                                         orange.ValueFilter_discrete(position = data.domain.attributes.index(astigm), values=[orange.Value(astigm, "yes")])
                                        ]
                          )
for ex in fya(data):
    print ex


print "\nYoung or presbyopic with astigmatism"
fya = orange.Filter_values(domain = data.domain,
                           conditions = [orange.ValueFilter_discrete(position = data.domain.attributes.index(age), values=[orange.Value(age, "young"), orange.Value(age, "presbyopic")], acceptSpecial = 0),
                                         orange.ValueFilter_discrete(position = data.domain.attributes.index(astigm), values=[orange.Value(astigm, "yes")])
                                        ],
                          )
for ex in fya(data):
    print ex

print "\nYoung or presbyopic with astigmatism"
fya = orange.Filter_values(domain = data.domain,
                           conditions = [orange.ValueFilter_discrete(position = data.domain.attributes.index(age), values=[orange.Value(age, "young"), orange.Value(age, "presbyopic")], acceptSpecial = 1),
                                         orange.ValueFilter_discrete(position = data.domain.attributes.index(astigm), values=[orange.Value(astigm, "yes")])
                                        ],
                          )
for ex in fya(data):
    print ex

print "\nYoung or presbyopic or astigmatic"
fya = orange.Filter_values(domain = data.domain,
                           conditions = [orange.ValueFilter_discrete(position = data.domain.attributes.index(age), values=[orange.Value(age, "young"), orange.Value(age, "presbyopic")], acceptSpecial = 1),
                                         orange.ValueFilter_discrete(position = data.domain.attributes.index(astigm), values=[orange.Value(astigm, "yes")])
                                        ],
                           conjunction = 0
                          )
for ex in fya(data):
    print ex
