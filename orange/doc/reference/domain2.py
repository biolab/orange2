# Description: Shows how to use orange.Domain for example conversion. Also shows how to add meta-attributes to domain descriptors and use them.
# Category:    basic classes, meta-attributes
# Classes:     Domain
# Uses:        monk1
# Referenced:  Domain.htm

import orange

data = orange.ExampleTable("monk1")

d2 = orange.Domain(["a", "b", "e", "y"], data.domain)

example = data[55]
print example

example2 = d2(example)
print example2

example2 = orange.Example(d2, example)
print example2

data2 = orange.ExampleTable(d2, data)
print data2[55]

d2.addmeta(orange.newmetaid(), orange.FloatVariable("w"))
data2 = orange.ExampleTable(d2, data)
print data2[55]

misses = orange.FloatVariable("misses")
id = orange.newmetaid()
data.domain.addmeta(id, misses)
print data[55]

print data.domain.hasmeta(id)
print data.domain.hasmeta(id-1)

for example in data:
    example[misses] = 0

classifier = orange.BayesLearner(data)
for example in data:
    if example.getclass() != classifier(example):
        example[misses] += 1

for example in data:
    print example

data = orange.ExampleTable("monk1")
domain = data.domain
d2 = orange.Domain(["a", "b", "e", "y"], domain)
for attr in ["c", "d", "f"]:
    d2.addmeta(orange.newmetaid(), domain[attr])
d2.addmeta(orange.newmetaid(), orange.EnumVariable("X"))
data2 = orange.ExampleTable(d2, data)

print data[55]
print data2[55]

ido = -99
idr = -100
data.domain.addmeta(idr, orange.FloatVariable("required"), False)
data.domain.addmeta(ido, orange.FloatVariable("optional"), True)
print data.domain.isOptionalMeta(ido)
print data.domain.isOptionalMeta(idr)
print data.domain.getmetas()
print data.domain.getmetas(True)
print data.domain.getmetas(False)
