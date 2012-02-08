# Description: Shows how to use meta-attributes with orange.Example
# Category:    basic classes, meta-attributes
# Classes:     Example
# Uses:        lenses
# Referenced:  Example.htm

import orange, random


data = orange.ExampleTable("lenses")
random.seed(0)
id = -42
# Note that this is wrong. Id should be assigned by
# id = orange.newmetaid()
# We only do this so that the script gives the same output each time it's run

for example in data:
    example[id] = orange.Value(random.random())

print data[0]

print orange.getClassDistribution(data)
print orange.getClassDistribution(data, id)

w = orange.FloatVariable("w")
data.domain.addmeta(id, w)

print data[0]

print data[0][id]
print data[0][w]
print data[0]["w"]

data[0][id] = orange.Value(w, 2.0)
data[0][id] = "2.0"
data[0][id] = 2.0

