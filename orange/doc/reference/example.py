# Description: Shows different ways for constructing orange.Example and conversion to native Python objects
# Category:    basic classes
# Classes:     Example
# Uses:        lenses
# Referenced:  Example.htm

import orange
data = orange.ExampleTable("lenses")
domain = data.domain

for attr in domain:
    print attr.name, attr.values

ex = orange.Example(domain)
print ex

ex = orange.Example(domain, ["young", "myope", "yes", "reduced", "soft"])
print ex

ex = orange.Example(domain, ["young", 0, 1, orange.Value(domain[3], \
                             "reduced"), "soft"])
print ex

reduced_dom = orange.Domain(["age", "lenses"], domain)
reduced_ex = orange.Example(reduced_dom, ex)
print reduced_ex

age = data.domain["age"]
example = data[0]
print example[0]
print example[age]
print example["age"]

print data[0]
d = data[0][0]
example[age] = (int(example[age])+1) % 3
print data[0]
if d == data[0][0]:
    raise "Error in Example: not a reference, but a copy"

print example.native()
print example.native(0)
print example.native(1)

e1 = orange.Example(data[0])
e2 = orange.Example(data[0])
print e1.compatible(e2)
e2.setclass((e1.getclass() + 1) % 3)
print e1.compatible(e2)
print e1.compatible(e2, False)
print e1.compatible(e2, True)
