# xtest: RANDOM

import orange

data1 = orange.ExampleTable("merge1")
data2 = orange.ExampleTable("merge2", use = data1.domain)

a1, a2 = data1.domain.attributes

metas = data1.domain.getmetas()
m1, m2 = data1.domain["m1"], data1.domain["m2"]
m1i, m2i = data1.domain.metaid(m1), data1.domain.metaid(m2)

a1, a3 = data2.domain.attributes
n1 = orange.FloatVariable("n1")
n2 = orange.FloatVariable("n2")

newdomain = orange.Domain([a1, a3, m1, n1])
newdomain.addmeta(m2i, m2)
newdomain.addmeta(orange.newmetaid(), a2)
newdomain.addmeta(orange.newmetaid(), n2)

merge = orange.Example(newdomain, [data1[0], data2[0]])
print "First example: ", data1[0]
print "Second example: ", data2[0]
print "Merge: ", merge