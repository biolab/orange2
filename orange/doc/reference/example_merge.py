import orange

data1 = orange.ExampleTable("merge1")
data2 = orange.ExampleTable("merge2", use = data1.domain)

a1, a2 = data1.domain.attributes
m1, m2 = data1.domain.getmetas().items()
a1, a3 = data2.domain.attributes
n1 = orange.FloatVariable("n1")
n2 = orange.FloatVariable("n2")

newdomain = orange.Domain([a1, a3, m1[1], n1])
newdomain.addmeta(m2[0], m2[1])
newdomain.addmeta(orange.newmetaid(), a2)
newdomain.addmeta(orange.newmetaid(), n2)

merge = orange.Example(newdomain, [data1[0], data2[0]])
print "First example: ", data1[0]
print "Second example: ", data2[0]
print "Merge: ", merge