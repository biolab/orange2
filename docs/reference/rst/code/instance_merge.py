# xtest: RANDOM

import Orange

data1 = Orange.data.Table("merge1")
data2 = Orange.data.Table("merge2")

a1, a2 = data1.domain.attributes

metas = data1.domain.getmetas()
m1, m2 = data1.domain["m1"], data1.domain["m2"]
m1i, m2i = data1.domain.metaid(m1), data1.domain.metaid(m2)

a1, a3 = data2.domain.attributes
n1 = Orange.feature.Continuous("n1")
n2 = Orange.feature.Continuous("n2")

new_domain = Orange.data.Domain([a1, a3, m1, n1])
new_domain.addmeta(m2i, m2)
new_domain.addmeta(Orange.data.new_meta_id(), a2)
new_domain.addmeta(Orange.data.new_meta_id(), n2)

merge = Orange.data.Instance(new_domain, [data1[0], data2[0]])
print "First example: ", data1[0]
print "Second example: ", data2[0]
print "Merge: ", merge
