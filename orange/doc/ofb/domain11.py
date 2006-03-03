# Description: Adds two new numerical attributes to iris data set, that are, respectively, computed from two existing attributes
# Category:    preprocessing
# Uses:        iris
# Classes:     Domain, FloatVariable
# Referenced:  domain.htm

import orange
data = orange.ExampleTable('iris')

sa = orange.FloatVariable("sepal area")
sa.getValueFrom = lambda e, getWhat: e['sepal length'] * e['sepal width']

pa = orange.FloatVariable("petal area")
pa.getValueFrom = lambda e, getWhat: e['petal length'] * e['petal width']

newdomain = orange.Domain(data.domain.attributes+[sa, pa, data.domain.classVar])
newdata = data.select(newdomain)

print
for a in newdata.domain.attributes:
  print "%13s" % a.name,
print "%16s" % newdata.domain.classVar.name
for i in [10,50,100,130]:
  for a in newdata.domain.attributes:
    print "%8s%5.2f" % (" ", newdata[i][a]),
  print "%16s" % (newdata[i].getclass())
