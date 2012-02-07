# Description: Shows how to construct an Orange.data.Table out of nothing
# Category:    basic classes
# Classes:     ExampleTable, Domain
# Uses:        
# Referenced:  ExampleTable.htm

import Orange

cards = [3, 3, 2, 3, 4, 2]
values = ["1", "2", "3", "4"]

features = [Orange.feature.Discrete(name, values=values[:card])
              for name, card in zip("abcdef", cards)]
classattr = Orange.feature.Discrete("y", values=["0", "1"])
domain = Orange.data.Domain(features + [classattr])
data = Orange.data.Table(domain)

import random
random.seed(0)

for i in range(5):
    inst = [random.randint(0, c - 1) for c in cards]
    inst.append(inst[0] == inst[1] or inst[4] == 0)
    data.append(inst)

for inst in data:
    print inst

loe = [["3", "1", "1", "2", "1", "1", "1"],
       ["3", "1", "1", "2", "2", "1", "0"],
       ["3", "3", "1", "2", "2", "1", "1"]
      ]

d2 = Orange.data.Table(domain, loe)

d2[0] = ["1", "1", 1, "1", "1", "1", "1"]

import numpy
d = Orange.data.Domain([Orange.feature.Continuous('a%i' % x) for x in range(5)])
a = numpy.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
t = Orange.data.Table(a)
print len(t)
print t[0]
print t[1]
