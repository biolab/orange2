# Description: Manual categorization of continuous attributes.
# Category:    preprocessing
# Uses:        iris
# Classes:     ClassifierFromVar, IntervalDiscretizer, getValueFrom
# Referenced:  o_categorization.htm

import orange

def printexamples(data, inxs, msg="First %i examples"):
  print msg % len(inxs)
  for i in inxs:
    print data[i]
  print

iris = orange.ExampleTable("iris")
pl = orange.EnumVariable("pl")

getValue = orange.ClassifierFromVar()
getValue.whichVar = iris.domain["petal length"]
getValue.classVar = pl
getValue.transformer = orange.IntervalDiscretizer()
getValue.transformer.points = [2.0, 4.0]

pl.getValueFrom = getValue
pl.values = ['low', 'medium', 'high']
d_iris = iris.select(["petal length", pl, iris.domain.classVar])
printexamples(d_iris, [0, 15, 35, 50, 98], "%i examples after discretization")
