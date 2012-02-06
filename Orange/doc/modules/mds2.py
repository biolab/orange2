# Description: Uses MDS on iris data set and plots the scatterplot to illustrate the effect and observe the degree of separation between groups of different classes
# Category:    association
# Classes:     orngMDS.MDS
# Uses:        iris.tab
# Referenced:  orngMDS.htm

import orange
import orngMDS

data=orange.ExampleTable("../datasets/iris.tab")
euclidean = orange.ExamplesDistanceConstructor_Euclidean(data)
distance = orange.SymMatrix(len(data))
for i in range(len(data)-1):
   for j in range(i+1, len(data)):
       distance[i, j] = euclidean(data[i], data[j])

mds=orngMDS.MDS(distance)
mds.run(100)

try:
   from pylab import *
   colors = ["red", "yellow", "blue"]

   points = []
   for (i,d) in enumerate(data):
      points.append((mds.points[i][0], mds.points[i][1], d.getclass()))
   for c in range(len(data.domain.classVar.values)):
       sel = filter(lambda x: x[-1]==c, points)
       x = [s[0] for s in sel]
       y = [s[1] for s in sel]
       scatter(x, y, c=colors[c])
   savefig('mds-iris.png', dpi=72)
   show()
except ImportError:
    print "Can't import pylab (matplotlib)"
