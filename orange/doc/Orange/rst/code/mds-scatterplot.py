# Description: 2D MDS scatterplot with Euclid distance, 100 optimization steps
# Category:    projection
# Uses:        iris
# Referenced:  Orange.projection.mds
# Classes:     Orange.projection.mds.MDS

import Orange

# Load some data
table = Orange.data.Table("iris.tab")

# Construct a distance matrix using Euclidean distance
euclidean = Orange.distances.EuclideanConstructor(table)
distance = Orange.core.SymMatrix(len(table))
for i in range(len(table)):
   for j in range(i+1):
       distance[i, j] = euclidean(table[i], table[j])

# Run 100 steps of MDS optimization
mds = Orange.projection.mds.MDS(distance)
mds.run(100)

# Initialize matplotlib
import pylab
colors = ["red", "yellow", "blue"]

# Construct points (x, y, instanceClass)
points = []
for (i, d) in enumerate(table):
   points.append((mds.points[i][0], mds.points[i][1], d.getclass()))

# Paint each class separately
for c in range(len(table.domain.classVar.values)):
    sel = filter(lambda x: x[-1] == c, points)
    x = [s[0] for s in sel]
    y = [s[1] for s in sel]
    pylab.scatter(x, y, c=colors[c])

pylab.savefig('mds-scatterplot.py.png')
