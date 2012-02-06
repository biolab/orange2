# Description: 2D MDS scatterplot with Euclid distance, 100 optimization steps
# Category:    projection
# Uses:        iris
# Referenced:  Orange.projection.mds
# Classes:     Orange.projection.mds.MDS

import Orange

# Load some data
iris = Orange.data.Table("iris.tab")

# Construct a distance matrix using Euclidean distance
euclidean = Orange.distance.Euclidean(iris)
distance = Orange.core.SymMatrix(len(iris))
for i in range(len(iris)):
   for j in range(i + 1):
       distance[i, j] = euclidean(iris[i], iris[j])

# Run 100 steps of MDS optimization
mds = Orange.projection.mds.MDS(distance)
mds.run(100)

# Initialize matplotlib
import pylab
colors = ["red", "yellow", "blue"]

# Construct points (x, y, instanceClass)
points = []
for (i, d) in enumerate(iris):
   points.append((mds.points[i][0], mds.points[i][1], d.getclass()))

# Paint each class separately
for c in range(len(iris.domain.class_var.values)):
    sel = filter(lambda x: x[-1] == c, points)
    x = [s[0] for s in sel]
    y = [s[1] for s in sel]
    pylab.scatter(x, y, c=colors[c])

pylab.savefig('mds-scatterplot.py.png')
