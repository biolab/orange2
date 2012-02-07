# Description: Advanced MDS test: 1000 optimization iterations, stress calculation after every 10th
# Category:    projection
# Uses:        iris
# Referenced:  Orange.projection.mds
# Classes:     Orange.projection.mds.MDS

import Orange
import math

# Load some data
iris = Orange.data.Table("iris.tab")

# Construct a distance matrix using Euclidean distance
dist = Orange.core.ExamplesDistanceConstructor_Euclidean(iris)
matrix = Orange.misc.SymMatrix(len(iris))
for i in range(len(iris)):
   for j in range(i+1):
       matrix[i, j] = dist(iris[i], iris[j])

# Run the Torgerson approximation and calculate stress
mds = Orange.projection.mds.MDS(matrix)
mds.Torgerson()
mds.calc_stress(Orange.projection.mds.KruskalStress)

# Optimization loop; calculate the stress only after each 10 optimization steps:
for i in range(100):
    old_stress = mds.avg_stress
    for j in range(10):
        mds.SMACOFstep()

    mds.calc_stress(Orange.projection.mds.KruskalStress)
    if old_stress * 1e-3 > math.fabs(old_stress - mds.avg_stress):
        break

# Print the points out
for (p, e) in zip(mds.points, iris):
    print p, e