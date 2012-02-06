# Description: 3D MDS with Euclid distance and Torgerson initialization, without iteration
# Category:    projection
# Uses:        iris
# Referenced:  Orange.projection.mds
# Classes:     Orange.projection.mds.MDS

import Orange

# Load some data
iris = Orange.data.Table("iris.tab")

# Construct a distance matrix using Euclidean distance
dist = Orange.distance.instances.EuclideanConstructor(iris)
matrix = Orange.core.SymMatrix(len(iris))
matrix.setattr('items', iris)
for i in range(len(iris)):
    for j in range(i+1):
        matrix[i, j] = dist(iris[i], iris[j])

# Run the MDS
mds = Orange.projection.mds.MDS(matrix, dim=3)
mds.Torgerson()

# Print a few points
print mds.points[:3]
