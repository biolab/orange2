# Description: 3D MDS with Euclid distance and Torgerson initialization, without iteration
# Category:    projection
# Uses:        iris
# Referenced:  Orange.projection.mds
# Classes:     Orange.projection.mds.MDS

import Orange

# Load some data
table = Orange.data.Table("iris.tab")

# Construct a distance matrix using Euclidean distance
dist = Orange.distances.EuclideanConstructor(table)
matrix = Orange.core.SymMatrix(len(table))
matrix.setattr('items', table)
for i in range(len(table)):
    for j in range(i+1):
        matrix[i, j] = dist(table[i], table[j])

# Run the MDS
mds = Orange.projection.mds.MDS(matrix, dim=3)
mds.Torgerson()

# Print a few points
print mds.points[:3]
