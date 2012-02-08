import orange
import orngClustering

data = orange.ExampleTable("iris")
root = orngClustering.hierarchicalClustering(data)
n = 3
cls = orngClustering.hierarhicalClustering_topClustersMembership(root, n)
