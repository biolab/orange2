import orange
from orngClustering import *

# data = orange.ExampleTable("doc//datasets//brown-selected.tab")
data = orange.ExampleTable("iris")
# data = orange.ExampleTable("doc//datasets//zoo.tab")
# data = orange.ExampleTable("doc//datasets//titanic.tab")
# m = [[], [ 3], [ 2, 4], [17, 5, 4], [ 2, 8, 3, 8], [ 7, 5, 10, 11, 2], [ 8, 4, 1, 5, 11, 13], [ 4, 7, 12, 8, 10, 1, 5], [13, 9, 14, 15, 7, 8, 4, 6], [12, 10, 11, 15, 2, 5, 7, 3, 1]]
# matrix = orange.SymMatrix(m)
dist = orange.ExamplesDistanceConstructor_Euclidean(data)
matrix = orange.SymMatrix(len(data))
# matrix.setattr('items', data)
for i in range(len(data)):
    for j in range(i+1):
        matrix[i, j] = dist(data[i], data[j])
root = orange.HierarchicalClustering(matrix, linkage=orange.HierarchicalClustering.Average)
# root.mapping.objects = [str(ex.getclass()) for ex in data]
d = DendrogramPlot(root, data=data, labels=[str(ex.getclass()) for ex in data], width=500, height=2000)
d.set_matrix_color_schema([(0, 255, 0), (255, 0, 0)], 0.0, 1.0)
# d.setClusterColors({root.left:(0,255,0), root.right:(0,0,255)})
d.plot("graph.png")
print "Sum:", sum([matrix[root.mapping[i], root.mapping[i+1]] for i in range(len(root.mapping)-1)])
orderLeaves(root, matrix)
print "Sum:", sum([matrix[root.mapping[i], root.mapping[i+1]] for i in range(len(root.mapping)-1)])
d = DendrogramPlot(root, data=data, labels=[str(ex.getclass()) for ex in data], width=500, height=2000)
d.set_matrix_color_schema([(0, 255, 0), (255, 0, 0)], 0.0, 1.0)
d.plot("tmp.png")
