import orange
import orngClustering

data = orange.ExampleTable("iris")
sample = data.selectref(orange.MakeRandomIndices2(data, 20), 0)
root = orngClustering.hierarchicalClustering(sample)
orngClustering.dendrogram_draw("hclust-dendrogram.png", root, data=sample, labels=[str(d.getclass()) for d in sample]) 

