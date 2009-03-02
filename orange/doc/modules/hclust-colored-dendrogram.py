import orange
import orngClustering

data = orange.ExampleTable("iris")
sample = data.selectref(orange.MakeRandomIndices2(data, 20), 0)
root = orngClustering.hierarchicalClustering(sample)
reduced = orange.ExampleTable(orange.Domain(sample.domain[:2], False), sample)

my_colors = [(255,0,0), (0,255,0), (0,0,255)]
cls = orngClustering.hierarchicalClustering_topClusters(root, 3)
colors = dict([(cl, col) for cl, col in zip(cls, my_colors)])

dendrogram = orngClustering.DendrogramPlot(root, reduced, labels=[str(d.getclass()) for d in sample],
    clusterColors=colors)
dendrogram.setMatrixColorScheme((0, 255, 0), (255, 0, 0))
dendrogram.plot("hclust-colored-dendrogram.png")
