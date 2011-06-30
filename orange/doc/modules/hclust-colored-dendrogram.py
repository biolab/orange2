import orange
import orngClustering

data = orange.ExampleTable("iris")
sample = data.selectref(orange.MakeRandomIndices2(data, 20), 0)
root = orngClustering.hierarchicalClustering(sample)
reduced = orange.ExampleTable(orange.Domain(sample.domain[:2], False), sample)

my_colors = [(255,0,0), (0,255,0), (0,0,255)]
cls = orngClustering.hierarchicalClustering_topClusters(root, 3)
colors = dict([(cl, col) for cl, col in zip(cls, my_colors)])
print data.native(2)
orngClustering.dendrogram_draw("hclust-colored-dendrogram.png", root, data = reduced, labels=[str(d.getclass()) for d in sample],
    cluster_colors=colors, color_palette=[(0, 255, 0), (0, 0, 0), (255, 0, 0)], gamma=0.5, minv=2.0, maxv=7.0)

