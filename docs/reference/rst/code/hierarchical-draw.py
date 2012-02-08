import Orange

data = Orange.data.Table("iris")
sample = data.select(Orange.data.sample.SubsetIndices2(data, 20), 0)
root = Orange.clustering.hierarchical.clustering(sample)
labels = [str(d.get_class()) for d in sample]
Orange.clustering.hierarchical.dendrogram_draw(
    "hclust-dendrogram.png", root, labels=labels) 

my_colors = [(255,0,0), (0,255,0), (0,0,255)]
top_clusters = Orange.clustering.hierarchical.top_clusters(root, 3)
colors = dict([(cl, col) for cl, col in zip(top_clusters, my_colors)])
Orange.clustering.hierarchical.dendrogram_draw(
    "hclust-colored-dendrogram.png", root, data=sample, labels=labels, 
    cluster_colors=colors, color_palette=[(0,255,0), (0,0,0), (255,0,0)], 
    gamma=0.5, minv=2.0, maxv=7.0)
