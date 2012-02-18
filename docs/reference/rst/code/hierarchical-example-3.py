import Orange

iris = Orange.data.Table("iris")

root = Orange.clustering.hierarchical.clustering(iris,
    distance_constructor=Orange.distance.Euclidean,
    linkage=Orange.clustering.hierarchical.AVERAGE)

root.mapping.objects = iris

for cluster in sorted(Orange.clustering.hierarchical.top_clusters(root, 4)):
    dist = Orange.statistics.distribution.Distribution(iris.domain.class_var, \
        [ ex for ex in cluster ])
    for e, d in enumerate(dist):
        print "%s: %3.0f " % (iris.domain.class_var.values[e], d),
    print

