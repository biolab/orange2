import Orange

iris = Orange.data.Table("iris")

matrix = Orange.misc.SymMatrix(len(iris))
matrix = Orange.distance.distance_matrix(iris, Orange.distance.Euclidean)

clustering = Orange.clustering.hierarchical.HierarchicalClustering()
clustering.linkage = Orange.clustering.hierarchical.AVERAGE
root = clustering(matrix)

root.mapping.objects = iris

topmost = sorted(Orange.clustering.hierarchical.top_clusters(root, 4), key=len)

for n, cluster in enumerate(topmost):
    print "\n\n Cluster %i \n" % n
    for instance in cluster:
        print instance

for cluster in topmost:
    dist = Orange.statistics.distribution.Distribution(iris.domain.class_var, \
        [ ex for ex in cluster ])
    for e, d in enumerate(dist):
        print "%s: %3.0f " % (iris.domain.class_var.values[e], d),
    print

