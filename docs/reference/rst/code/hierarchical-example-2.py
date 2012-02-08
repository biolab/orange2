import Orange

iris = Orange.data.Table("iris")
matrix = Orange.misc.SymMatrix(len(iris))
distance = Orange.distance.Euclidean(iris)
for i1, instance1 in enumerate(iris):
    for i2 in range(i1 + 1, len(iris)):
        matrix[i1, i2] = distance(instance1, iris[i2])

clustering = Orange.clustering.hierarchical.HierarchicalClustering()
clustering.linkage = clustering.Average
clustering.overwrite_matrix = 1
root = clustering(matrix)
root.mapping.objects = iris

def prune(cluster, togo):
    if cluster.branches:
        if togo < 0:
            cluster.branches = None
        else:
            for branch in cluster.branches:
                prune(branch, togo - cluster.height)

def listOfClusters0(cluster, alist):
    if not cluster.branches:
        alist.append(list(cluster))
    else:
        for branch in cluster.branches:
            listOfClusters0(branch, alist)

def listOfClusters(root):
    l = []
    listOfClusters0(root, l)
    return l
tables = [Orange.data.Table(cluster) for cluster in listOfClusters(root)]

prune(root, 1.4)
for n, cluster in enumerate(listOfClusters(root)):
    print "\n\n Cluster %i \n" % n
    for instance in cluster:
        print instance

for cluster in listOfClusters(root):
    dist = Orange.statistics.distribution.Distribution(iris.domain.class_var, cluster)
    for e, d in enumerate(dist):
        print "%s: %3.0f " % (iris.domain.class_var.values[e], d),
    print
