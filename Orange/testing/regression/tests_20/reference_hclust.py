import orange, time

def repTime(msg):
    #print "%s: %s" % (time.asctime(), msg)
    pass

def callback(f, o):
    print int(round(100*f)),
    
repTime("Loading data")    
data = orange.ExampleTable("iris")

repTime("Computing distances")
matrix = orange.SymMatrix(len(data))
matrix.setattr("objects", data)
distance = orange.ExamplesDistanceConstructor_Euclidean(data)
for i1, ex1 in enumerate(data):
    for i2 in range(i1+1, len(data)):
        matrix[i1, i2] = distance(ex1, data[i2])

repTime("Hierarchical clustering (single linkage)")
clustering = orange.HierarchicalClustering()
clustering.linkage = clustering.Average
clustering.overwriteMatrix = 1
root = clustering(matrix)

repTime("Done.")

def prune(cluster, togo):
    if cluster.branches:
        if togo<0:
            cluster.branches = None
        else:
            for branch in cluster.branches:
                prune(branch, togo-cluster.height)

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

prune(root, 1.4)

for n, cluster in enumerate(listOfClusters(root)):
    print "\n\n*** Cluster %i ***\n" % n
    for ex in cluster:
        print ex

for cluster in listOfClusters(root):
    dist = orange.getClassDistribution(cluster)
    for e, d in enumerate(dist):
        print "%s: %3.0f   " % (data.domain.classVar.values[e], d),
    print

def listOfClustersT0(cluster, alist):
    if not cluster.branches:
        alist.append(orange.ExampleTable(cluster))
    else:
        for branch in cluster.branches:
            listOfClustersT0(branch, alist)

def listOfClustersT(root):
    l = []
    listOfClustersT0(root, l)
    return l

for t in listOfClustersT(root):
    print type(t), "of length", len(t)