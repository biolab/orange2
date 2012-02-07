import Orange
m = [[],
     [ 3],
     [ 2, 4],
     [17, 5, 4],
     [ 2, 8, 3, 8],
     [ 7, 5, 10, 11, 2],
     [ 8, 4, 1, 5, 11, 13],
     [ 4, 7, 12, 8, 10, 1, 5],
     [13, 9, 14, 15, 7, 8, 4, 6],
     [12, 10, 11, 15, 2, 5, 7, 3, 1]]
matrix = Orange.misc.SymMatrix(m)
root = Orange.clustering.hierarchical.HierarchicalClustering(matrix,
        linkage=Orange.clustering.hierarchical.HierarchicalClustering.Average)

def printClustering(cluster):
    if cluster.branches:
        return "(%s%s)" % (printClustering(cluster.left), printClustering(cluster.right))
    else:
        return str(cluster[0])

def printClustering2(cluster):
    if cluster.branches:
        return "(%s%s)" % (printClustering2(cluster.left), printClustering2(cluster.right))
    else:
        return str(tuple(cluster))

matrix.objects = ["Ann", "Bob", "Curt", "Danny", "Eve",
                  "Fred", "Greg", "Hue", "Ivy", "Jon"]

root.mapping.objects = ["Ann", "Bob", "Curt", "Danny", "Eve", "Fred", "Greg", "Hue", "Ivy", "Jon"]
    
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