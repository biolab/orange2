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
        linkage=Orange.clustering.hierarchical.AVERAGE)

def print_clustering(cluster):
    if cluster.branches:
        return "(%s %s)" % (print_clustering(cluster.left), print_clustering(cluster.right))
    else:
        return str(cluster[0])

print print_clustering(root)

print root.height

root.mapping.objects = ["Ann", "Bob", "Curt", "Danny", "Eve", 
    "Fred", "Greg", "Hue", "Ivy", "Jon"]

print print_clustering(root)

for el in root.left:
    print el

print print_clustering(root)
root.left.swap()
print print_clustering(root)
root.permute([1, 0])
print print_clustering(root)

def prune(cluster, h):
    if cluster.branches:
        if cluster.height < h:
            cluster.branches = None
        else:
            for branch in cluster.branches:
                prune(branch, h)

def print_clustering2(cluster):
    if cluster.branches:
        return "(%s %s)" % (print_clustering2(cluster.left), print_clustering2(cluster.right))
    else:
        return str(tuple(cluster))

prune(root, 5)
print print_clustering2(root)

def list_of_clusters0(cluster, alist):
    if not cluster.branches:
        alist.append(list(cluster))
    else:
        for branch in cluster.branches:
            list_of_clusters0(branch, alist)

def list_of_clusters(root):
    l = []
    list_of_clusters0(root, l)
    return l

print list_of_clusters(root)

root.mapping.objects = None
print list_of_clusters(root)

print root.left.last
print root.right.first
