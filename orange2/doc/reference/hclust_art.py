import orange

def printClustering(cluster):
    if cluster.branches:
        return "(%s%s)" % (printClustering(cluster.left), printClustering(cluster.right))
    else:
        return `cluster[0]`

m = [[],
     [ 3],
     [ 2,  4],
     [17,  5,  4],
     [ 2,  8,  3,  8],
     [ 7,  5, 10, 11, 2],
     [ 8,  4,  1,  5, 11, 13],
     [ 4,  7, 12,  8, 10,  1,  5],
     [13,  9, 14, 15,  7,  8,  4,  6],
     [12, 10, 11, 15,  2,  5,  7,  3,  1]]

matrix = orange.SymMatrix(m)
root = orange.HierarchicalClustering(matrix, linkage=orange.HierarchicalClustering.Average)

print printClustering(root)
print root.height

for el in root.left:
    print el,
print

root.mapping.setattr("objects", ["Ann", "Bob", "Curt", "Danny", "Eve", "Fred", "Greg", "Hue", "Ivy", "Jon"])
print printClustering(root)

matrix.setattr("objects", ["Ann", "Bob", "Curt", "Danny", "Eve", "Fred", "Greg", "Hue", "Ivy", "Jon"])
root = orange.HierarchicalClustering(matrix, linkage=orange.HierarchicalClustering.Average)

for el in root.left:
    print el,
print

print root.left[-1]

root.left.swap()
print printClustering(root)

root.permute([1, 0])
print printClustering(root)

def prune(cluster, togo):
    if cluster.branches:
        if togo<0:
            cluster.branches = None
        else:
            for branch in cluster.branches:
                prune(branch, togo-cluster.height)

def printClustering2(cluster):
    if cluster.branches:
        return "(%s%s)" % (printClustering2(cluster.left), printClustering2(cluster.right))
    else:
        return str(tuple(cluster))

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

prune(root, 9)
print printClustering2(root)

print listOfClusters(root)

del root.mapping.objects
print printClustering(root)
print root.mapping
print root.left.first
print root.left.last
print root.left.left.first
print root.left.left.last
print root.left.mapping[root.left.first:root.left.last]



root.mapping.setattr("objects", ["Ann", "Bob", "Curt", "Danny", "Eve", "Fred", "Greg", "Hue", "Ivy", "Jon"])
print listOfClusters(root)
