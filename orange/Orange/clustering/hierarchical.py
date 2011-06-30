r"""
******************************************
Hierarchical clustering (``hierarchical``)
******************************************

.. index::
   single: clustering, hierarchical, dendrogram
.. index:: aglomerative clustering

The method for hierarchical clustering, encapsulated in class
:class:`HierarchicalClustering` works on a distance matrix stored as
:class:`SymMatrix`. The method works in approximately O(n2) time (with
the worst case O(n3)). For orientation, clustering ten thousand of 
elements should take roughly 15 seconds on a 2 GHz computer. 
The algorithm can either make a copy of the distances matrix and work on 
it, or work on the original distance matrix, destroying it in the process. 
The latter is useful for clustering larger number of objects. Since the 
distance matrix stores (n+1)(n+2)/2 floats (cca 2 MB for 1000 objects and 
200 MB for 10000, assuming the a float takes 4 bytes), by copying it we 
would quickly run out of physical memory. Using virtual memory is not 
an option since the matrix is accessed in a random manner.

The distance should contain no negative elements. This limitation is
due to implementation details of the algorithm (it is not absolutely 
necessary and can be lifted in future versions if often requested; it 
only helps the algorithm run a bit faster). The elements on the diagonal 
(representing the element's distance from itself) are ignored.

Distance matrix can have the attribute objects describing the objects we 
are clustering (this is available only in Python). This can be any sequence 
of the same length as the matrix - an ExampleTable, a list of examples, a 
list of attributes (if you're clustering attributes), or even a string of 
the correct length. This attribute is not used in clustering but is only 
passed to the clusters' attribute ``mapping`` (see below), which will hold a 
reference to it (if you modify the list, the changes will affect the 
clusters as well).

.. class:: HierarchicalClustering
    
    .. attribute:: linkage
        
        Specifies the linkage method, which can be either :
        
            1. ``HierarchicalClustering.Single`` (default), where distance
                between groups is defined as the distance between the closest
                pair of objects, one from each group,
            2. ``HierarchicalClustering.Average`` , where the distance between
                two clusters is defined as the average of distances between
                all pairs of objects, where each pair is made up of one object
                from each group, or
            3. ``HierarchicalClustering.Complete``, where the distance between
                groups is defined as the distance between the most distant
                pair of objects, one from each group. Complete linkage is
                also called farthest neighbor.
            4. ``HierarchicalClustering.Ward`` uses Ward's distance.
            
    .. attribute:: overwriteMatrix
        
        If true (default is false), the algorithm will work on the original
        distance matrix, destroying it in the process. The benefit is that it
        will need much less memory (not much more than what is needed to store
        the tree of clusters).
        
    .. attribute:: progressCallback
        
        A callback function (None by default). It can be any function or
        callable class in Python, which accepts a single float as an
        argument. The function only gets called if the number of objects
        being clustered is at least 1000. It will be called for 101 times,
        and the argument will give the proportion of the work been done.
        The time intervals between the function calls won't be equal (sorry
        about that...) since the clustering proceeds faster as the number
        of clusters decreases.
        
    .. method:: __call__(matrix)
          
        The ``HierarchicalClustering`` is called with a distance matrix as an
        argument. It returns an instance of HierarchicalCluster representing
        the root of the hierarchy (instance of :class:`HierarchicalCluster`).
        See examples section for details.
        
        :param matrix: A distance matrix to perform the clustering on.
        :type matrix: :class:`Orange.core.SymMatrix`


.. class:: HierarchicalCluster

    Represents a node in the clustering tree, as returned by
    :obj:`HierarchicalClustering``

    .. attribute:: branches
    
        A list of sub-clusters (:class:`HierarchicalCluster` instances). If this
        is a leaf node this attribute is `None`
        
    .. attribute:: left
    
        The left sub-cluster (defined only if there are only two branches).
        
        .. note:: Same as ``branches[0]``
        
    .. attribute:: right
    
        The right sub-cluster (defined only if there are only two branches).
        
        .. note:: Same as ``branches[1]``
        
    .. attribute:: height
    
        Height of the cluster (distance between the sub-clusters).
        
    .. attribute:: mapping
    
        A list of indices to the original distance matrix. It is the same
        for all clusters in the hierarchy - it simply represents the indices
        ordered according to the clustering.
        
    .. attribute:: first
    .. attribute:: last
    
        ``first`` and ``last`` are indices into the elements of ``mapping`` that
        belong to that cluster. (Seems weird, but is trivial - wait for the
        examples. On the other hand, you probably won't need to understand this
        anyway).

    .. method:: __len__()
    
        Asking for the length of the cluster gives the number of the objects
        belonging to it. This equals ``last - first``.
    
    .. method:: __getitem__(index)
    
        By indexing the cluster we address its elements; these are either 
        indices or objects (you'll understand this after seeing the examples).
        For instance cluster[2] gives the third element of the cluster, and
        list(cluster) will return the cluster elements as a list. The cluster
        elements are read-only. To actually modify them, you'll have to go
        through mapping, as described below. This is intentionally complicated
        to discourage a naive user from doing what he does not understand.
    
    .. method:: swap()
    
        Swaps the ``left`` and the ``right`` subcluster; obviously this will
        report an error when the cluster has more than two subclusters. This 
        function changes the mapping and first and last of all clusters below
        this one and thus needs O(len(cluster)) time.
        
    .. method:: permute(permutation)
    
        Permutes the subclusters. Permutation gives the order in which the
        subclusters will be arranged. As for swap, this function changes the
        mapping and first and last of all clusters below this one. 
    
    
Example 1 - Toy matrix
----------------------

Let us construct a simple distance matrix and run clustering on it.
::

    import Orange
    from Orange.clustering import hierarchical
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
    matrix = Orange.core.SymMatrix(m)
    root = hierarchical.HierarchicalClustering(matrix,
            linkage=hierarchical.HierarchicalClustering.Average)
    
Root is a root of the cluster hierarchy. We can print using a
simple recursive function.
::

    def printClustering(cluster):
        if cluster.branches:
            return "(%s%s)" % (printClustering(cluster.left), printClustering(cluster.right))
        else:
            return str(cluster[0])
            
The output is not exactly nice, but it will have to do. Our clustering,
printed by calling printClustering(root) looks like this 
::
    
    (((04)((57)(89)))((1(26))3))
    
The elements are separated into two groups, the first containing elements
0, 4, 5, 7, 8, 9, and the second 1, 2, 6, 3. The difference between them
equals ``root.height``, 9.0 in our case. The first cluster is further
divided onto 0 and 4 in one, and 5, 7, 8, 9 in the other subcluster...

It is easy to print out the cluster's objects. Here's what's in the
left subcluster of root.
::

    >>> for el in root.left:
        ... print el,
    0 4 5 7 8 9 
    
Everything that can be iterated over, can as well be cast into a list or
tuple. Let us demonstrate this by writing a better function for printing
out the clustering (which will also come handy for something else in a
while). The one above supposed that each leaf contains a single object.
This is not necessarily so; instead of printing out the first (and
supposedly the only) element of cluster, cluster[0], we shall print
it out as a tuple. 
::

    def printClustering2(cluster):
        if cluster.branches:
            return "(%s%s)" % (printClustering2(cluster.left), printClustering2(cluster.right))
        else:
            return str(tuple(cluster))
            
The distance matrix could have been given a list of objects. We could,
for instance, put
::
    
    matrix.objects = ["Ann", "Bob", "Curt", "Danny", "Eve",
                      "Fred", "Greg", "Hue", "Ivy", "Jon"]

above calling the HierarchicalClustering.

.. note:: This code will actually trigger a warning;
    to avoid it, use matrix.setattr("objects", ["Ann", "Bob"....
    Why this is needed is explained in the page on `Orange peculiarities`_.
    
If we've forgotten to store the objects into matrix prior to clustering,
nothing is lost. We can add it into clustering later, by
::

    root.mapping.objects = ["Ann", "Bob", "Curt", "Danny", "Eve", "Fred", "Greg", "Hue", "Ivy", "Jon"]
    
So, what do these "objects" do? Call printClustering(root) again and you'll
see. Or, let us print out the elements of the first left cluster, as we did
before. 
::

    >>> for el in root.left:
        ... print el,
    Ann Eve Fred Hue Ivy Jon
    
If objects are given, the cluster's elements, as got by indexing
(eg root.left[2]) or by iteration, as in the above case, won't be
indices but the elements we clustered. If we put an ExampleTable
into objects, root.left[-1] will be the last example of the first
left cluster.

Now for swapping and permutations. ::

    >>> printClustering(root)
    ((('Ann''Eve')(('Fred''Hue')('Ivy''Jon')))(('Bob'('Curt''Greg'))'Danny'))
    >>> root.left.swap()
    >>> printClustering(root)
    (((('Fred''Hue')('Ivy''Jon'))('Ann''Eve'))(('Bob'('Curt''Greg'))'Danny'))
    >>> root.permute([1, 0])
    >>> printClustering(root)
    ((('Bob'('Curt''Greg'))'Danny')((('Fred''Hue')('Ivy''Jon'))('Ann''Eve')))
    
Calling ``root.left.swap`` reversed the order of subclusters of ``root.left``
and ``root.permute([1, 0])`` (which is equivalent to ``root.swap`` - there
aren't many possible permutations of two elements) reverses the order
of ``root.left`` and ``root.right``.

Let us write function for cluster pruning. ::

    def prune(cluster, togo):
        if cluster.branches:
            if togo<0:
                cluster.branches = None
            else:
                for branch in cluster.branches:
                    prune(branch, togo-cluster.height)

We shall use ``printClustering2`` here, since we can have multiple elements
in a leaf of the clustering hierarchy. ::
    
    >>> prune(root, 9)
    >>> print printClustering2(root)
    ((('Bob', 'Curt', 'Greg')('Danny',))(('Fred', 'Hue', 'Ivy', 'Jon')('Ann', 'Eve')))
    
We've ended up with four clusters. Need a list of clusters?
Here's the function. ::
    
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
        
The function returns a list of lists, in our case
``[['Bob', 'Curt', 'Greg'], ['Danny'], ['Fred', 'Hue', 'Ivy', 'Jon'], ['Ann', 'Eve']]``    
If there were no ``objects`` the list would contains indices instead of names.

Example 2 - Clustering of examples
----------------------------------

The most common things to cluster are certainly examples. To show how to
this is done, we shall now load the Iris data set, initialize a distance
matrix with the distances measure by :class:`ExamplesDistance_Euclidean`
and cluster it with average linkage. Since we don't need the matrix,
we shall let the clustering overwrite it (not that it's needed for
such a small data set as Iris). ::

    import Orange
    from Orange.clustering import hierarchical
    from Orange import distances
    
    data = Orange.data.Table("iris")
    matrix = Orange.core.SymMatrix(len(data))
    matrix.setattr("objects", data)
    distance = distances.ExamplesDistanceConstructor_Euclidean(data)
    for i1, instance1 in enumerate(data):
        for i2 in range(i1+1, len(data)):
            matrix[i1, i2] = distance(instance1, data[i2])
            
    clustering = hierarchical.HierarchicalClustering()
    clustering.linkage = clustering.Average
    clustering.overwrite_matrix = 1
    root = clustering(matrix)

Note that we haven't forgotten to set the ``matrix.objects``. We did it
through ``matrix.setattr`` to avoid the warning. Let us now prune the
clustering using the function we've written above, and print out the
clusters. ::
    
    prune(root, 1.4)
    for n, cluster in enumerate(listOfClusters(root)):
        print "\n\n Cluster %i \n" % n
        for instance in cluster:
            print instance
            
Since the printout is pretty long, it might be more informative to just
print out the class distributions for each cluster. ::
    
    for cluster in listOfClusters(root):
        dist = Orange.core.get_class_distribution(cluster)
        for e, d in enumerate(dist):
            print "%s: %3.0f " % (data.domain.class_var.values[e], d),
        print
        
Here's what it shows. ::

    Iris-setosa:  49    Iris-versicolor:   0    Iris-virginica:   0
    Iris-setosa:   1    Iris-versicolor:   0    Iris-virginica:   0
    Iris-setosa:   0    Iris-versicolor:  50    Iris-virginica:  17
    Iris-setosa:   0    Iris-versicolor:   0    Iris-virginica:  33
    
Note something else: ``listOfClusters`` does not return a list of
:class:`Orange.data.Table`, but a list of lists of instances. Therefore,
in the above script, cluster is a list of examples, not an ``Table``, but
it gets converted to it automatically when the function is called.
Most Orange functions will do this for you automatically. You can, for
instance, call a learning algorithms, passing a cluster as an argument.
It won't mind. If you, however, want to have a list of table, you can
easily convert the list by ::

    tables = [Orange.data.Table(cluster) for cluster in listOfClusters(root)]
    
Finally, if you are dealing with examples, you may want to take the function
``listOfClusters`` and replace ``alist.append(list(cluster))`` by
``alist.append(Orange.data.Table(cluster))``. This function is less general,
it will fail if objects are not of type :class:`Orange.data.Instance`.
However, instead of list of lists, it will return a list of tables.

How the data in ``HierarchicalCluster`` is really stored?
---------------------------------------------------------

To demonstrate how the data in clusters is stored, we shall continue with
the clustering we got in the first example. ::
    
    >>> del root.mapping.objects
    >>> print printClustering(root)
    (((1(26))3)(((57)(89))(04)))
    >>> print root.mapping
    <1, 2, 6, 3, 5, 7, 8, 9, 0, 4>
    >>> print root.left.first
    0
    >>> print root.left.last
    4
    >>> print root.left.mapping[root.left.first:root.left.last]
    <1, 2, 6, 3>
    >>> print root.left.left.first
    0
    >>> print root.left.left.last
    3
    
We removed objects to just to see more clearly what is going on.
``mapping`` is an ordered list of indices to the rows/columns of distance
matrix (and, at the same time, indices into objects, if they exist). Each
cluster's fields ``first`` and ``last`` are indices into mapping, so the
clusters elements are actually
``cluster.mapping[cluster.first:cluster.last]``. ``cluster[i]`` therefore
returns ``cluster.mapping[cluster.first+i]`` or, if objects are specified,
``cluster.objects[cluster.mapping[cluster.first+i]]``. Space consumption
is minimal since all clusters share the same objects ``mapping`` and
``objects``.


Subclusters are ordered so that ``cluster.left.last`` always equals
``cluster.right.first`` or, in general, ``cluster.branches[i].last``
equals ``cluster.branches[i+1].first``.


Swapping and permutation do three things: change the order of elements in
``branches``, permute the corresponding regions in ``mapping`` and adjust
the ``first`` and ``last`` for all the clusters below. For the latter, when
subclusters of cluster are permuted, the entire subtree starting at
``cluster.branches[i]`` is moved by the same offset.


The hierarchy of objects that represent a clustering is open, everything is
accessible from Python. You can write your own clustering algorithms that
build this same structure, or you can use Orange's clustering and then do to
the structure anything you want. For instance prune it, as we have shown
earlier. However, it is easy to do things wrong: shuffle the mapping, for
instance, and forget to adjust the ``first`` and ``last`` pointers. Orange
does some checking for the internal consistency, but you are surely smarter
and can find a way to crash it. For instance, just create a cycle in the
structure, call ``swap`` for some cluster above the cycle and you're there.
But don't blame it on me then.


Utility Functions
=================

.. autofunction:: clustering
.. autofunction:: clustering_features
.. autofunction:: cluster_to_list
.. autofunction:: top_clusters
.. autofunction:: top_cluster_membership
.. autofunction:: order_leaves

.. autofunction:: postorder
.. autofunction:: preorder
.. autofunction:: dendrogram_layout
.. autofunction:: dendrogram_draw
.. autofunction:: clone
.. autofunction:: prune
.. autofunction:: pruned
.. autofunction:: cluster_depths
.. autofunction:: instance_distance_matrix
.. autofunction:: feature_distance_matrix
.. autofunction:: joining_cluster
.. autofunction:: cophenetic_distances
.. autofunction:: cophenetic_correlation

"""

import orange
import Orange
from Orange.core import HierarchicalClustering, \
                        HierarchicalCluster, \
                        HierarchicalClusterList

from Orange.misc import progressBarMilestones
                        
import sys

SINGLE = HierarchicalClustering.Single
AVERAGE = HierarchicalClustering.Average
COMPLETE = HierarchicalClustering.Complete
WARD = HierarchicalClustering.Ward

def clustering(data,
               distanceConstructor=orange.ExamplesDistanceConstructor_Euclidean,
               linkage=AVERAGE,
               order=False,
               progressCallback=None):
    """ Return a hierarchical clustering of the instances in a data set.
    
    :param data: Input data table for clustering.
    :type data: :class:`Orange.data.Table`
    :param distance_constructor: Instance distance constructor
    :type distance_constructor: :class:`Orange.distances.ExamplesDistanceConstructor`
    :param linkage: Linkage flag. Must be one of global module level flags:
    
        - SINGLE
        - AVERAGE
        - COMPLETE
        - WARD
        
    :type linkage: int
    :param order: If `True` run `order_leaves` on the resulting clustering.
    :type order: bool
    :param progress_callback: A function (taking one argument) to use for
        reporting the on the progress.
    :type progress_callback: function
    
    """
    distance = distanceConstructor(data)
    matrix = orange.SymMatrix(len(data))
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = distance(data[i], data[j])
    root = HierarchicalClustering(matrix, linkage=linkage, progressCallback=(lambda value, obj=None: progressCallback(value*100.0/(2 if order else 1))) if progressCallback else None)
    if order:
        order_leaves(root, matrix, progressCallback=(lambda value: progressCallback(50.0 + value/2)) if progressCallback else None)
    return root

def clustering_features(data, distance=None, linkage=orange.HierarchicalClustering.Average, order=False, progressCallback=None):
    """ Return hierarchical clustering of attributes in a data set.
    
    :param data: Input data table for clustering.
    :type data: :class:`Orange.data.Table`
    :param distance: Attribute distance constructor 
        .. note:: currently not used.
    :param linkage: Linkage flag. Must be one of global module level flags:
    
        - SINGLE
        - AVERAGE
        - COMPLETE
        - WARD
        
    :type linkage: int
    :param order: If `True` run `order_leaves` on the resulting clustering.
    :type order: bool
    :param progress_callback: A function (taking one argument) to use for
        reporting the on the progress.
    :type progress_callback: function
    
    """
    matrix = orange.SymMatrix(len(data.domain.attributes))
    for a1 in range(len(data.domain.attributes)):
        for a2 in range(a1):
            matrix[a1, a2] = (1.0 - orange.PearsonCorrelation(a1, a2, data, 0).r) / 2.0
    root = orange.HierarchicalClustering(matrix, linkage=linkage, progressCallback=(lambda value, obj=None: progressCallback(value*100.0/(2 if order else 1))) if progressCallback else None)
    if order:
        order_leaves(root, matrix, progressCallback=(lambda value: progressCallback(50.0 + value/2)) if progressCallback else None)
    return root

def cluster_to_list(node, prune=None):
    """ Return a list of clusters down from the node of hierarchical clustering.
    
    :param node: Cluster node.
    :type node: :class:`HierarchicalCluster`
    :param prune: If not `None` it must be a positive integer. Any cluster
        with less then `prune` items will be left out of the list.
    :type node: int or `NoneType`
    
    """
    if prune:
        if len(node) <= prune:
            return [] 
    if node.branches:
        return [node] + cluster_to_list(node.left, prune) + cluster_to_list(node.right, prune)
    return [node]

def top_clusters(root, k):
    """ Return k topmost clusters from hierarchical clustering.
    
    :param root: Root cluster.
    :type root: :class:`HierarchicalCluster`
    :param k: Number of top clusters.
    :type k: int
    
    """
    candidates = set([root])
    while len(candidates) < k:
        repl = max([(max(c.left.height, c.right.height), c) for c in candidates if c.branches])[1]
        candidates.discard(repl)
        candidates.add(repl.left)
        candidates.add(repl.right)
    return candidates

def top_cluster_membership(root, k):
    """ Return data instances' cluster membership (list of indices) to k topmost clusters.
    
    :param root: Root cluster.
    :type root: :class:`HierarchicalCluster`
    :param k: Number of top clusters.
    :type k: int
    
    """
    clist = top_clusters(root, k)
    cmap = [None] * len(root)
    for i, c in enumerate(clist):
        for e in c:
            cmap[e] = i
    return cmap

def order_leaves(tree, matrix, progressCallback=None):
    """Order the leaves in the clustering tree.
    
    (based on Ziv Bar-Joseph et al. (Fast optimal leaf ordering for hierarchical clustering))
    
    :param tree: Binary hierarchical clustering tree.
    :type tree: :class:`HierarchicalCluster`
    :param matrix: SymMatrix that was used to compute the clustering.
    :type matrix: :class:`Orange.core.SymMatrix`
    :param progress_callback: Function used to report on progress.
    :type progress_callback: function
    
    .. note:: The ordering is done inplace. 
    
    """
    
    objects = getattr(tree.mapping, "objects", None)
    tree.mapping.setattr("objects", range(len(tree)))
    M = {}
    ordering = {}
    visitedClusters = set()
    
#    def _optOrderingRecursive(tree):
#        if len(tree)==1:
#            for leaf in tree:
#                M[tree, leaf, leaf] = 0
#        else:
#            _optOrderingRecursive(tree.left)
#            _optOrderingRecursive(tree.right)
#            
#            Vl = set(tree.left)
#            Vr = set(tree.right)
#            Vlr = set(tree.left.right or tree.left)
#            Vll = set(tree.left.left or tree.left)
#            Vrr = set(tree.right.right or tree.right)
#            Vrl = set(tree.right.left or tree.right)
#            other = lambda e, V1, V2: V2 if e in V1 else V1
#            tree_left, tree_right = tree.left, tree.right
#            for u in Vl:
#                for w in Vr:
##                    if True: #Improved search
#                    C = min([matrix[m, k] for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)])
#                    orderedMs = sorted(other(u, Vll, Vlr), key=lambda m: M[tree_left, u, m])
#                    orderedKs = sorted(other(w, Vrl, Vrr), key=lambda k: M[tree_right, w, k])
#                    k0 = orderedKs[0]
#                    curMin = 1e30000 
#                    curM = curK = None
#                    for m in orderedMs:
#                        if M[tree_left, u, m] + M[tree_right, w, k0] + C >= curMin:
#                            break
#                        for k in  orderedKs:
#                            if M[tree_left, u, m] + M[tree_right, w, k] + C >= curMin:
#                                break
#                            testMin = M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
#                            if curMin > testMin:
#                                curMin = testMin
#                                curM = m
#                                curK = k
#                    M[tree, u, w] = M[tree, w, u] = curMin
#                    ordering[tree, u, w] = (tree_left, u, curM, tree_right, w, curK)
#                    ordering[tree, w, u] = (tree_right, w, curK, tree_left, u, curM)
##                    else:
##                    def MFunc((m, k)):
##                        return M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
##                    m, k = min([(m, k) for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)], key=MFunc)
##                    M[tree, u, w] = M[tree, w, u] = MFunc((m, k))
##                    ordering[tree, u, w] = (tree_left, u, m, tree_right, w, k)
##                    ordering[tree, w, u] = (tree_right, w, k, tree_left, u, m)
#
#            if progressCallback:
#                progressCallback(100.0 * len(visitedClusters) / len(tree.mapping))
#                visitedClusters.add(tree)
#    
#    with recursion_limit(sys.getrecursionlimit() + len(tree)):
#        _optOrderingRecursive(tree)
        
    def _optOrderingIterative(tree):
        if len(tree)==1:
            for leaf in tree:
                M[tree, leaf, leaf] = 0
        else:
#            _optOrdering(tree.left)
#            _optOrdering(tree.right)
            
            Vl = set(tree.left)
            Vr = set(tree.right)
            Vlr = set(tree.left.right or tree.left)
            Vll = set(tree.left.left or tree.left)
            Vrr = set(tree.right.right or tree.right)
            Vrl = set(tree.right.left or tree.right)
            other = lambda e, V1, V2: V2 if e in V1 else V1
            tree_left, tree_right = tree.left, tree.right
            for u in Vl:
                for w in Vr:
#                    if True: #Improved search
                        C = min([matrix[m, k] for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)])
                        orderedMs = sorted(other(u, Vll, Vlr), key=lambda m: M[tree_left, u, m])
                        orderedKs = sorted(other(w, Vrl, Vrr), key=lambda k: M[tree_right, w, k])
                        k0 = orderedKs[0]
                        curMin = 1e30000 
                        curM = curK = None
                        for m in orderedMs:
                            if M[tree_left, u, m] + M[tree_right, w, k0] + C >= curMin:
                                break
                            for k in  orderedKs:
                                if M[tree_left, u, m] + M[tree_right, w, k] + C >= curMin:
                                    break
                                testMin = M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
                                if curMin > testMin:
                                    curMin = testMin
                                    curM = m
                                    curK = k
                        M[tree, u, w] = M[tree, w, u] = curMin
                        ordering[tree, u, w] = (tree_left, u, curM, tree_right, w, curK)
                        ordering[tree, w, u] = (tree_right, w, curK, tree_left, u, curM)
#                    else:
#                        def MFunc((m, k)):
#                            return M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
#                        m, k = min([(m, k) for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)], key=MFunc)
#                        M[tree, u, w] = M[tree, w, u] = MFunc((m, k))
#                        ordering[tree, u, w] = (tree_left, u, m, tree_right, w, k)
#                        ordering[tree, w, u] = (tree_right, w, k, tree_left, u, m)

#            if progressCallback:
#                progressCallback(100.0 * len(visitedClusters) / len(tree.mapping))
#                visitedClusters.add(tree)
    from Orange.misc import progressBarMilestones
    
    subtrees = postorder(tree)
    milestones = progressBarMilestones(len(subtrees), 1000)
    
    for i, subtree in enumerate(subtrees):
        _optOrderingIterative(subtree)
        if progressCallback and i in milestones:
            progressCallback(100.0 * i / len(subtrees))

#    def _orderRecursive(tree, u, w):
#        """ Order the tree based on the computed optimal ordering. 
#        """
#        if len(tree)==1:
#            return
#        left, u, m, right, w, k = ordering[tree, u, w]
#        if len(left)>1 and m not in left.right:
#            left.swap()
#        _orderRecursive(left, u, m)
#        
#        if len(right)>1 and k not in right.left:
#            right.swap()
#        _orderRecursive(right, k, w)
        
    def _orderIterative(tree, u, w):
        """ Order the tree based on the computed optimal ordering. 
        """
        opt_uw = {tree: (u, w)}
        for subtree in preorder(tree):
            if subtree.branches:
                u, w = opt_uw[subtree]
                left, u, m, right, w, k = ordering[subtree, u, w]
                opt_uw[left] = (u, m)
                opt_uw[right] = (k, w)
                
                if left.branches and m not in left.right:
                    left.swap()
                
                if right.branches and k not in right.left:
                    right.swap()
    
    u, w = min([(u, w) for u in tree.left for w in tree.right], key=lambda (u, w): M[tree, u, w])
    
##    print "M(v) =", M[tree, u, w]
    
#    with recursion_limit(sys.getrecursionlimit() + len(tree)):
#        _orderRecursive(tree, u, w)
            
    _orderIterative(tree, u, w)
            

#    def _check(tree, u, w):
#        if len(tree)==1:
#            return
#        left, u, m, right, w, k = ordering[tree, u, w]
#        if tree[0] == u and tree[-1] == w:
#            _check(left, u, m)
#            _check(right, k, w)
#        else:
#            print "Error:", u, w, tree[0], tree[-1]
#
#    with recursion_limit(sys.getrecursionlimit() + len(tree)):
#        _check(tree, u ,w)

    if objects:
        tree.mapping.setattr("objects", objects)

""" Matplotlib dendrogram ploting.
"""
try:
    import numpy
except ImportError:
    numpy = None

try:
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.table import Table, Cell
    from matplotlib.text import Text
    from matplotlib.artist import Artist
##    import  matplotlib.pyplot as plt
except (ImportError, IOError, RuntimeError), ex:
    matplotlib = None
    Text , Artist, Table, Cell = object, object, object, object

class TableCell(Cell):
    PAD = 0.05
    def __init__(self, *args, **kwargs):
        Cell.__init__(self, *args, **kwargs)
        self._text.set_clip_on(True)

class TablePlot(Table):
    max_fontsize = 12
    def __init__(self, xy, axes=None, bbox=None):
        Table.__init__(self, axes or plt.gca(), bbox=bbox)
        self.xy = xy
        self.set_transform(self._axes.transData)
        self._fixed_widhts = None
        import matplotlib.pyplot as plt
        self.max_fontsize = plt.rcParams.get("font.size", 12)

    def add_cell(self, row, col, *args, **kwargs):
        xy = (0,0)

        cell = TableCell(xy, *args, **kwargs)
        cell.set_figure(self.figure)
        cell.set_transform(self.get_transform())

        cell.set_clip_on(True)
        cell.set_clip_box(self._axes.bbox)
        cell._text.set_clip_box(self._axes.bbox)
        self._cells[(row, col)] = cell

    def draw(self, renderer):
        if not self.get_visible(): return
        self._update_positions(renderer)

        keys = self._cells.keys()
        keys.sort()
        for key in keys:
            self._cells[key].draw(renderer)

    def _update_positions(self, renderer):
        keys = numpy.array(self._cells.keys())
        cells = numpy.array([[self._cells.get((row, col), None) for col in range(max(keys[:, 1] + 1))] \
                             for row in range(max(keys[:, 0] + 1))])
        
        widths = self._get_column_widths(renderer)
        x = self.xy[0] + numpy.array([numpy.sum(widths[:i]) for i in range(len(widths))])
        y = self.xy[1] - numpy.arange(cells.shape[0]) - 0.5
        
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                cells[i, j].set_xy((x[j], y[i]))
                cells[i, j].set_width(widths[j])
                cells[i, j].set_height(1.0)

        self._width = numpy.sum(widths)
        self._height = cells.shape[0]

        self.pchanged()

    def _get_column_widths(self, renderer):
        keys = numpy.array(self._cells.keys())
        widths = numpy.zeros(len(keys)).reshape((numpy.max(keys[:,0]+1), numpy.max(keys[:,1]+1)))
        fontSize = self._calc_fontsize(renderer)
        for (row, col), cell in self._cells.items():
            cell.set_fontsize(fontSize)
            l, b, w, h = cell._text.get_window_extent(renderer).bounds
            transform = self._axes.transData.inverted()
            x1, _ = transform.transform_point((0, 0))
            x2, _ = transform.transform_point((w + w*TableCell.PAD + 10, 0))
            w = abs(x1 - x2)
            widths[row, col] = w
        return numpy.max(widths, 0)

    def _calc_fontsize(self, renderer):
        transform = self._axes.transData
        _, y1 = transform.transform_point((0, 0))
        _, y2 = transform.transform_point((0, 1))
        return min(max(int(abs(y1 - y2)*0.85) ,4), self.max_fontsize)

    def get_children(self):
        return self._cells.values()

    def get_bbox(self):
        return matplotlib.transform.Bbox([self.xy[0], self.xy[1], self.xy[0] + 10, self.xy[1] + 180])

class DendrogramPlotPylab(object):
    def __init__(self, root, data=None, labels=None, dendrogram_width=None, heatmap_width=None, label_width=None, space_width=None, border_width=0.05, plot_attr_names=False, cmap=None, params={}):
        if not matplotlib:
            raise ImportError("Could not import matplotlib module. Please make sure matplotlib is installed on your system.")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.root = root
        self.data = data
        self.labels = labels if labels else [str(i) for i in range(len(root))]
        self.dendrogram_width = dendrogram_width
        self.heatmap_width = heatmap_width
        self.label_width = label_width
        self.space_width = space_width
        self.border_width = border_width
        self.params = params
        self.plot_attr_names = plot_attr_names

    def plotDendrogram(self):
        self.text_items = []
        def draw_tree(tree):
            if tree.branches:
                points = []
                for branch in tree.branches:
                    center = draw_tree(branch)
                    self.plt.plot([center[0], tree.height], [center[1], center[1]], color="black")
                    points.append(center)
                self.plt.plot([tree.height, tree.height], [points[0][1], points[-1][1]], color="black")
                return (tree.height, (points[0][1] + points[-1][1])/2.0)
            else:
                return (0.0, tree.first)
        draw_tree(self.root)
        
    def plotHeatMap(self):
        import numpy.ma as ma
        import numpy
        dx, dy = self.root.height, 0
        fx, fy = self.root.height/len(self.data.domain.attributes), 1.0
        data, c, w = self.data.toNumpyMA()
        data = (data - ma.min(data))/(ma.max(data) - ma.min(data))
        x = numpy.arange(data.shape[1] + 1)/float(numpy.max(data.shape))
        y = numpy.arange(data.shape[0] + 1)/float(numpy.max(data.shape))*len(self.root)
        self.heatmap_width = numpy.max(x)

        X, Y = numpy.meshgrid(x, y - 0.5)

        self.meshXOffset = numpy.max(X)

        self.plt.jet()
        mesh = self.plt.pcolormesh(X, Y, data[self.root.mapping], edgecolor="b", linewidth=2)

        if self.plot_attr_names:
            names = [attr.name for attr in self.data.domain.attributes]
            self.plt.xticks(numpy.arange(data.shape[1] + 1)/float(numpy.max(data.shape)), names)
        self.plt.gca().xaxis.tick_top()
        for label in self.plt.gca().xaxis.get_ticklabels():
            label.set_rotation(45)

        for tick in self.plt.gca().xaxis.get_major_ticks():
            tick.tick1On = False
            tick.tick2On = False

    def plotLabels_(self):
        import numpy
##        self.plt.yticks(numpy.arange(len(self.labels) - 1, 0, -1), self.labels)
##        for tick in self.plt.gca().yaxis.get_major_ticks():
##            tick.tick1On = False
##            tick.label1On = False
##            tick.label2On = True
##        text = TableTextLayout(xy=(self.meshXOffset+1, len(self.root)), tableText=[[label] for label in self.labels])
        text = TableTextLayout(xy=(self.meshXOffset*1.005, len(self.root) - 1), tableText=[[label] for label in self.labels])
        text.set_figure(self.plt.gcf())
        self.plt.gca().add_artist(text)
        self.plt.gca()._set_artist_props(text)

    def plotLabels(self):
##        table = TablePlot(xy=(self.meshXOffset*1.005, len(self.root) -1), axes=self.plt.gca())
        table = TablePlot(xy=(0, len(self.root) -1), axes=self.plt.gca())
        table.set_figure(self.plt.gcf())
        for i,label in enumerate(self.labels):
            table.add_cell(i, 0, width=1, height=1, text=label, loc="left", edgecolor="w")
        table.set_zorder(0)
        self.plt.gca().add_artist(table)
        self.plt.gca()._set_artist_props(table)
    
    def plot(self, filename=None, show=False):
        self.plt.rcParams.update(self.params)
        labelLen = max(len(label) for label in self.labels)
        w, h = 800, 600
        space = 0.01 if self.space_width == None else self.space_width
        border = self.border_width
        width = 1.0 - 2*border
        height = 1.0 - 2*border
        textLineHeight = min(max(h/len(self.labels), 4), self.plt.rcParams.get("font.size", 12))
        maxTextLineWidthEstimate = textLineHeight*labelLen
##        print maxTextLineWidthEstimate
        textAxisWidthRatio = 2.0*maxTextLineWidthEstimate/w
##        print textAxisWidthRatio
        labelsAreaRatio = min(textAxisWidthRatio, 0.4) if self.label_width == None else self.label_width
        x, y = len(self.data.domain.attributes), len(self.data)

        heatmapAreaRatio = min(1.0*y/h*x/w, 0.3) if self.heatmap_width == None else self.heatmap_width
        dendrogramAreaRatio = 1.0 - labelsAreaRatio - heatmapAreaRatio - 2*space if self.dendrogram_width == None else self.dendrogram_width

        self.fig = self.plt.figure()
        self.labels_offset = self.root.height/20.0
        dendrogramAxes = self.plt.axes([border, border, width*dendrogramAreaRatio, height])
        dendrogramAxes.xaxis.grid(True)
        import matplotlib.ticker as ticker

        dendrogramAxes.yaxis.set_major_locator(ticker.NullLocator())
        dendrogramAxes.yaxis.set_minor_locator(ticker.NullLocator())
        dendrogramAxes.invert_xaxis()
        self.plotDendrogram()
        heatmapAxes = self.plt.axes([border + width*dendrogramAreaRatio + space, border, width*heatmapAreaRatio, height], sharey=dendrogramAxes)

        heatmapAxes.xaxis.set_major_locator(ticker.NullLocator())
        heatmapAxes.xaxis.set_minor_locator(ticker.NullLocator())
        heatmapAxes.yaxis.set_major_locator(ticker.NullLocator())
        heatmapAxes.yaxis.set_minor_locator(ticker.NullLocator())
        
        self.plotHeatMap()
        labelsAxes = self.plt.axes([border + width*(dendrogramAreaRatio + heatmapAreaRatio + 2*space), border, width*labelsAreaRatio, height], sharey=dendrogramAxes)
        self.plotLabels()
        labelsAxes.set_axis_off()
        labelsAxes.xaxis.set_major_locator(ticker.NullLocator())
        labelsAxes.xaxis.set_minor_locator(ticker.NullLocator())
        labelsAxes.yaxis.set_major_locator(ticker.NullLocator())
        labelsAxes.yaxis.set_minor_locator(ticker.NullLocator())
        if filename:
            import matplotlib.backends.backend_agg
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(self.fig)
            canvas.print_figure(filename)
        if show:
            self.plt.show()
        
        
""" Dendrogram ploting using Orange.misc.reander
"""

from orngMisc import ColorPalette, EPSRenderer
class DendrogramPlot(object):
    """ A class for drawing dendrograms
    Example:
    >>> a = DendrogramPlot(tree)
    """
    def __init__(self, tree, attr_tree = None, labels=None, data=None, width=None, height=None, tree_height=None, heatmap_width=None, text_width=None, 
                 spacing=2, cluster_colors={}, color_palette=ColorPalette([(255, 0, 0), (0, 255, 0)]), maxv=None, minv=None, gamma=None, renderer=EPSRenderer):
        self.tree = tree
        self.attr_tree = attr_tree
        self.labels = [str(ex.getclass()) for ex in data] if not labels and data and data.domain.classVar else (labels or [])
#        self.attr_labels = [str(attr.name) for attr in data.domain.attributes] if not attr_labels and data else attr_labels or []
        self.data = data
        self.width, self.height = float(width) if width else None, float(height) if height else None
        self.tree_height = tree_height
        self.heatmap_width = heatmap_width
        self.text_width = text_width
        self.font_size = 10.0
        self.linespacing = 0.0
        self.cluster_colors = cluster_colors
        self.horizontal_margin = 10.0
        self.vertical_margin = 10.0
        self.spacing = float(spacing) if spacing else None
        self.color_palette = color_palette
        self.minv = minv
        self.maxv = maxv
        self.gamma = gamma
        self.set_matrix_color_schema(color_palette, minv, maxv, gamma)
        self.renderer = renderer
        
    def set_matrix_color_schema(self, color_palette, minv, maxv, gamma=None):
        """ Set the matrix color scheme.
        """
        if isinstance(color_palette, ColorPalette):
            self.color_palette = color_palette
        else:
            self.color_palette = ColorPalette(color_palette)
        self.minv = minv
        self.maxv = maxv
        self.gamma = gamma
        
    def color_shema(self):
        vals = [float(val) for ex in self.data for val in ex if not val.isSpecial() and val.variable.varType==orange.VarTypes.Continuous] or [0]
        avg = sum(vals)/len(vals)
        
        maxVal = self.maxv if self.maxv else max(vals)
        minVal = self.minv if self.minv else min(vals)
        
        def _colorSchema(val):
            if val.isSpecial():
                return self.color_palette(None)
            elif val.variable.varType==orange.VarTypes.Continuous:
                r, g, b = self.color_palette((float(val) - minVal) / abs(maxVal - minVal), gamma=self.gamma)
            elif val.variable.varType==orange.VarTypes.Discrete:
                r = g = b = int(255.0*float(val)/len(val.variable.values))
            return (r, g, b)
        return _colorSchema
    
    def layout(self):
        height_final = False
        width_final = False
        tree_height = self.tree_height or 100
        if self.height:
            height, height_final = self.height, True
            heatmap_height = height - (tree_height + self.spacing if self.attr_tree else 0) - 2 * self.horizontal_margin
            font_size =  heatmap_height / len(self.labels) #self.font_size or (height - (tree_height + self.spacing if self.attr_tree else 0) - 2 * self.horizontal_margin) / len(self.labels)
        else:
            font_size = self.font_size
            heatmap_height = font_size * len(self.labels)
            height = heatmap_height + (tree_height + self.spacing if self.attr_tree else 0) + 2 * self.horizontal_margin
             
        text_width = self.text_width or max([len(label) for label in self.labels] + [0]) * font_size #max([self.renderer.string_size_hint(label) for label in self.labels])
        
        if self.width:
            width = self.width
            heatmap_width = width - 2 * self.vertical_margin - tree_height - (2 if self.data else 1) * self.spacing - text_width if self.data else 0
        else:
            heatmap_width = len(self.data.domain.attributes) * heatmap_height / len(self.data) if self.data else 0
            width = 2 * self.vertical_margin + tree_height + (heatmap_width + self.spacing if self.data else 0) + self.spacing + text_width
            
        return width, height, tree_height, heatmap_width, heatmap_height, text_width, font_size
    
    def plot(self, filename="graph.eps"):
        width, height, tree_height, heatmap_width, heatmap_height, text_width, font_size = self.layout()
        heatmap_cell_height = heatmap_height / len(self.labels)

        heatmap_cell_width = 0.0 if not self.data else heatmap_width / len(self.data.domain.attributes)
        
        self.renderer = self.renderer(width, height)
        
        def draw_tree(cluster, root, treeheight, treewidth, color):
            height = treeheight * cluster.height / root.height
            if cluster.branches:
                centers = []
                for branch in cluster.branches:
                    center = draw_tree(branch, root, treeheight, treewidth, self.cluster_colors.get(branch, color))
                    centers.append(center)
                    self.renderer.draw_line(center[0], center[1], center[0], height, stroke_color = self.cluster_colors.get(branch, color))
                    
                self.renderer.draw_line(centers[0][0], height, centers[-1][0], height, stroke_color = self.cluster_colors.get(cluster, color))
                return (centers[0][0] + centers[-1][0]) / 2.0, height
            else:
                return float(treewidth) * cluster.first / len(root), 0.0
        self.renderer.save_render_state()
        self.renderer.translate(self.vertical_margin + tree_height, self.horizontal_margin + (tree_height + self.spacing if self.attr_tree else 0) + heatmap_cell_height / 2.0)
        self.renderer.rotate(90)
#        print self.renderer.transform()
        draw_tree(self.tree, self.tree, tree_height, heatmap_height, self.cluster_colors.get(self.tree, (0,0,0)))
        self.renderer.restore_render_state()
        if self.attr_tree:
            self.renderer.save_render_state()
            self.renderer.translate(self.vertical_margin + tree_height + self.spacing + heatmap_cell_width / 2.0, self.horizontal_margin + tree_height)
            self.renderer.scale(1.0, -1.0)
#            print self.renderer.transform()
            draw_tree(self.attr_tree, self.attr_tree, tree_height, heatmap_width, self.cluster_colors.get(self.attr_tree, (0,0,0)))
            self.renderer.restore_render_state()
        
        self.renderer.save_render_state()
        self.renderer.translate(self.vertical_margin + tree_height + self.spacing, self.horizontal_margin + (tree_height + self.spacing if self.attr_tree else 0))
#        print self.renderer.transform()
        if self.data:
            colorSchema = self.color_shema()
            for i, ii in enumerate(self.tree):
                ex = self.data[ii]
                for j, jj in enumerate((self.attr_tree if self.attr_tree else range(len(self.data.domain.attributes)))):
                    r, g, b = colorSchema(ex[jj])
                    self.renderer.draw_rect(j * heatmap_cell_width, i * heatmap_cell_height, heatmap_cell_width, heatmap_cell_height, fill_color=(r, g, b), stroke_color=(255, 255, 255))
        
        self.renderer.translate(heatmap_width + self.spacing, heatmap_cell_height)
#        print self.renderer.transform()
        self.renderer.set_font("Times-Roman", font_size)
        for index in self.tree: #label in self.labels:
            self.renderer.draw_text(0.0, 0.0, self.labels[index])
            self.renderer.translate(0.0, heatmap_cell_height)
        self.renderer.restore_render_state()
        self.renderer.save(filename)
        
def dendrogram_draw(filename, *args, **kwargs):
    """ Plot the dendrogram to `filename`.
    
    .. todo:: Finish documentation.
    """
    import os
    from orngMisc import PILRenderer, EPSRenderer, SVGRenderer
    name, ext = os.path.splitext(filename)
    kwargs["renderer"] = {".eps":EPSRenderer, ".svg":SVGRenderer, ".png":PILRenderer}.get(ext.lower(), PILRenderer)
#    print kwargs["renderer"], ext
    d = DendrogramPlot(*args, **kwargs)
    d.plot(filename)
    
def postorder(cluster):
    """ Return a post order list of clusters.
    
    :param cluster: Cluster
    :type cluster: :class:`HierarchicalCluster`
    
    """
    order = []
    visited = set()
    stack = [cluster]
    while stack:
        cluster = stack.pop(0)
        
        if cluster.branches:
            if cluster in visited:
                order.append(cluster)
            else:
                stack = cluster.branches + [cluster] + stack
                visited.add(cluster)
        else:
            order.append(cluster)
            visited.add(cluster)
    return order
    
    
def preorder(cluster):
    """ Return a pre order list of clusters.
    
    :param cluster: Cluster
    :type cluster: :class:`HierarchicalCluster`
    
    """
    order = []
    stack = [cluster]
    while stack:
        cluster = stack.pop(0)
        order.append(cluster)
        if cluster.branches:
            stack = cluster.branches + stack
    return order
    
    
def dendrogram_layout(root_cluster, expand_leaves=False):
    """ Return a layout of the cluster dendrogram on a 2D plane. The return 
    value if a list of (subcluster, (start, center, end)) tuples where
    subcluster is an instance of :class:`HierarchicalCluster` and start,
    end are the two end points for the cluster branch. The list is sorted
    in post-order.
    
    :param root_cluster: Cluster to layout.
    :type root_cluster: :class:`HierarchicalCluster`
    
    :param expand_leaves: If True leaves will span proportional to the number
        of items they map, else all leaves will be the same width. 
     
    """
    result = []
    cluster_geometry = {}
    leaf_idx = 0
    for cluster in postorder(root_cluster):
        if not cluster.branches:
            if expand_leaves:
                start, end = float(cluster.first), float(cluster.last)
            else:
                start = end = leaf_idx
                leaf_idx += 1
            center = (start + end) / 2.0
            cluster_geometry[cluster] = (start, center, end)
            result.append((cluster, (start, center, end)))
        else:
            left = cluster.branches[0]
            right = cluster.branches[1]
            left_center = cluster_geometry[left][1]
            right_center = cluster_geometry[right][1]
            start, end = left_center, right_center
            center = (start + end) / 2.0
            cluster_geometry[cluster] = (start, center, end)
            result.append((cluster, (start, center, end)))
            
    return result
    
def clone(cluster):
    """ Clone a cluster, including it's subclusters.
    
    :param cluster: Cluster to clone
    :type cluster: :class:`HierarchicalCluster`
    
    """
    import copy
    clones = {}
    mapping = copy.copy(cluster.mapping)
    for node in postorder(cluster):
        node_clone = copy.copy(node)
        if node.branches:
            node_clone.branches = [clones[b] for b in node.branches]
        node_clone.mapping = mapping
        clones[node] = node_clone
        
    return clones[cluster]
    
def pruned(root_cluster, level=None, height=None, condition=None):
    """ Return a new pruned clustering instance.
    
    .. note:: This uses :obj:`clone` to create a copy of the `root_cluster`
        instance.
    
    :param cluster: Cluster to prune.
    :type cluster: :class:`HierarchicalCluster`
    
    :param level: If not None prune all clusters deeper then `level`.
    :type level: int
    
    :param height: If not None prune all clusters with height lower then
        `height`.
    :type height: float
    
    :param condition: If not None condition must be a function taking a
        single :class:`HierarchicalCluster` instance and returning a True 
        or False value indicating if the cluster should be pruned.
    :type condition: function 
    
    """
    root_cluster = clone(root_cluster)
    prune(root_cluster, level, height, condition)
    return root_cluster
    
    
def prune(root_cluster, level=None, height=None, condition=None):
    """ Prune clustering instance in place
    
    :param cluster: Cluster to prune.
    :type cluster: :class:`HierarchicalCluster`
    
    :param level: If not None prune all clusters deeper then `level`.
    :type level: int
    
    :param height: If not None prune all clusters with height lower then
        `height`.
    :type height: float
    
    :param condition: If not None condition must be a function taking a
        single :class:`HierarchicalCluster` instance and returning a True 
        or False value indicating if the cluster should be pruned.
    :type condition: function
    
    """
    if not any(arg is not None for arg in [level, height, condition]):
        raise ValueError("At least one pruning argument must be supplied")
    
    level_check = height_check = condition_check = lambda cl: False
    cluster_depth = cluster_depths(root_cluster)
    
    if level is not None:
        level_check = lambda cl: cluster_depth[cl] > level
        
    if height is not None:
        height_check = lambda cl: cl.height < height

    if condition is not None:
        condition_check = condition
        
    pruned_set = set()
    
    def check_all(cl):
        return any([check(cl) for check in [level_check, height_check,
                                            condition_check]])
        
    for cluster in preorder(root_cluster):
        if cluster not in pruned_set:
            if check_all(cluster):
                cluster.branches = None
                pruned_set.update(set(preorder(cluster)))
            else:
                pass
    
    
def cluster_depths(root_cluster):
    """ Return a dictionary mapping :class:`HierarchicalCluster` instances to
    their depths in the `root_cluster` hierarchy.
    
    """
    depths = {}
    depths[root_cluster] = 0
    for cluster in preorder(root_cluster):
        cl_depth = depths[cluster]
        if cluster.branches:
            depths.update(dict.fromkeys(cluster.branches, cl_depth + 1))
    return depths


def instance_distance_matrix(data,
            distance_constructor=orange.ExamplesDistanceConstructor_Euclidean,
            progress_callback=None):
    """ A helper function that computes an :class:`Orange.core.SymMatrix` of all
    pairwise distances between instances in `data`.
    
    :param data: A data table
    :type data: :class:`Orange.data.Table`
    
    :param distance_constructor: An ExamplesDistance_Constructor instance.
    :type distance_constructor: :class:`Orange.distances.ExampleDistConstructor`
    
    """
    matrix = orange.SymMatrix(len(data))
    dist = distance_constructor(data)
    
    iter_count = matrix.dim * (matrix.dim - 1) / 2
    milestones = progressBarMilestones(iter_count, 100)
    
    for count, ((i, ex1), (j, ex2)) in enumerate(_pairs(enumerate(data))):
        matrix[i, j] = dist(ex1, ex2)
        if progress_callback and count in milestones:
            progress_callback(100.0 * count / iter_count)
            
    return matrix 


def feature_distance_matrix(data, distance=None, progress_callback=None):
    """ A helper function that computes an :class:`Orange.core.SymMatrix` of
    all pairwise distances between features in `data`.
    
    :param data: A data table
    :type data: :class:`Orange.data.Table`
    :param distance: a function taking two lists and returning the distance.
    :type distance: function
     
    """
    attributes = data.domain.attributes
    matrix = orange.SymMatrix(len(attributes))
    iter_count = matrix.dim * (matrix.dim - 1) / 2
    milestones = progressBarMilestones(iter_count, 100)
    
    for count, ((i, a1), (j, a2)) in enumerate(_pairs(enumerate(attributes))):
        matrix[i, j] = (1.0 - orange.PearsonCorrelation(a1, a2, data, 0).r) / 2.0
        if progress_callback and count in milestones:
            progress_callback(100.0 * count / iter_count)
            
    return matrix


def _pairs(seq, same = False):
    """ Return all pairs from elements of `seq`.
    """
    seq = list(seq)
    same = 0 if same else 1
    for i in range(len(seq)):
        for j in range(i + same, len(seq)):
            yield seq[i], seq[j]
    
    
def joining_cluster(root_cluster, item1, item2):
    """ Return the cluster where `item1` and `item2` are first joined
    
    :param root_cluster: Clustering.
    :type root_cluster: :class:`HierarchicalCluster`
    :param item1: An element of `root_cluster.mapping` or `root_cluster.mapping.objects`
    :param item2: An element of `root_cluster.mapping` or `root_cluster.mapping.objects`
    
    """
    cluster = root_cluster
    while True:
        if cluster.branches:
            for branch in cluster.branches:
                if item1 in branch and item2 in branch:
                    cluster = branch
                    break
            else:
                return cluster
        else:
            return cluster
        

def cophenetic_distances(root_cluster):
    """ Return the cophenetic distance matrix between items in clustering.
    Cophenetic distance is defined as the height of the cluster where the 
    two items are first joined.
    
    :param root_cluster: Clustering.
    :type root_cluster: :class:`HierarchicalCluster`
     
    """

    mapping = root_cluster.mapping  
    matrix = Orange.core.SymMatrix(len(mapping))
    for cluster in postorder(root_cluster):
        if cluster.branches:
            for branch1, branch2 in _pairs(cluster.branches):
                for idx1 in mapping[branch1.first: branch1.last]:
                    for idx2 in mapping[branch2.first: branch2.last]:
                        matrix[idx1, idx2] = cluster.height
                
        else:
            for ind1, ind2 in _pairs(mapping[cluster.first: cluster.last]):
                matrix[ind1, ind2] = cluster.height
    
    return matrix
    
def cophenetic_correlation(root_cluster, matrix):
    """ Return the `cophenetic correlation coefficient
    <http://en.wikipedia.org/wiki/Cophenetic_correlation>`_ of the given
    clustering.
    
    :param root_cluster: Clustering.
    :type root_cluster: :class:`HierarchicalCluster`
    
    :param matrix: The distance matrix from which the `root_cluster` was
        derived.
     
    """
    import numpy
    cophenetic = cophenetic_distances(root_cluster)
    cophenetic = [list(row) for row in cophenetic]
    original = [list(row) for row in matrix]
    cophenetic = numpy.ravel(cophenetic)
    original = numpy.ravel(original)
    return numpy.corrcoef(cophenetic, original)[0, 1]
    
    
if __name__=="__main__":
    data = orange.ExampleTable("doc//datasets//brown-selected.tab")
#    data = orange.ExampleTable("doc//datasets//iris.tab")
    root = hierarchicalClustering(data, order=True) #, linkage=orange.HierarchicalClustering.Single)
    attr_root = hierarchicalClustering_attributes(data, order=True)
#    print root
#    d = DendrogramPlotPylab(root, data=data, labels=[str(ex.getclass()) for ex in data], dendrogram_width=0.4, heatmap_width=0.3,  params={}, cmap=None)
#    d.plot(show=True, filename="graph.png")

    dendrogram_draw("graph.eps", root, attr_tree=attr_root, data=data, labels=[str(e.getclass()) for e in data], tree_height=50, #width=500, height=500,
                          cluster_colors={root.right:(255,0,0), root.right.right:(0,255,0)}, 
                          color_palette=ColorPalette([(255, 0, 0), (0,0,0), (0, 255,0)], gamma=0.5, 
                                                     overflow=(255, 255, 255), underflow=(255, 255, 255))) #, minv=-0.5, maxv=0.5)
