"""

.. index:: clustering

Everything about clustering, including agglomerative and hierarchical clustering.

==================
k-Means clustering
==================

.. index:: aglomerative clustering

.. autoclass:: Orange.cluster.KMeans
   :members:

Examples
========

The following code runs k-means clustering and prints out the cluster indexes for the last 10 data instances (`kmeans-run.py`_, uses `iris.tab`_):

.. literalinclude:: code/kmeans-run.py

The output of this code is::

    [1, 1, 2, 1, 1, 1, 2, 1, 1, 2]

Invoking a call-back function may be useful when tracing the progress of the clustering. Below is a code that uses an :obj:`inner_callback` to report on the number of instances that have changed the cluster and to report on the clustering score. For the score to be computed at each iteration we have to set :obj:`minscorechange`, but we can leave it at 0 or even set it to a negative value, which allows the score to deteriorate by some amount (`kmeans-run-callback.py`_, uses `iris.tab`_):

.. literalinclude:: code/kmeans-run-callback.py

The convergence on Iris data set is fast::

    Iteration: 1, changes: 150, score: 10.9555
    Iteration: 2, changes: 12, score: 10.3867
    Iteration: 3, changes: 2, score: 10.2034
    Iteration: 4, changes: 2, score: 10.0699
    Iteration: 5, changes: 2, score: 9.9542
    Iteration: 6, changes: 1, score: 9.9168
    Iteration: 7, changes: 2, score: 9.8624
    Iteration: 8, changes: 0, score: 9.8624

Call-back above is used for reporting of the progress, but may as well call a function that plots a selection data projection with corresponding centroid at a given step of the clustering. This is exactly what we did with the following script (`kmeans-trace.py`_, uses `iris.tab`_):

.. literalinclude:: code/kmeans-trace.py

Only the first four scatterplots are shown below. Colors of the data instances indicate the cluster membership. Notice that since the Iris data set includes four attributes, the closest centroid in a particular 2-dimensional projection is not necessary also the centroid of the cluster that the data point belongs to.


.. image:: files/kmeans-scatter-001.png

.. image:: files/kmeans-scatter-002.png

.. image:: files/kmeans-scatter-003.png

.. image:: files/kmeans-scatter-004.png


k-Means Utility Functions
=========================

.. automethod:: Orange.cluster.kmeans_init_random

.. automethod:: Orange.cluster.kmeans_init_diversity

.. automethod:: Orange.cluster.KMeans_init_hierarchicalClustering

.. automethod:: Orange.cluster.data_center

.. automethod:: Orange.cluster.data_center

.. automethod:: Orange.cluster.plot_silhouette

.. automethod:: Orange.cluster.score_distance_to_centroids

.. automethod:: Orange.cluster.score_silhouette

.. automethod:: Orange.cluster.score_fastsilhouette

Typically, the choice of seeds has a large impact on the k-means clustering, with better initialization methods yielding a clustering that converges faster and finds more optimal centroids. The following code compares three different initialization methods (random, diversity-based and hierarchical clustering-based) in terms of how fast they converge (`kmeans-cmp-init.py`_, uses `iris.tab`_, `housing.tab`_, `vehicle.tab`_):

.. literalinclude:: code/kmeans-cmp-init.py

As expected, k-means converges faster with diversity and clustering-based initialization that with random seed selection::

               Rnd Div  HC
          iris  12   3   4
       housing  14   6   4
       vehicle  11   4   3

The following code computes the silhouette score for k=2..7 and plots a silhuette plot for k=3 (`kmeans-silhouette.py`_, uses `iris.tab`_):

.. literalinclude:: code/kmeans-silhouette.py

The analysis suggests that k=2 is preferred as it yields the maximal silhouette coefficient::

    2 0.629467553352
    3 0.504318855054
    4 0.407259377854
    5 0.358628975081
    6 0.353228492088
    7 0.366357876944

.. figure:: files/kmeans-silhouette.png

   Silhouette plot for k=3.

.. _iris.tab: code/iris.tab
.. _housing.tab: code/housing.tab
.. _vehicle.tab: code/vehicle.tab
.. _kmeans-run.py: code/kmeans-run.py
.. _kmeans-run-callback.py: code/kmeans-run-callback.py
.. _kmeans-trace.py: code/kmeans-trace.py
.. _kmeans-cmp-init.py: code/kmeans-cmp-init.py
.. _kmeans-silhouette.py: code/kmeans-sillhouette.py

================
Other clustering
================

.. class:: Orange.cluster.ExampleCluster

   To je pa moj dodaten opis ExampleClusterja

   .. method:: fibonaci(n)

      Funkcija, ki izracuna fibonacija.

      :param n: katero fibonacijevo stevilo zelis
      :type n: integer
   

.. autoclass:: Orange.cluster.HierarchicalCluster
   :members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: Orange.cluster.HierarchicalClusterList
   :members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: Orange.cluster.HierarchicalClustering
   :members:
   :show-inheritance:
   :undoc-members:



"""

from __future__ import with_statement

from orange import \
    ExampleCluster, HierarchicalCluster, HierarchicalClusterList, HierarchicalClustering

import math
import sys
import orange
import random
import statc
    
__docformat__ = 'restructuredtext'

# miscellaneous functions (used across this module)

class ExampleCluster2(orange.ExampleCluster):
    """New example cluster"""
    pass

def _modus(dist):
    #Check bool(dist) - False means no known cases
    #Check dist.cases > 0 - We cant return some value from the domain without knowing if it is even present
    #in the data. TOOD: What does this mean for k-means convergence?
    if bool(dist) and dist.cases > 0:
        return dist.modus()
    else:
        return None
    
def data_center(data):
    """
    Returns a center of the instances in the data set (average across data instances for continuous attributes, most frequent value for discrete attributes).
    """
    atts = data.domain.attributes
    astats = orange.DomainBasicAttrStat(data)
    center = [astats[a].avg if a.varType == orange.VarTypes.Continuous \
#              else max(enumerate(orange.Distribution(a, data)), key=lambda x:x[1])[0] if a.varType == orange.VarTypes.Discrete
              else _modus(orange.Distribution(a, data)) if a.varType == orange.VarTypes.Discrete
              else None
              for a in atts]
    if data.domain.classVar:
        center.append(0)
    return orange.Example(data.domain, center)

def minindex(x):
    """Return the index of the minimum element"""
    return x.index(min(x))

def avg(x):
    """Return the average (mean) of a given list"""
    return (float(sum(x)) / len(x)) if x else 0

#
# data distances
#

class ExamplesDistanceConstructor_PearsonR(orange.ExamplesDistanceConstructor):
    def __new__(cls, data=None, **argkw):
        self = orange.ExamplesDistanceConstructor.__new__(cls, **argkw)
        self.__dict__.update(argkw)
        if data:
            return self.__call__(data)
        else:
            return self

    def __call__(self, data):
        indxs = [i for i, a in enumerate(data.domain.attributes) if a.varType==orange.VarTypes.Continuous]
        return ExamplesDistance_PearsonR(domain=data.domain, indxs=indxs)

class ExamplesDistance_PearsonR(orange.ExamplesDistance):
    """ Pearson distance. (1 - R)/2, where R is the Pearson correlation between examples. In [0..1]. """

    def __init__(self, **argkw):
        self.__dict__.update(argkw)
    def __call__(self, e1, e2):
        X1 = []; X2 = []
        for i in self.indxs:
            if not(e1[i].isSpecial() or e2[i].isSpecial()):
                X1.append(float(e1[i]))
                X2.append(float(e2[i]))
        if not X1:
            return 1.0
        try:
            return (1.0 - statc.pearsonr(X1, X2)[0]) / 2.
        except:
            return 1.0

class ExamplesDistanceConstructor_SpearmanR(orange.ExamplesDistanceConstructor):
    def __new__(cls, data=None, **argkw):
        self = orange.ExamplesDistanceConstructor.__new__(cls, **argkw)
        self.__dict__.update(argkw)
        if data:
            return self.__call__(data)
        else:
            return self

    def __call__(self, data):
        indxs = [i for i, a in enumerate(data.domain.attributes) if a.varType==orange.VarTypes.Continuous]
        return ExamplesDistance_SpearmanR(domain=data.domain, indxs=indxs)

class ExamplesDistance_SpearmanR(orange.ExamplesDistance):
    def __init__(self, **argkw):
        self.__dict__.update(argkw)
    def __call__(self, e1, e2):
        X1 = []; X2 = []
        for i in self.indxs:
            if not(e1[i].isSpecial() or e2[i].isSpecial()):
                X1.append(float(e1[i]))
                X2.append(float(e2[i]))
        if not X1:
            return 1.0
        try:
            return (1.0 - statc.spearmanr(X1, X2)[0]) / 2.
        except:
            return 1.0

# k-means clustering

# clustering scoring functions 

def score_distance_to_centroids(km):
    """Returns an average distance of data instances to their associated centroids.

    :param km: a k-means clustering object.
    :type km: :class:`KMeans`
    """
    return sum(km.distance(km.centroids[km.clusters[i]], d) for i,d in enumerate(km.data))

score_distance_to_centroids.minimize = True

def score_conditionalEntropy(km):
    """UNIMPLEMENTED cluster quality measured by conditional entropy"""
    pass

def score_withinClusterDistance(km):
    """UNIMPLEMENTED weighted average within-cluster pairwise distance"""
    pass

score_withinClusterDistance.minimize = True

def score_betweenClusterDistance(km):
    """Sum of distances from elements to 'nearest miss' centroids"""
    return sum(min(km.distance(c, d) for j,c in enumerate(km.centroids) if j!=km.clusters[i]) for i,d in enumerate(km.data))

def score_silhouette(km, index=None):
    """Returns an average silhouette score of data instances.
    
    :param km: a k-means clustering object.
    :type km: :class:`KMeans`

    :param index: if given, the functon returns just the silhouette score of that particular data instance.
    :type index: integer
    """
    
    if index == None:
        return avg([score_silhouette(km, i) for i in range(len(km.data))])
    cind = km.clusters[index]
    a = avg([km.distance(km.data[index], ex) for i, ex in enumerate(km.data) if
             km.clusters[i] == cind and i != index])
    b = min([avg([km.distance(km.data[index], ex) for i, ex in enumerate(km.data) if
                 km.clusters[i] == c])
            for c in range(len(km.centroids)) if c != cind])
    return float(b - a) / max(a, b) if max(a, b) > 0 else 0.0

def score_fastsilhouette(km, index=None):
    """Same as score_silhouette, but computes an approximation and is faster.
    
    :param km: a k-means clustering object.
    :type km: :class:`KMeans`
    """

    if index == None:
        return avg([score_fastsilhouette(km, i) for i in range(len(km.data))])
    cind = km.clusters[index]
    a = km.distance(km.data[index], km.centroids[km.clusters[index]])
    b = min([km.distance(km.data[index], c) for i,c in enumerate(km.centroids) if i != cind])
    return float(b - a) / max(a, b) if max(a, b) > 0 else 0.0

def compute_bic(km):
	"""Compute bayesian information criteria score for given clustering. NEEDS REWRITE!!!"""
	data = km.data
	medoids = km.centroids

	M = len(data.domain.attributes)
	R = float(len(data))
	Ri = [km.clusters.count(i) for i in range(km.k)]
	numFreePar = (len(km.data.domain.attributes) + 1.) * km.k * math.log(R, 2.) / 2.
	# sigma**2
	s2 = 0.
	cidx = [i for i, attr in enumerate(data.domain.attributes) if attr.varType in [orange.VarTypes.Continuous, orange.VarTypes.Discrete]]
	for x, midx in izip(data, mapping):
		medoid = medoids[midx] # medoids has a dummy element at the beginning, so we don't need -1 
		s2 += sum( [(float(x[i]) - float(medoid[i]))**2 for i in cidx] )
	s2 /= (R - K)
	if s2 < 1e-20:
		return None, [None]*K
	# log-lokehood of clusters: l(Dn)
	# log-likehood of clustering: l(D)
	ld = 0
	bicc = []
	for k in range(1, 1+K):
		ldn = -1. * Ri[k] * ((math.log(2. * math.pi, 2) / -2.) - (M * math.log(s2, 2) / 2.) + (K / 2.) + math.log(Ri[k], 2) - math.log(R, 2))
		ld += ldn
		bicc.append(ldn - numFreePar)
	return ld - numFreePar, bicc


#
# silhouette plot
#

def plot_silhouette(km, filename='tmp.png', fast=False):
    """ Saves a silhuette plot to filename, showing the distributions of silhouette scores in clusters. kmeans is a k-means clustering object. If fast is True use score_fastsilhouette to compute scores instead of score_silhouette.

    :param km: a k-means clustering object.
    :type km: :class:`KMeans`
    :param filename: name of output plot.
    :type filename: string
    :param fast: if True use :func:`score_fastsilhouette` to compute scores instead of :func:`score_silhouette`
    :type fast: boolean.

    """
    import matplotlib.pyplot as plt
    plt.figure()
    scoring = score_fastsilhouette if fast else score_silhouette
    scores = [[] for i in range(km.k)]
    for i, c in enumerate(km.clusters):
        scores[c].append(scoring(km, i))
    csizes = map(len, scores)
    cpositions = [sum(csizes[:i]) + (i+1)*3 + csizes[i]/2 for i in range(km.k)]
    scores = reduce(lambda x,y: x + [0]*3 + sorted(y), scores, [])
    plt.barh(range(len(scores)), scores, linewidth=0, color='c')
    plt.yticks(cpositions, map(str, range(km.k)))
    #plt.title('Silhouette plot')
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette value')
    plt.savefig(filename)

# clustering initialization (seeds)
# initialization functions should be of the type f(data, k, distfun)

def kmeans_init_random(data, k, _):
    """A function that can be used for initialization of k-means clustering returns k data instances from the data. This type of initialization is also known as Fory's initialization (Forgy, 1965; He et al., 2004).
    
    :param data: data instances.
    :type data: :class:`orange.ExampleTable`
    :param k: the number of clusters.
    :type k: integer
    :param distfun: a distance function.
    :type distfun: :class:`orange.ExamplesDistance`
     """
    return data.getitems(random.sample(range(len(data)), k))

def kmeans_init_diversity(data, k, distfun):
    """A function that can be used for intialization of k-means clustering. Returns a set of centroids where the first one is a data point being the farthest away from the center of the data, and consequent centroids data points of which the minimal distance to the previous set of centroids is maximal. Differs from the initialization proposed by Katsavounidis et al. (1994) only in the selection of the first centroid (where they use a data instance with the highest norm).

    :param data: data instances.
    :type data: :class:`orange.ExampleTable`
    :param k: the number of clusters.
    :type k: integer
    :param distfun: a distance function.
    :type distfun: :class:`orange.ExamplesDistance`
    """
    center = data_center(data)
    # the first seed should be the farthest point from the center
    seeds = [max([(distfun(d, center), d) for d in data])[1]]
    # other seeds are added iteratively, and are data points that are farthest from the current set of seeds
    for i in range(1,k):
        seeds.append(max([(min([distfun(d, s) for s in seeds]), d) for d in data if d not in seeds])[1])
    return seeds

class KMeans_init_hierarchicalClustering():
    """Is actually a class that returns an clustering initialization function that performs hierarhical clustering, uses it to infer k clusters, and computes a list of cluster-based data centers."""

    def __init__(self, n=100):
        """
        :param n: number of data instances to sample.
        :type n: integer
        """
        self.n = n

    def __call__(self, data, k, disfun):
        """
        :param data: data instances.
        :type data: :class:`orange.ExampleTable`
        :param k: the number of clusters.
        :type k: integer
        :param distfun: a distance function.
        :type distfun: :class:`orange.ExamplesDistance`
        """
        sample = orange.ExampleTable(random.sample(data, min(self.n, len(data))))
        root = hierarchicalClustering(sample)
        cmap = hierarchicalClustering_topClusters(root, k)
        return [data_center(orange.ExampleTable([sample[e] for e in cl])) for cl in cmap]

#    
# k-means clustering, main implementation
#

class KMeans:
    """
    Implements a k-means clustering algorithm:

    #. Choose the number of clusters, k.
    #. Choose a set of k initial centroids.
    #. Assign each instances in the data set to the closest centroid.
    #. For each cluster, compute a new centroid as a center of clustered 
       data instances.
    #. Repeat the previous two steps, until some convergence criterion is 
       met (e.g., the cluster assignment has not changed).

    The main advantages of this algorithm are simplicity and low memory  
    requirements. The principal disadvantage is the dependence of results 
    on the selection of initial set of centroids.

    :var k: Number of clusters.
    :var data: Instances to cluster.
    :var centroids: Current set of centroids. 
    :var scoring: Current clustering score.
    :var iteration: Current clustering iteration.
    :var clusters:  A list of cluster indexes. An i-th element provides an index to a centroid associated with i-th data instance from the input data set. 
    """

    def __init__(self, data=None, centroids=3, maxiters=None, minscorechange=None,
                 stopchanges=0, nstart=1, initialization=kmeans_init_random,
                 distance=orange.ExamplesDistanceConstructor_Euclidean,
                 scoring=score_distance_to_centroids, inner_callback = None,
                 outer_callback = None):
        """
        :param data: Data instances to be clustered. If not None, clustering will be executed immediately after initialization unless initialize_only=True.
        :type data: :class:`orange.ExampleTable` or None
        :param centroids: either specify a number of clusters or provide a list of examples that will serve as clustering centroids.
        :type centroids: integer or a list of :class:`orange.Example`
        :param nstart: If greater than one, nstart runs of the clustering algorithm will be executed, returning the clustering with the best (lowest) score.
        :type nstart: integer
        :param distance: an example distance constructor, which measures the distance between two instances.
        :type distance: :class:`orange.ExamplesDistanceConstructor`
        :param initialization: a function to select centroids given data instances, k and a example distance function. This module implements different approaches (:func:`kmeans_init_random`, :func:`kmeans_init_diversity`, :class:`KMeans_init_hierarchicalClustering`). 
        :param scoring: a function that takes clustering object and returns the clustering score. It could be used, for instance, in procedure that repeats the clustering nstart times, returning the clustering with the lowest score.
        :param inner_callback: invoked after every clustering iteration.
        :param outer_callback: invoked after every clustering restart (if nstart is greater than 1).

        Stopping criteria:

        :param maxiters: maximum number of clustering iterations
        :type maxiters: integer
        :param minscorechange: minimal improvement of the score from previous generation (if lower, the clustering will stop). If None, the score will not be computed between iterations
        :type minscorechange: float or None
        :param stopchanges: if the number of instances changing the cluster is lower or equal to stopchanges, stop the clustering.
        :type stopchanges: integer
        """

        self.data = data
        self.k = centroids if type(centroids)==int else len(centroids)
        self.centroids = centroids if type(centroids) == orange.ExampleTable else None
        self.maxiters = maxiters
        self.minscorechange = minscorechange
        self.stopchanges = stopchanges
        self.nstart = nstart
        self.initialization = initialization
        self.distance_constructor = distance
        self.distance = self.distance_constructor(self.data) if self.data else None
        self.scoring = scoring
        self.minimize_score = True if hasattr(scoring, 'minimize') else False
        self.inner_callback = inner_callback
        self.outer_callback = outer_callback
        if self.data:
            self.run()
        
    def __call__(self, data = None):
        """Runs the k-means clustering algorithm, with optional new data."""
        if data:
            self.data = data
            self.distance = self.distance_constructor(self.data)
        self.run()
    
    def init_centroids(self):
        """Initialize cluster centroids"""
        if self.centroids and not self.nstart > 1: # centroids were specified
            return
        self.centroids = self.initialization(self.data, self.k, self.distance)
        
    def compute_centeroid(self, data):
        """Return a centroid of the data set."""
        return data_center(data)
    
    def compute_cluster(self):
        """calculate membership in clusters"""
        return [minindex([self.distance(s, d) for s in self.centroids]) for d in self.data]
    
    def runone(self):
        """Runs a single clustering iteration, starting with re-computation of centroids, followed by computation of data membership (associating data instances to their nearest centroid).""" 
        self.centroids = [self.compute_centeroid(self.data.getitems(
            [i for i, c in enumerate(self.clusters) if c == cl])) for cl in range(self.k)]
        self.clusters = self.compute_cluster()
        
    def run(self):
        """
        Runs clustering until the convergence conditions are met. If nstart is greater than one, nstart runs of the clustering algorithm will be executed, returning the clustering with the best (lowest) score.
        """
        self.winner = None
        for startindx in range(self.nstart):
            self.init_centroids()
            self.clusters = old_cluster = self.compute_cluster()
            if self.minscorechange != None:
                self.score = old_score = self.scoring(self)
            self.nchanges = len(self.data)
            self.iteration = 0
            stopcondition = False
            if self.inner_callback:
                self.inner_callback(self)
            while not stopcondition:
                self.iteration += 1
                self.runone()
                self.nchanges = sum(map(lambda x,y: x!=y, old_cluster, self.clusters))
                old_cluster = self.clusters
                if self.minscorechange != None:
                    self.score = self.scoring(self)
                    scorechange = (self.score - old_score) / old_score if old_score > 0 else self.minscorechange
                    if self.minimize_score:
                        scorechange = -scorechange
                    old_score = self.score
                stopcondition = (self.nchanges <= self.stopchanges or
                                 self.iteration == self.maxiters or
                                 (self.minscorechange != None and
                                  scorechange <= self.minscorechange))
                if self.inner_callback:
                    self.inner_callback(self)
            if self.scoring and self.minscorechange == None:
                self.score = self.scoring(self)
            if self.nstart > 1:
                if not self.winner or (self.score < self.winner[0] if
                        self.minimize_score else self.score > self.winner[0]):
                    self.winner = (self.score, self.clusters, self.centroids)
                if self.outer_callback:
                    self.outer_callback(self)

        if self.nstart > 1:
            self.score, self.clusters, self.centroids = self.winner

# hierarhical clustering

def hierarchicalClustering(data,
                           distanceConstructor=orange.ExamplesDistanceConstructor_Euclidean,
                           linkage=orange.HierarchicalClustering.Average,
                           order=False,
                           progressCallback=None):
    """Return a hierarhical clustering of the data set."""
    distance = distanceConstructor(data)
    matrix = orange.SymMatrix(len(data))
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = distance(data[i], data[j])
    root = orange.HierarchicalClustering(matrix, linkage=linkage, progressCallback=(lambda value, obj=None: progressCallback(value*100.0/(2 if order else 1))) if progressCallback else None)
    if order:
        orderLeaves(root, matrix, progressCallback=(lambda value: progressCallback(50.0 + value/2)) if progressCallback else None)
    return root

def hierarchicalClustering_attributes(data, distance=None, linkage=orange.HierarchicalClustering.Average, order=False, progressCallback=None):
    """Return hierarhical clustering of attributes in the data set."""
    matrix = orange.SymMatrix(len(data.domain.attributes))
    for a1 in range(len(data.domain.attributes)):
        for a2 in range(a1):
            matrix[a1, a2] = (1.0 - orange.PearsonCorrelation(a1, a2, data, 0).r) / 2.0
    root = orange.HierarchicalClustering(matrix, linkage=linkage, progressCallback=(lambda value, obj=None: progressCallback(value*100.0/(2 if order else 1))) if progressCallback else None)
    if order:
        orderLeaves(root, matrix, progressCallback=(lambda value: progressCallback(50.0 + value/2)) if progressCallback else None)
    return root

def hierarchicalClustering_clusterList(node, prune=None):
    """Return a list of clusters down from the node of hierarchical clustering."""
    if prune:
        if len(node) <= prune:
            return [] 
    if node.branches:
        return [node] + hierarchicalClustering_clusterList(node.left, prune) + hierarchicalClustering_clusterList(node.right, prune)
    return [node]

def hierarchicalClustering_topClusters(root, k):
    """Return k topmost clusters from hierarchical clustering."""
    candidates = set([root])
    while len(candidates) < k:
        repl = max([(max(c.left.height, c.right.height), c) for c in candidates if c.branches])[1]
        candidates.discard(repl)
        candidates.add(repl.left)
        candidates.add(repl.right)
    return candidates

def hierarhicalClustering_topClustersMembership(root, k):
    """Return data instances' cluster membership (list of indices) to k topmost clusters."""
    clist = hierarchicalClustering_topClusters(root, k)
    cmap = [None] * len(root)
    for i, c in enumerate(clist):
        for e in c:
            cmap[e] = i
    return cmap

def orderLeaves(tree, matrix, progressCallback=None):
    """Order the leaves in the clustering tree.

    (based on Ziv Bar-Joseph et al. (Fast optimal leaf ordering for hierarchical clustering')
    Arguments:
        tree   --binary hierarchical clustering tree of type orange.HierarchicalCluster
        matrix --orange.SymMatrix that was used to compute the clustering
        progressCallback --function used to report progress
    """
    objects = getattr(tree.mapping, "objects", None)
    tree.mapping.setattr("objects", range(len(tree)))
    M = {}
    ordering = {}
    visitedClusters = set()
    def _optOrdering(tree):
        if len(tree)==1:
            for leaf in tree:
                M[tree, leaf, leaf] = 0
##                print "adding:", tree, leaf, leaf
        else:
            _optOrdering(tree.left)
            _optOrdering(tree.right)
##            print "ordering", [i for i in tree]
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
                    if True: #Improved search
                        C = min([matrix[m, k] for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)])
                        orderedMs = sorted(other(u, Vll, Vlr), key=lambda m: M[tree_left, u, m])
                        orderedKs = sorted(other(w, Vrl, Vrr), key=lambda k: M[tree_right, w, k])
                        k0 = orderedKs[0]
                        curMin = 1e30000 
                        curMK = ()
                        for m in orderedMs:
                            if M[tree_left, u, m] + M[tree_right, w, k0] + C >= curMin:
                                break
                            for k in  orderedKs:
                                if M[tree_left, u, m] + M[tree_right, w, k] + C >= curMin:
                                    break
                                if curMin > M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]:
                                    curMin = M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
                                    curMK = (m, k)
                        M[tree, u, w] = M[tree, w, u] = curMin
                        ordering[tree, u, w] = (tree_left, u, curMK[0], tree_right, w, curMK[1])
                        ordering[tree, w, u] = (tree_right, w, curMK[1], tree_left, u, curMK[0])
                    else:
                        def MFunc((m, k)):
                            return M[tree_left, u, m] + M[tree_right, w, k] + matrix[m, k]
                        m, k = min([(m, k) for m in other(u, Vll, Vlr) for k in other(w, Vrl, Vrr)], key=MFunc)
                        M[tree, u, w] = M[tree, w, u] = MFunc((m, k))
                        ordering[tree, u, w] = (tree_left, u, m, tree_right, w, k)
                        ordering[tree, w, u] = (tree_right, w, k, tree_left, u, m)

            if progressCallback:
                progressCallback(100.0 * len(visitedClusters) / len(tree.mapping))
                visitedClusters.add(tree)
        
    _optOrdering(tree)

    def _order(tree, u, w):
        if len(tree)==1:
            return
        left, u, m, right, w, k = ordering[tree, u, w]
        if len(left)>1 and m not in left.right:
            left.swap()
        _order(left, u, m)
##        if u!=left[0] or m!=left[-1]:
##            print "error 4:", u, m, list(left)
        if len(right)>1 and k not in right.left:
            right.swap()
        _order(right, k, w)
##        if k!=right[0] or w!=right[-1]:
##            print "error 5:", k, w, list(right)
    
    u, w = min([(u, w) for u in tree.left for w in tree.right], key=lambda (u, w): M[tree, u, w])
    
##    print "M(v) =", M[tree, u, w]
    
    _order(tree, u, w)

    def _check(tree, u, w):
        if len(tree)==1:
            return
        left, u, m, right, w, k = ordering[tree, u, w]
        if tree[0] == u and tree[-1] == w:
            _check(left, u, m)
            _check(right, k, w)
        else:
            print "Error:", u, w, tree[0], tree[-1]

    _check(tree, u ,w)
    

    if objects:
        tree.mapping.setattr("objects", objects)

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
except (ImportError, IOError), ex:
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
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(self.fig)
            canvas.print_figure(filename)
        if show:
            self.plt.show()
        
        
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
        heatmap_cell_width = heatmap_width / len(self.data.domain.attributes)
        
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
    import os
    from orngMisc import PILRenderer, EPSRenderer, SVGRenderer
    name, ext = os.path.splitext(filename)
    kwargs["renderer"] = {".eps":EPSRenderer, ".svg":SVGRenderer, ".png":PILRenderer}.get(ext.lower(), PILRenderer)
#    print kwargs["renderer"], ext
    d = DendrogramPlot(*args, **kwargs)
    d.plot(filename)
    
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

