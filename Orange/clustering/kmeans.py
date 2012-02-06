"""
*******************************
K-means clustering (``kmeans``)
*******************************

.. index::
   single: clustering, kmeans
.. index:: agglomerative clustering


.. autoclass:: Orange.clustering.kmeans.Clustering
   :members:

Examples
========

The following code runs k-means clustering and prints out the cluster indexes
for the last 10 data instances (:download:`kmeans-run.py <code/kmeans-run.py>`, uses :download:`iris.tab <code/iris.tab>`):

.. literalinclude:: code/kmeans-run.py

The output of this code is::

    [1, 1, 2, 1, 1, 1, 2, 1, 1, 2]

Invoking a call-back function may be useful when tracing the progress of the clustering.
Below is a code that uses an :obj:`inner_callback` to report on the number of instances
that have changed the cluster and to report on the clustering score. For the score 
o be computed at each iteration we have to set :obj:`minscorechange`, but we can
leave it at 0 or even set it to a negative value, which allows the score to deteriorate
by some amount (:download:`kmeans-run-callback.py <code/kmeans-run-callback.py>`, uses :download:`iris.tab <code/iris.tab>`):

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

Call-back above is used for reporting of the progress, but may as well call a function that plots a selection data projection with corresponding centroid at a given step of the clustering. This is exactly what we did with the following script (:download:`kmeans-trace.py <code/kmeans-trace.py>`, uses :download:`iris.tab <code/iris.tab>`):

.. literalinclude:: code/kmeans-trace.py

Only the first four scatterplots are shown below. Colors of the data instances indicate the cluster membership. Notice that since the Iris data set includes four attributes, the closest centroid in a particular 2-dimensional projection is not necessary also the centroid of the cluster that the data point belongs to.


.. image:: files/kmeans-scatter-001.png

.. image:: files/kmeans-scatter-002.png

.. image:: files/kmeans-scatter-003.png

.. image:: files/kmeans-scatter-004.png


k-Means Utility Functions
=========================

.. automethod:: Orange.clustering.kmeans.init_random

.. automethod:: Orange.clustering.kmeans.init_diversity

.. autoclass:: Orange.clustering.kmeans.init_hclustering
   :members:

.. automethod:: Orange.clustering.kmeans.plot_silhouette

.. automethod:: Orange.clustering.kmeans.score_distance_to_centroids

.. automethod:: Orange.clustering.kmeans.score_silhouette

.. automethod:: Orange.clustering.kmeans.score_fast_silhouette

Typically, the choice of seeds has a large impact on the k-means clustering, 
with better initialization methods yielding a clustering that converges faster 
and finds more optimal centroids. The following code compares three different 
initialization methods (random, diversity-based and hierarchical clustering-based) 
in terms of how fast they converge (:download:`kmeans-cmp-init.py <code/kmeans-cmp-init.py>`, uses :download:`iris.tab <code/iris.tab>`,
:download:`housing.tab <code/housing.tab>`, :download:`vehicle.tab <code/vehicle.tab>`):

.. literalinclude:: code/kmeans-cmp-init.py

As expected, k-means converges faster with diversity and clustering-based 
initialization that with random seed selection::

               Rnd Div  HC
          iris  12   3   4
       housing  14   6   4
       vehicle  11   4   3

The following code computes the silhouette score for k=2..7 and plots a 
silhuette plot for k=3 (:download:`kmeans-silhouette.py <code/kmeans-silhouette.py>`, uses :download:`iris.tab <code/iris.tab>`):

.. literalinclude:: code/kmeans-silhouette.py

The analysis suggests that k=2 is preferred as it yields
the maximal silhouette coefficient::

    2 0.629467553352
    3 0.504318855054
    4 0.407259377854
    5 0.358628975081
    6 0.353228492088
    7 0.366357876944

.. figure:: files/kmeans-silhouette.png

   Silhouette plot for k=3.

"""

import math
import sys
import orange
import random
from Orange import statc

import Orange.clustering.hierarchical

# miscellaneous functions 

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

def score_fast_silhouette(km, index=None):
    """Same as score_silhouette, but computes an approximation and is faster.
    
    :param km: a k-means clustering object.
    :type km: :class:`KMeans`
    """

    if index == None:
        return avg([score_fast_silhouette(km, i) for i in range(len(km.data))])
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
    """ Saves a silhuette plot to filename, showing the distributions of silhouette scores in clusters. kmeans is a k-means clustering object. If fast is True use score_fast_silhouette to compute scores instead of score_silhouette.

    :param km: a k-means clustering object.
    :type km: :class:`KMeans`
    :param filename: name of output plot.
    :type filename: string
    :param fast: if True use :func:`score_fast_silhouette` to compute scores instead of :func:`score_silhouette`
    :type fast: boolean.

    """
    import matplotlib.pyplot as plt
    plt.figure()
    scoring = score_fast_silhouette if fast else score_silhouette
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

def init_random(data, k, _):
    """A function that can be used for initialization of k-means clustering returns k data instances from the data. This type of initialization is also known as Fory's initialization (Forgy, 1965; He et al., 2004).
    
    :param data: data instances.
    :type data: :class:`orange.ExampleTable`
    :param k: the number of clusters.
    :type k: integer
    """
    return data.getitems(random.sample(range(len(data)), k))

def init_diversity(data, k, distfun):
    """A function that can be used for intialization of k-means clustering. Returns a set of centroids where the first one is a data point being the farthest away from the center of the data, and consequent centroids data points of which the minimal distance to the previous set of centroids is maximal. Differs from the initialization proposed by Katsavounidis et al. (1994) only in the selection of the first centroid (where they use a data instance with the highest norm).

    :param data: data instances.
    :type data: :class:`orange.ExampleTable`
    :param k: the number of clusters.
    :type k: integer
    :param distfun: a distance function.
    :type distfun: :class:`Orange.distance.Distance`
    """
    center = data_center(data)
    # the first seed should be the farthest point from the center
    seeds = [max([(distfun(d, center), d) for d in data])[1]]
    # other seeds are added iteratively, and are data points that are farthest from the current set of seeds
    for i in range(1,k):
        seeds.append(max([(min([distfun(d, s) for s in seeds]), d) for d in data if d not in seeds])[1])
    return seeds

class init_hclustering():
    """
    A class that returns an clustering initialization function that performs
    hierarhical clustering, uses it to infer k clusters, and computes a
    list of cluster-based data centers
    """

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
        :type distfun: :class:`Orange.distance.Distance`
        """
        sample = orange.ExampleTable(random.sample(data, min(self.n, len(data))))
        root = Orange.clustering.hierarchical.clustering(sample)
        cmap = Orange.clustering.hierarchical.top_clusters(root, k)
        return [data_center(orange.ExampleTable([sample[e] for e in cl])) for cl in cmap]

#    
# k-means clustering, main implementation
#

class Clustering:
    """Implements a k-means clustering algorithm:

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

    .. attribute:: k

        Number of clusters.

    .. attribute:: data

        Instances to cluster.

    .. attribute:: centroids

        Current set of centroids. 

    .. attribute:: scoring

        Current clustering score.

    .. attribute:: iteration

        Current clustering iteration.

    .. attribute:: clusters

        A list of cluster indexes. An i-th element provides an
        index to a centroid associated with i-th data instance from the input 
        data set. 
    """

    def __init__(self, data=None, centroids=3, maxiters=None, minscorechange=None,
                 stopchanges=0, nstart=1, initialization=init_random,
                 distance=Orange.distance.Euclidean,
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
        :type distance: :class:`Orange.distance.DistanceConstructor`
        :param initialization: a function to select centroids given data instances, k and a example distance function. This module implements different approaches (:func:`init_random`, :func:`init_diversity`, :class:`init_hclustering`). 
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

