"""
************************************
Consensus clustering (``consensus``)
************************************

.. note:: This module is still under
  development and its interface may change at any time.

.. autofunction:: consensus_matrix

"""

import Orange.feature
import numpy

def consensus_matrix(data, n, subset_fn, cluster_fn):
    """
    Compute a consensus matrix.

    :param data: data set to work on

    :param n: number of different clusterings.

    :param subset_fn: a function that takes a data table and returns a new
      data table with a subset of instances or attributes (or both). Subset
      function needs to preserve meta attribute "index".

    :param cluster_fn: a function that takes a data set and returns a
      list of examples in the clusters.
    """
    # add indices to features
    mid = Orange.feature.Descriptor.new_meta_id()
    var = Orange.feature.String(name="index")
    data.domain.add_meta(mid, var)
    for i, ex in enumerate(data):
        ex[mid] = str(i)

    appear = numpy.zeros([len(data), len(data)]) #in the data
    cluster = appear.copy() #in the cluster 

    def addones(exlist, table):
        """ +1 if two elements appear together """
        for a in exlist:
            for b in exlist:
                table[int(a[mid].value), int(b[mid].value)] += 1

    for _ in range(n):
        d = subset_fn(data)
        clusters = cluster_fn(d)
        addones(d, appear)
        for c in clusters:
            addones(c, cluster)
    
    #TODO what to do if two samples did not appear at all?
    #numpy prints a warning and sets "nan"
    print appear
    print cluster

    data.domain.remove_meta(mid)

    return cluster/appear

import random
def select_half(data):
    examples = random.sample(data, len(data)/2)
    return Orange.data.Table(data.domain, examples)

import Orange.clustering.kmeans
from collections import defaultdict
def ckmeans(data):
    clusters = Orange.clustering.kmeans.Clustering(data, 3)
    d = defaultdict(list)
    for c, ex in zip(clusters.clusters, data):
        d[c].append(ex)
    return d.values()

if __name__ == "__main__":
    data = Orange.data.Table("iris")
    print consensus_matrix(data, 100, select_half, ckmeans)
