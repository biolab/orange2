from Orange.clustering.hierarchical import (clustering,
    clustering_features,  cluster_to_list,
    top_clusters, HierarchicalClustering)
                           
from Orange.clustering.kmeans import Clustering
from Orange.distance.instances import *
                           
import Orange.misc.testing as testing
import orange
import unittest

@testing.datasets_driven
class TestHClustering(testing.DataTestCase):    
    @testing.test_on_data
    def test_example_clustering_on(self, data):
        constructors = [EuclideanConstructor, ManhattanConstructor]
        for distanceConstructor in constructors:
            clust = clustering(data, distanceConstructor, HierarchicalClustering.Single)
            clust = clustering(data, distanceConstructor, HierarchicalClustering.Average)
            clust = clustering(data, distanceConstructor, HierarchicalClustering.Complete)
            clust = clustering(data, distanceConstructor, HierarchicalClustering.Ward)
            top_clust = top_clusters(clust, 5)
            cluster_list = cluster_to_list(clust, 5)
            
    @testing.test_on_data
    def test_attribute_clustering_on(self, data):
        constructors = [EuclideanConstructor, ManhattanConstructor]
        for distanceConstructor in constructors:
            clust = clustering(data)
            cluster_list = cluster_to_list(clust, 5)
            
    @testing.test_on_datasets(datasets=["iris"])
    def test_pickling_on(self, data):
#        data = iter(self.datasets()).next()
        cluster = clustering(data, EuclideanConstructor, HierarchicalClustering.Single)
        import cPickle
        s = cPickle.dumps(cluster)
        cluster_clone = cPickle.loads(s)
        self.assertEqual(cluster.mapping, cluster_clone.mapping)
        
from Orange.clustering import hierarchical as hier
import Orange

class TestHClusteringUtility(unittest.TestCase):
    def setUp(self):
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
        self.matrix = Orange.core.SymMatrix(m)
        self.matrix.setattr("objects", ["Ann", "Bob", "Curt", "Danny", "Eve", "Fred", "Greg", "Hue", "Ivy", "Jon"])
        self.cluster = hier.HierarchicalClustering(self.matrix)
        
    def test_clone(self):
        cloned_cluster = hier.clone(self.cluster)
        self.assertTrue(self.cluster.mapping.objects is cloned_cluster.mapping.objects)
        self.assertEqual(self.cluster.mapping, cloned_cluster.mapping)
        
    def test_order(self):
        post = hier.postorder(self.cluster)
        pre = hier.preorder(self.cluster)
        
    def test_prunning(self):
        pruned1 = hier.pruned(self.cluster, level=2)
        pruned2 = hier.pruned(self.cluster, height=10)
        pruned3 = hier.pruned(self.cluster, condition=lambda cl: len(cl) <= 3)
        

@testing.datasets_driven
class TestKMeans(unittest.TestCase):
    @testing.test_on_data
    def test_kmeans_on(self, data):
        km = Clustering(data, 5, maxiters=100, nstart=3)
    
    
    @unittest.expectedFailure
    def test_kmeans_fail(self):
        """ Test the reaction when centroids is larger then example table length
        """
        data = iter(testDatasets()).next()
        Clustering(data, len(data) + 1)
        
        
if __name__ == "__main__":
    unittest.main()
        
            