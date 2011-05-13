from Orange.clustering.hierarchical import (clustering,
    clustering_features,  cluster_to_list,
    top_clusters, HierarchicalClustering)
                           
from Orange.clustering.kmeans import Clustering
from Orange.distances import *
                           
import Orange.misc.testing as testing
import orange
import unittest

@testing.expand_tests
class TestHClustering(testing.BaseTestOnData):
    FLAGS = testing.TEST_ALL + testing.TEST_CLASSLESS
    
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
            
    def test_pickle(self):
        data = iter(self.datasets()).next()
        cluster = clustering(data, EuclideanConstructor, HierarchicalClustering.Single)
        import cPickle
        s = cPickle.dumps(cluster)
        cluster_clone = cPickle.loads(s)
        
        self.assertEqual(cluster.mapping, cluster_clone.mapping)


@testing.expand_tests
class TestKMeans(testing.BaseTestOnData):
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
        
            