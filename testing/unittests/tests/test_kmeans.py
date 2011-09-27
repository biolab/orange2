import unittest
from Orange.misc import testing
from Orange.clustering import kmeans
from Orange.clustering.kmeans import Clustering
from Orange.distance.instances import *

@testing.datasets_driven
class TestKMeans(unittest.TestCase):
    @testing.test_on_data
    def test_kmeans_on(self, table):
        km = Clustering(table, 5, maxiters=100, nstart=3)
        self.assertEqual(len(km.centroids), 5)
        self.assertEqual(max(set(km.clusters)) + 1, 5)
        self.assertEqual(len(km.clusters), len(table))
        
        self._test_score_functions(km)
    
    def _test_score_functions(self, km):
        kmeans.score_distance_to_centroids(km)
        kmeans.score_fast_silhouette(km, index=None)
        kmeans.score_silhouette(km, index=None)
        
    @testing.test_on_data
    def test_init_functs(self, table):
        distfunc = EuclideanConstructor(table)
        for k in [1, 5, 10]:
            self._test_init_func(table, k, distfunc)
        
    def _test_init_func(self, table, k, distfunc):
        centers = kmeans.init_random(table, k, distfunc)
        self.assertEqual(len(centers), k)
        self.assertEqual(centers[0].domain, table.domain)
        
        centers = kmeans.init_diversity(table, k, distfunc)
        self.assertEqual(len(centers), k)
        self.assertEqual(centers[0].domain, table.domain)
        
        centers = kmeans.init_hclustering(n=50)(table, k, distfunc)
        self.assertEqual(len(centers), k)
        self.assertEqual(centers[0].domain, table.domain)
        
    
    @unittest.expectedFailure
    def test_kmeans_fail(self):
        """ Test the reaction when centroids is larger then example table length
        """
        data = iter(testDatasets()).next()
        Clustering(data, len(data) + 1)


if __name__ == "__main__":
    unittest.main()
            