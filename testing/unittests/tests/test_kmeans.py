import unittest
from Orange.misc import testing
from Orange.clustering.kmeans import Clustering
from Orange.distance.instances import *

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
            