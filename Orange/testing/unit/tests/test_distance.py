import unittest
from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_data

from Orange.distance import *

@datasets_driven
class TestEuclideanDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = Euclidean()

@datasets_driven    
class TestMannhatanDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = Manhattan()
    
@datasets_driven
class TestHammingDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = Hamming()
    
@datasets_driven
class TestReliefDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = Relief()

@datasets_driven
class TestPearsonRDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = PearsonR()

@datasets_driven
class TestSpearmanRDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = SpearmanR()
    
@datasets_driven
class TestPearsonRAbsoluteDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = PearsonRAbsolute()
    
@datasets_driven
class TestSpearmanRAbsoluteDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = SpearmanRAbsolute()
    
@datasets_driven
class TestMahalanobisDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = Mahalanobis()
    
if __name__ == "__main__":
    unittest.main()
