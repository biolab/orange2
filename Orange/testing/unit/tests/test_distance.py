import unittest
from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_data

from Orange.distance import instances

@datasets_driven
class TestEuclideanDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = instances.EuclideanConstructor()

@datasets_driven    
class TestMannhatanDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = instances.ManhattanConstructor()
    
@datasets_driven
class TestHammingDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = instances.HammingConstructor()
    
@datasets_driven
class TestReliefDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = instances.ReliefConstructor()

@datasets_driven
class TestPearsonRDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = instances.PearsonRConstructor()

@datasets_driven
class TestSpearmanRDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = instances.SpearmanRConstructor()
    
@datasets_driven
class TestPearsonRAbsoluteDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = instances.PearsonRAbsoluteConstructor()
    
@datasets_driven
class TestSpearmanRAbsoluteDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = instances.SpearmanRAbsoluteConstructor()
    
@datasets_driven
class TestMahalanobisDistance(testing.DistanceTestCase):
    DISTANCE_CONSTRUCTOR = instances.MahalanobisConstructor()
    
if __name__ == "__main__":
    unittest.main()