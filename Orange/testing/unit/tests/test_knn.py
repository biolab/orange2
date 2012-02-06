import unittest
import Orange
from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_data
from Orange.classification import knn
from Orange.distance.instances import EuclideanConstructor


@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS + \
                         testing.REGRESSION_DATASETS)
class TestKNNLearner(testing.LearnerTestCase):
    def setUp(self):
        self.learner = knn.kNNLearner(distance_constructor=EuclideanConstructor())
    
    @testing.test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        instance = dataset.randomexample()
        self.assertEqual(len(self.classifier.find_nearest(3, instance)), 3)
        
    
    
    