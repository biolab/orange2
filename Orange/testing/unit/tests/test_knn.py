try:
    import unittest2 as unittest
except:
    import unittest
import Orange
from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_data
from Orange.classification import knn
from Orange.distance import Euclidean


@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS + \
                         testing.REGRESSION_DATASETS)
class TestKNNLearner(testing.LearnerTestCase):
    def setUp(self):
        self.learner = knn.kNNLearner(distance_constructor=Euclidean())

    @testing.test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        instance = dataset.random_instance()
        self.assertEqual(len(self.classifier.find_nearest(3, instance)), 3)

if __name__ == "__main__":
    unittest.main()