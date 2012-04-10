from Orange.testing import testing
from Orange.testing.testing import datasets_driven, test_on_data, test_on_datasets
from Orange.regression import linear
import Orange
try:
    import unittest2 as unittest
except:
    import unittest

@datasets_driven(datasets=testing.REGRESSION_DATASETS)
class TestLinearRegressionLearner(testing.LearnerTestCase):

    def setUp(self):
        self.learner = linear.LinearRegressionLearner()

    @test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        self.assertTrue(isinstance(self.classifier.to_string(), str))

@datasets_driven(datasets=testing.REGRESSION_DATASETS)
class TestRidgeRegressionLearner(testing.LearnerTestCase):

    def setUp(self):
        self.learner = linear.LinearRegressionLearner(ridge_lambda=2)

    @test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        self.assertTrue(isinstance(self.classifier.to_string(), str))


if __name__ == "__main__":
    unittest.main()
