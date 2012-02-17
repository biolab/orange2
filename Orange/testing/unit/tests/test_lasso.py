from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_data, test_on_datasets
from Orange.regression import lasso
import Orange
try:
    import unittest2 as unittest
except:
    import unittest

@datasets_driven(datasets=testing.REGRESSION_DATASETS)
class TestLassoRegressionLearner(testing.LearnerTestCase):

    def setUp(self):
        self.learner = lasso.LassoRegressionLearner(n_boot=2, n_perm=2)

    @test_on_data
    def test_learner_on(self, dataset):
        testing.LearnerTestCase.test_learner_on(self, dataset)
        self.assertTrue(isinstance(self.classifier.to_string(), str))

if __name__ == "__main__":
    unittest.main()
