from Orange.misc import testing
try:
    import unittest2 as unittest
except:
    import unittest
import orange

@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestC45(testing.LearnerTestCase):
    def setUp(self):
        self.learner = orange.C45Learner

    @testing.test_on_data
    def test_learner_on(self, dataset):
        try:
            orange.C45Learner()
        except orange.KernelException:
            raise unittest.SkipTest("C45 dll not found")
        testing.LearnerTestCase.test_learner_on(self, dataset)

    @testing.test_on_data
    def test_pickling_on(self, dataset):
        try:
            orange.C45Learner()
        except orange.KernelException:
            raise unittest.SkipTest("C45 dll not found")
        testing.LearnerTestCase.test_pickling_on(self, dataset)
