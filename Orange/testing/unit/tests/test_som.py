from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_data
try:
    import unittest2 as unittest
except:
    import unittest

from Orange.projection import som

def test_som_projection_helper(self, map, data):
    pass

@datasets_driven
class TestSOM(testing.LearnerTestCase):
    def setUp(self):
        self.learner = som.SOMLearner

    @test_on_data
    def test_learner_on(self, dataset):
        if dataset.domain.class_var:
            # Test the learner/classification interface
            testing.LearnerTestCase.test_learner_on(self, dataset)
        else:
            self.classifier = self.learner(dataset)
        test_som_projection_helper(self, self.classifier, dataset)

    @test_on_data
    def test_pickling_on(self, dataset):
        if dataset.domain.class_var:
            testing.LearnerTestCase.test_pickling_on(self, dataset)


@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS + \
                 testing.REGRESSION_DATASETS)
class TestSOMSupervised(testing.LearnerTestCase):
    def setUp(self):
        self.learner = som.SOMSupervisedLearner()

    @test_on_data
    def test_learner_on(self, dataset):
        if dataset.domain.class_var:
            # Test the learner/classification interface
            testing.LearnerTestCase.test_learner_on(self, dataset)
        else:
            self.classifier = self.learner(dataset)
        test_som_projection_helper(self, self.classifier, dataset)

    @test_on_data
    def test_pickling_on(self, dataset):
        if dataset.domain.class_var:
            testing.LearnerTestCase.test_pickling_on(self, dataset)



if __name__ == "__main__":
    unittest.main()
