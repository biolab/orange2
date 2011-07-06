from Orange.misc import testing
import unittest
import orange

@testing.datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestC45(testing.LearnerTestCase):
    def setUp(self):
        self.learner = orange.C45Learner
       