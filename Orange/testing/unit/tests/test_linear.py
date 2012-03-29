import Orange
from Orange.testing import testing
from Orange.testing.testing import datasets_driven
from Orange.classification.svm import LinearSVMLearner
try:
    import unittest2 as unittest
except:
    import unittest


@testing.test_on_data
def test_learner_on(self, dataset):
    testing.LearnerTestCase.test_learner_on(self, dataset)
    n_vals = len(dataset.domain.class_var.values)
    if n_vals > 2:
        self.assertEquals(len(self.classifier.weights), n_vals)
    else:
        self.assertEquals(len(self.classifier.weights), 1)

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L2LOSS_DUAL(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L2LOSS_DUAL)

    test_learner_on = test_learner_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L2LOSS)
    test_learner_on = test_learner_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L1LOSS_DUAL(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L1LOSS_DUAL)
    test_learner_on = test_learner_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L1LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L2LOSS)
    test_learner_on = test_learner_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL1R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L1R_L2LOSS)
    test_learner_on = test_learner_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL1R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.MCSVM_CS)
    test_learner_on = test_learner_on

if __name__ == "__main__":
    unittest.main()
