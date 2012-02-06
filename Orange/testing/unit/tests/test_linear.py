import Orange
from Orange.misc import testing
from Orange.misc.testing import datasets_driven
from Orange.classification.svm import LinearLearner
import unittest


@testing.test_on_data
def test_learner_on(self, dataset):
    testing.LearnerTestCase.test_learner_on(self, dataset)
    n_vals = len(dataset.domain.class_var.values)
    if n_vals > 2:
        self.assertEquals(len(self.classifier.weights), n_vals)
    else:
        self.assertEquals(len(self.classifier.weights), 1)
        
    n_features = len(dataset.domain.attributes)
    self.assert_(all(len(w) == n_features for w in self.classifier.weights))
        
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearLearnerL2R_L2LOSS_DUAL(testing.LearnerTestCase):
    LEARNER = LinearLearner(sover_type=LinearLearner.L2R_L2LOSS_DUAL)
    
    test_learner_on=test_learner_on
    
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearLearnerL2R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearLearner(sover_type=LinearLearner.L2R_L2LOSS)
    test_learner_on=test_learner_on
    
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearLearnerL2R_L1LOSS_DUAL(testing.LearnerTestCase):
    LEARNER = LinearLearner(sover_type=LinearLearner.L2R_L1LOSS_DUAL)
    test_learner_on=test_learner_on
    
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearLearnerL2R_L1LOSS(testing.LearnerTestCase):
    LEARNER = LinearLearner(sover_type=LinearLearner.L2R_L2LOSS)
    test_learner_on=test_learner_on
    
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearLearnerL1R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearLearner(sover_type=LinearLearner.L1R_L2LOSS)
    test_learner_on=test_learner_on
    
@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearLearnerL1R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearLearner(sover_type=LinearLearner.MCSVM_CS)
    test_learner_on=test_learner_on

if __name__ == "__main__":
    unittest.main()