import Orange
from Orange.testing import testing
from Orange.testing.testing import datasets_driven
from Orange.classification.svm import LinearSVMLearner
try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np

def multiclass_from_1_vs_rest(dec_values, class_var):
    if len(class_var.values) > 2:
        return class_var(int(np.argmax(dec_values)))
    else:
        return class_var(0 if dec_values[0] > 0 else 1)

def binary_classifier_test(self, data):
    class_var = data.domain.class_var
    if isinstance(class_var, Orange.feature.Discrete):
        cl_values = class_var.values
        if self.classifier.bias >= 0:
            bias = [self.classifier.bias]
        else:
            bias = []
        for inst in data[:]:
            dec_values = []
            inst_v = [float(v) if not v.is_special() else 0.0 \
                      for v in Orange.data.Instance(self.classifier.domain, inst)]
            inst_v = inst_v[:-1] + bias
            for w in self.classifier.weights:
                dec_values.append(np.dot(inst_v, w))
            pval1 = self.classifier(inst)
            pval2 = multiclass_from_1_vs_rest(dec_values, class_var)
            if len(cl_values) > 2:
                self.assertEqual(pval1, pval2)
            else:
                #TODO: handle order switch
                pass

@testing.test_on_data
def test_learner_on(self, dataset):
    testing.LearnerTestCase.test_learner_on(self, dataset)
    n_vals = len(dataset.domain.class_var.values)
    if n_vals > 2:
        self.assertEquals(len(self.classifier.weights), n_vals)
    else:
        self.assertEquals(len(self.classifier.weights), 1)
    n_features = len(self.classifier.domain.attributes)
    if self.classifier.bias >= 0:
        n_features += 1
    
    self.assertTrue(all(len(w) == n_features \
                        for w in self.classifier.weights
                        ))
    
    binary_classifier_test(self, dataset)

@testing.test_on_data
def test_learner_with_bias_on(self, dataset):
    import cPickle
    learner = self.learner
    learner_b = cPickle.loads(cPickle.dumps(learner))
    learner_b.bias = 1
    try:
        self.learner = learner_b
    finally:
        self.learner = learner
    test_learner_on(self, dataset)
         

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L2LOSS_DUAL(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L2LOSS_DUAL)

    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L2LOSS)
    
    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L1LOSS_DUAL(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L1LOSS_DUAL)
    
    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L1LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L2LOSS)
        
    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL1R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L1R_L2LOSS)
    
    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL1R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.MCSVM_CS)
    
    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on

if __name__ == "__main__":
    unittest.main()
