import cPickle

import Orange
from Orange.testing import testing
from Orange.testing.testing import datasets_driven
from Orange.classification.svm import LinearSVMLearner
try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np


def decision_values(classifier, instance):
    """Return the decision values (numpy.array) for classifying `instance`.
    """
    instance = Orange.data.Table(classifier.domain, [instance])
    (instance,) = instance.to_numpy_MA("A")

    x = instance.filled(0.0)
    if classifier.bias > 0.0:
        x = np.hstack([x, [[classifier.bias]]])

    w = np.array(classifier.weights)

    return np.dot(x, w.T).ravel()


def classify_from_weights(classifier, instance):
    """Classify the instance using classifier's weights.
    """
    dec_values = decision_values(classifier, instance)

    class_var = classifier.class_var
    if len(class_var.values) > 2:
        # TODO: Check how liblinear handles ties
        return class_var(int(np.argmax(dec_values)))
    else:
        return class_var(0 if dec_values[0] > 0 else 1)


def classify_from_weights_test(self, classifier, data):
    class_var = data.domain.class_var
    if isinstance(class_var, Orange.feature.Discrete):
        for inst in data[:]:
            pval1 = classifier(inst)
            pval2 = classify_from_weights(classifier, inst)
            self.assertEqual(pval1, pval2,
                             msg="classifier and classify_from_weights return "
                                 "different values")


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

    classify_from_weights_test(self, self.classifier, dataset)


@testing.test_on_data
def test_learner_with_bias_on(self, dataset):
    learner = self.learner
    learner_b = cPickle.loads(cPickle.dumps(learner))
    learner_b.bias = 1
    try:
        self.learner = learner_b
        test_learner_on(self, dataset)
    finally:
        self.learner = learner


def split(data, value):
    pos = [inst for inst in data if inst.get_class() == value]
    neg = [inst for inst in data if inst.get_class() != value]
    return Orange.data.Table(pos), Orange.data.Table(neg)


def missing_instances_test(self):
    """Test the learner on a dataset with no instances for
    some class.

    """
    data = Orange.data.Table("iris")
    class_var = data.domain.class_var

    for i, value in enumerate(class_var.values):
        _, train = split(data, value)
        classifier = self.learner(train)

        self.assertEqual(len(classifier.weights), len(class_var.values),
                        msg="Number of weight vectors differs from the number "
                            "of class values")

        dec_values = [decision_values(classifier, instance) \
                      for instance in data]

        self.assertTrue(all(val[i] == 0.0 for val in dec_values),
                        msg="Non zero decision value for unseen class")

        classify_from_weights_test(self, classifier, data)


@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L2LOSS_DUAL(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L2LOSS_DUAL)

    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on
    test_missing_instances = missing_instances_test


@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L2LOSS)

    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on
    test_missing_instances = missing_instances_test


@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L1LOSS_DUAL(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L1LOSS_DUAL)

    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on
    test_missing_instances = missing_instances_test


@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL2R_L1LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L2R_L2LOSS)

    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on
    test_missing_instances = missing_instances_test


@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerL1R_L2LOSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.L1R_L2LOSS)

    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on
    test_missing_instances = missing_instances_test


@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestLinearSVMLearnerMCSVM_CSS(testing.LearnerTestCase):
    LEARNER = LinearSVMLearner(sover_type=LinearSVMLearner.MCSVM_CS)

    test_learner_on = test_learner_on
    test_learner_with_bias_on = test_learner_with_bias_on
    test_missing_instances = missing_instances_test


if __name__ == "__main__":
    unittest.main()
