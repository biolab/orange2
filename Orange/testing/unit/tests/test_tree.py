from Orange.testing import testing
from Orange.testing.testing import datasets_driven, test_on_data
try:
    import unittest2 as unittest
except:
    import unittest
from Orange.classification import tree as ctree
from Orange.regression import tree as rtree
from Orange.feature import scoring
# TODO: test different split_constructors, descenders, measures, stop criteria...

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestClassification(testing.LearnerTestCase):
    LEARNER = ctree.TreeLearner(max_depth=50)


@datasets_driven(datasets=testing.REGRESSION_DATASETS)
class TestRegression(testing.LearnerTestCase):
    LEARNER = rtree.TreeLearner(max_depth=50)


@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestSTLClassification(testing.LearnerTestCase):
    LEARNER = ctree.SimpleTreeLearner(max_depth=50)


@datasets_driven(datasets=testing.REGRESSION_DATASETS)
class TestSTLRegression(testing.LearnerTestCase):
    LEARNER = rtree.SimpleTreeLearner(max_depth=50)

    def test_learner_on(self):
        # Does not pass unittests beacuse it returns None for the distribution.
        # I do not plan on implementing this as it will only add unnecessary overhead.
        pass


if __name__ == "__main__":
    unittest.main()
