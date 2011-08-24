from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_data
import unittest
from Orange.classification import tree as ctree
from Orange.regression import tree as rtree
from Orange.feature import scoring
# TODO: test different split_constructors, descenders, measures, stop criteria...

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestClassification(testing.LearnerTestCase):
    LEARNER = ctree.TreeLearner()


@datasets_driven(datasets=testing.REGRESSION_DATASETS)    
class TestRegression(testing.LearnerTestCase):
    LEARNER = rtree.TreeLearner()
        
if __name__ == "__main__":
    unittest.main()