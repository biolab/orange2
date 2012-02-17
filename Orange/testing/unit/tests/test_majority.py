from Orange.misc import testing
from Orange.misc.testing import datasets_driven, test_on_data
from Orange.classification import majority
from Orange.statistics import distribution
import Orange
try:
    import unittest2 as unittest
except:
    import unittest

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS + \
                 testing.REGRESSION_DATASETS)
class TestMajorityLearner(testing.LearnerTestCase):
    LEARNER = majority.MajorityLearner()

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestMajorityLearnerWithMEstimator(testing.LearnerTestCase):
    LEARNER = majority.MajorityLearner(estimator_constructor=\
                    Orange.core.ProbabilityEstimatorConstructor_m(m=3))

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS)
class TestMajorityLearnerWithLaplaceEstimator(testing.LearnerTestCase):
    LEARNER = majority.MajorityLearner(estimator_constructor=\
                    Orange.core.ProbabilityEstimatorConstructor_Laplace())

@datasets_driven(datasets=testing.REGRESSION_DATASETS)
class TestMajorityLearnerWithKernelEstimator(testing.LearnerTestCase):
    LEARNER = majority.MajorityLearner(estimator_constructor=\
                    Orange.core.ProbabilityEstimatorConstructor_kernel())

@datasets_driven(datasets=testing.REGRESSION_DATASETS)
class TestMajorityLearnerWithLoessEstimator(testing.LearnerTestCase):
    LEARNER = majority.MajorityLearner(estimator_constructor=\
                    Orange.core.ProbabilityEstimatorConstructor_loess())

#@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS + \
#                 testing.REGRESSION_DATASETS)
#class TestMajorityLearner(testing.LearnerTestCase):
#    LEARNER = majority.MajorityLearner()


if __name__ == "__main__":
    unittest.main()
