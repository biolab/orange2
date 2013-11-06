try:
    import unittest2 as unittest
except:
    import unittest


from Orange.testing import testing
from Orange.regression import pls


@testing.datasets_driven(datasets=testing.REGRESSION_DATASETS)
class TestPLS(testing.LearnerTestCase):
    LEARNER = pls.PLSRegressionLearner


# TODO: Test the PLS by passing x_vars, y_vars
