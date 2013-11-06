try:
    import unittest2 as unittest
except ImportError:
    import unittest

from Orange.testing import testing
from Orange.classification import neural


@testing.datasets_driven(datasets=["iris", "lenses", "monks-1"])
class TestNeuralNetwork(testing.LearnerTestCase):
    LEARNER = neural.NeuralNetworkLearner
