try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy
from distutils.version import StrictVersion

from Orange.testing import testing
from Orange.classification import neural

np_version = StrictVersion(numpy.version.short_version).version


@unittest.skipIf(np_version < (1, 5), "numpy >= 1.5 required")
@testing.datasets_driven(datasets=["iris", "lenses", "monks-1"])
class TestNeuralNetwork(testing.LearnerTestCase):
    LEARNER = neural.NeuralNetworkLearner
