from Orange.misc import testing
import unittest
import orange

@testing.expand_tests
class TestC45(testing.LearnerTestCase):
    LEARNER = orange.C45Learner
       