try:
    import unittest2 as unittest
except ImportError:
    import unittest

import Orange
from Orange import utils


class TestUtils(unittest.TestCase):
    def test_new_wrappers(self):
        class A(Orange.core.OrangeBase):
            __new__ = utils._orange__new__(Orange.core.OrangeBase)

            def __call__(self, data):
                return data, self.name, self.msg

        msg = "This is a test"
        name = "A"
        a = A(name=name, msg=msg)
        self.assertIsInstance(a, A)
        self.assertEqual(a.name, name)
        self.assertEqual(a.msg, msg)

        iris = Orange.data.Table("iris")
        b, name1, msg1 = A(iris, name=name, msg=msg)
        self.assertIs(b, iris)
        self.assertEqual(name1, name)
        self.assertEqual(msg1, msg)

        class L(Orange.core.Learner):
            __new__ = utils._orange__new__(Orange.core.Learner)

            def __call__(self, data, weight=0):
                return data, weight, self.msg

        a = L(msg=msg)
        self.assertIsInstance(a, L)
        self.assertEqual(a.msg, msg)

        b, weight, msg1 = L(iris, 2, msg=msg)
        self.assertIs(b, iris)
        self.assertEqual(weight, 2)
        self.assertEqual(msg1, msg)

if __name__ == "__main__":
    unittest.main()
