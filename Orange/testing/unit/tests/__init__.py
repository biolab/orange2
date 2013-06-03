import os
import unittest
if not hasattr(unittest.TestLoader, 'discover'):
    import unittest2 as unittest


def suite():
    test_dir = os.path.dirname(__file__)
    return unittest.TestLoader().discover(test_dir, )

test_suite = suite()


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
