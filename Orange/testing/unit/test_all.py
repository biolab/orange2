import os.path
try:
    import unittest2 as unittest
except:
    import unittest


def load_tests(loader, standard_tests, pattern):
    # top level directory cached on loader instance
    this_dir = os.path.join(os.path.dirname(__file__), "tests")
    package_tests = loader.discover(start_dir=this_dir, pattern="test*.py")
    standard_tests.addTests(package_tests)
    return standard_tests
