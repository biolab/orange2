from Orange.misc import testing
from Orange.misc.testing import data_driven, datasets_driven, test_on_data
try:
    import unittest2 as unittest
except:
    import unittest

data = [("one", (1,)),
        ("two", (2,))]

# Data driven with data_iter arg
@data_driven(data_iter=data)
class TestDemo(unittest.TestCase):
    @testing.test_on_data
    def test_instance_on(self, arg):
        print arg
        self.assertIsInstance(arg, int)

    @testing.test_on_data
    def test_add(self, arg):
        print arg
        res = arg + arg

# data_driven without arg
@data_driven
class TestDemo1(unittest.TestCase):
    @test_on_data(data_iter=data)
    def test_instance_on(self, arg):
        self.assertIsInstance(arg, int)

    @test_on_data(data_iter=data)
    def test_add(self, arg):
        res = arg + arg

# data_driven without arg, using a static data_iter method
@data_driven
class TestDemo2(unittest.TestCase):
    @test_on_data
    def test_instance_on(self, arg):
        self.assertIsInstance(arg, int)

    @test_on_data
    def test_add(self, arg):
        res = arg + arg

    @staticmethod
    def data_iter():
        return data

@datasets_driven(datasets=testing.CLASSIFICATION_DATASETS + \
                 testing.CLASSLES_DATASETS)
class TestDefaultLearner(unittest.TestCase):
    @test_on_data(data_iter=testing.datasets_iter(testing.CLASSIFICATION_DATASETS))
    def test_learner_on(self, dataset):
        import Orange
        Orange.classification.majority.MajorityLearner(dataset)

    # this overloads the class decorator's data_iter
    @test_on_data(data_iter=testing.datasets_iter(testing.CLASSLES_DATASETS))
    def test_raise_missing_class_on(self, dataset):
        import Orange
        self.assertRaises(Exception, Orange.classification.majority.MajorityLearner, dataset)

if __name__ == "__main__":
    unittest.main()


