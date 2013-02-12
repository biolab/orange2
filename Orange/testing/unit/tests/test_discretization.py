try:
    import unittest2 as unittest
except ImportError:
    import unittest

import Orange
from Orange.data.discretization import DiscretizeTable
from Orange.feature.discretization import EqualFreq, EqualWidth, Entropy


class TestDataDiscretization(unittest.TestCase):
    def test_data_discretization(self):
        iris = Orange.data.Table("iris")
        disc_iris = DiscretizeTable(iris, method=EqualFreq(n=3))

        self.assertTrue(all(len(feature.values) == 3
                            for feature in disc_iris.domain.features))

        # Should not touch the class
        self.assertIs(disc_iris.domain.class_var, iris.domain.class_var)

        iris_no_class = Orange.data.Table(
            Orange.data.Domain(iris.domain.features, None),
            iris
        )

        disc_iris = DiscretizeTable(iris_no_class, method=EqualFreq(n=3),
                                    discretize_class=True)

        self.assertTrue(all(len(feature.values) == 3
                            for feature in disc_iris.domain.features))

        self.assertIs(disc_iris.domain.class_var, None)

        housing = Orange.data.Table(Orange.data.Table("housing.tab"))
        disc_housing = DiscretizeTable(housing, method=EqualWidth(n=3),
                                       discretize_class=True)

        self.assertTrue(all(len(feature.values) == 3
                            for feature in disc_housing.domain.variables))

        heart = Orange.data.Table("heart_disease.tab")
        heart_disc = DiscretizeTable(heart, method=Entropy(), clean=True)

        self.assertTrue(all(len(feature.values) > 1
                            for feature in heart_disc.domain.features))
