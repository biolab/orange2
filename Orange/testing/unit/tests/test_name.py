import orange
import Orange
import unittest

class TestName(unittest.TestCase):
    def test_Learner(self):
        b = orange.BayesLearner()
        self.assertEqual(b.name, "bayes")
        b.name = "foo"
        self.assertEqual(b.name, "foo")
        b.name = "BayesLearner"
        self.assertEqual(b.name, "BayesLearner")
        b.name = "x.BayesLearner"
        self.assertEqual(b.name, "x.BayesLearner")
        b.name = ""
        self.assertEqual(b.name, "")

    def test_class(self):
        class MyBla(orange.BayesLearner):
            pass
        b = MyBla()
        self.assertEqual(b.name, "myBla")
        b.name = "foo"
        self.assertEqual(b.name, "foo")

    def test_classLearner(self):
        class MyBlaLearner(orange.BayesLearner):
            pass
        b = MyBlaLearner()
        self.assertEqual(b.name, "myBla")

    def test_class_short(self):
        class A(orange.BayesLearner):
            pass
        b = A()
        self.assertEqual(b.name, "a")
        b.name = "foo"
        self.assertEqual(b.name, "foo")

    def test_Discretizer(self):
        b = orange.EquiDistDiscretizer()
        # The class is renamed internally
        # "Discretizer" is removed and E is changed to e
        self.assertEqual(b.name, "equalWidth")

    def test_Classifier(self):
        b = orange.TreeClassifier()
        self.assertEqual(b.name, "tree")

    def test_Orange(self):
        b = Orange.classification.bayes.NaiveLearner()
        self.assertEqual(b.name, "naive")

    def test_static_name(self):
        # Tests that class attributes work and are left
        # (stripping off 'Learner' and lower cases are
        # applied only to class names
        class NaiveLearner(orange.BayesLearner):
            name = "BayesLearner"
        b = NaiveLearner()
        self.assertEqual(b.name, "BayesLearner")


if __name__ == "__main__":
    unittest.main()