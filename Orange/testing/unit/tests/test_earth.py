import Orange
from Orange.testing import testing
from Orange.testing.testing import datasets_driven, test_on_data
from Orange.regression import earth
import numpy

try:
    import unittest2 as unittest
except:
    import unittest


@datasets_driven(datasets=testing.REGRESSION_DATASETS + \
                 testing.CLASSIFICATION_DATASETS)
class TestEarthLearner(testing.LearnerTestCase):

    def setUp(self):
        self.learner = earth.EarthLearner(degree=2, terms=10)

    @test_on_data
    def test_learner_on(self, dataset):
        if len(dataset) < 30:
            raise unittest.SkipTest("Not enough examples.")
        testing.LearnerTestCase.test_learner_on(self, dataset)
        str = self.classifier.to_string()
        evimp = self.classifier.evimp()

        # Test base_features (make sure the domain translation works)
        basis_features = self.classifier.base_features()
        basis_domain = Orange.data.Domain(basis_features, None)
        basis_table = Orange.data.Table(basis_domain, dataset)
        basis_matrix = self.classifier.base_matrix(dataset)
        # Filter best set
        basis_matrix = basis_matrix[:, self.classifier.best_set]
        # Remove intercept
        basis_matrix = basis_matrix[:, 1:]
        basis_matrix_a = basis_table.to_numpy_MA("A")[0]
        # Fill unknowns
        basis_matrix[basis_matrix_a.mask] = 0
        basis_matrix_a = basis_matrix_a.filled(0)
        diff = basis_matrix - basis_matrix_a
        self.assertAlmostEqual(numpy.max(diff), 0, places=3)

    @test_on_data
    def test_bagged_evimp(self, dataset):
        from Orange.ensemble.bagging import BaggedLearner
        bagged_learner = BaggedLearner(earth.EarthLearner(terms=10, degree=2),
                                       t=5)

        bagged_classifier = bagged_learner(dataset)
        evimp = earth.bagged_evimp(bagged_classifier, used_only=False)


@datasets_driven(datasets=testing.REGRESSION_DATASETS + \
                 testing.CLASSIFICATION_DATASETS)
class TestScoreEarthImportance(testing.MeasureAttributeTestCase):
    def setUp(self):
        from Orange.regression.earth import ScoreEarthImportance
        self.measure = ScoreEarthImportance(t=5, score_what="rss")


@datasets_driven(datasets=["multitarget-synthetic"])
class TestEarthMultitarget(unittest.TestCase):
    @test_on_data
    def test_multi_target_on_data(self, dataset):
        self.learner = earth.EarthLearner(degree=2, terms=10)

        self.predictor = self.multi_target_test(self.learner, dataset)

        self.assertTrue(bool(self.predictor.multitarget))

        s = str(self.predictor)
        self.assertEqual(s, self.predictor.to_string())
        self.assertNotEqual(s, self.predictor.to_string(3, 6))

    def multi_target_test(self, learner, data):
        indices = Orange.data.sample.SubsetIndices2(p0=0.3)(data)
        learn = data.select(indices, 1)
        test = data.select(indices, 0)

        predictor = learner(learn)
        self.assertIsInstance(predictor, Orange.classification.Classifier)
        self.multi_target_predictor_interface(predictor, learn.domain)

        from Orange.evaluation import testing as _testing

        r = _testing.test_on_data([predictor], test)

        def all_values(vals):
            for v in vals:
                self.assertIsInstance(v, Orange.core.Value)

        def all_dists(dist):
            for d in dist:
                self.assertIsInstance(d, Orange.core.Distribution)

        for ex in test:
            preds = predictor(ex, Orange.core.GetValue)
            all_values(preds)

            dist = predictor(ex, Orange.core.GetProbabilities)
            all_dists(dist)

            preds, dist = predictor(ex, Orange.core.GetBoth)
            all_values(preds)
            all_dists(dist)

            for d in dist:
                if isinstance(d, Orange.core.ContDistribution):
                    dist_sum = sum(d.values())
                else:
                    dist_sum = sum(d)

                self.assertGreater(dist_sum, 0.0)
                self.assertLess(abs(dist_sum - 1.0), 1e-3)

        return predictor

    def multi_target_predictor_interface(self, predictor, domain):
        self.assertTrue(hasattr(predictor, "class_vars"))
        self.assertIsInstance(predictor.class_vars, (list, Orange.core.VarList))
        self.assertTrue(all(c1 == c2 for c1, c2 in \
                            zip(predictor.class_vars, domain.class_vars)))


def load_tests(loader, tests, ignore):
    import doctest
    tests.addTests(doctest.DocTestSuite(earth))
    return tests


if __name__ == "__main__":
    unittest.main()

