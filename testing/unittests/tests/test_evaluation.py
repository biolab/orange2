import operator, unittest
from collections import Counter

import Orange

class DummyLearner(Orange.classification.majority.MajorityLearner):
    def __init__(self, *args, **kwds):
        self.data = []
        self.classifiers = []
        super(DummyLearner, self).__init__(*args, **kwds)

    def __call__(self, dataset, weight):
        self.data.append(dataset)
        cl = super(DummyLearner, self).__call__(dataset, weight)
        cl = DummyClassifier(cl)
        self.classifiers.append(cl)
        return cl

class DummyClassifier(object):
    def __init__(self, base_class):
        self.base_class = base_class
        self.data = []

    def __call__(self, example, options=None):
        self.data.append(example)
        return self.base_class.__call__(example, options)

class DummyPreprocessor(object):
    def __init__(self, meta_id):
        self.meta_id = meta_id

    def __call__(self, *datasets):
        # TODO: What is LT Preprocessor and how to use it?
        new_datasets = []
        for dataset in datasets:
            new_data = Orange.data.Table(dataset)
            for example in new_data:
                example[self.meta_id] = 1.
            new_datasets.append(new_data)

        return new_data if len(datasets) == 1 else new_datasets

class BrokenPreprocessor(object):
    def __call__(self, *args):
        return []

def create_examples(id):
    data = Orange.data.Table("iris")
    for i, inst in enumerate(data):
        inst[id] = i
    return data


class TestTestWithIndices(unittest.TestCase):
    def setUp(self):
        self.meta_id = Orange.data.variable.new_meta_id()
        self.learners = [DummyLearner()]
        self.examples = create_examples(self.meta_id)

        self.folds = 3
        examples_in_fold = len(self.examples) // self.folds
        self.indices = [i // examples_in_fold for i in range(len(self.examples))]

    def test_run(self):
        test_results = Orange.evaluation.testing.test_with_indices(self.learners, self.examples, self.indices)
        expected_results = [1]*50+[0]*100
        predicted_classes = map(lambda x:x.classes[0], test_results.results)
        
        self.assertItemsEqual(expected_results, predicted_classes)
        
        self.assertFalse(hasattr(test_results, "examples"))
        self.assertEquals(len(test_results.classifiers), 0)

    def test_if_called_with_right_instances(self):
        test_results = Orange.evaluation.testing.test_with_indices(self.learners, self.examples, self.indices)
        
        example_cnt = 0
        for fold, data in enumerate(self.learners[0].data):
            for example in data:
                # Classifier should be trained on examples where fold is different from current fold
                self.assertNotEqual(self.indices[int(example[self.meta_id])], fold)
                example_cnt += 1
        # Each example should be used for training (n-1)x, where n is the number of folds
        self.assertEqual(example_cnt, len(self.examples) * (self.folds-1))

        example_cnt = 0
        for fold, classifier in enumerate(self.learners[0].classifiers):
            for example in classifier.data:
                # Classifier should perform classification on examples with same fold number
                self.assertEqual(self.indices[int(example[self.meta_id])], fold)
                example_cnt += 1
        # Each example should be used for testing only once
        self.assertEqual(example_cnt, len(self.examples))

    def test_run_with_store_examples(self):
        test_results = Orange.evaluation.testing.test_with_indices(self.learners, self.examples, self.indices, store_examples=True)

        self.assertGreater(len(test_results.examples), 0)


    def test_run_with_store_classifiers(self):
        test_results = Orange.evaluation.testing.test_with_indices(self.learners, self.examples, self.indices, store_classifiers=True)

        self.assertGreater(len(test_results.classifiers), 0)
        classifiers = map(operator.itemgetter(0), test_results.classifiers)
        self.assertItemsEqual(self.learners[0].classifiers, classifiers)

    def callback(self):
        self.callback_calls += 1

    def test_run_with_callback(self):
        self.callback_calls = 0
        Orange.evaluation.testing.test_with_indices(self.learners, self.examples, self.indices, callback=self.callback)
        self.assertEqual(self.callback_calls, self.folds)

    def test_run_with_weight(self):
        examples = self.examples, 0
        Orange.evaluation.testing.test_with_indices(self.learners, examples, self.indices)

    def test_run_with_preprocessor(self):
        both = Orange.data.variable.new_meta_id()
        learn = Orange.data.variable.new_meta_id()
        test = Orange.data.variable.new_meta_id()
        learntest = Orange.data.variable.new_meta_id()
        preprocessors = (("B", DummyPreprocessor(both)),
                         ("L", DummyPreprocessor(learn)),
                         ("T", DummyPreprocessor(test)),
                         ("LT", DummyPreprocessor(learntest)))

        Orange.evaluation.testing.test_with_indices(self.learners,
                                                    self.examples,
                                                    self.indices,
                                                    preprocessors=preprocessors)

        # Original examples should be left intact
        for example in self.examples:
            self.assertFalse(example.has_meta(both))
            self.assertFalse(example.has_meta(learn))
            self.assertFalse(example.has_meta(test))
            self.assertFalse(example.has_meta(learntest))

        for fold, data in enumerate(self.learners[0].data):
            for example in data:
                # Preprocessors both, learn and learntest should be applied to learn data.
                self.assertTrue(example.has_meta(both))
                self.assertTrue(example.has_meta(learn))
                self.assertTrue(example.has_meta(learntest))
                # Preprocessor test should not be applied to learn data.
                self.assertFalse(example.has_meta(test))

        for fold, classifier in enumerate(self.learners[0].classifiers):
            for example in classifier.data:
                # Preprocessors both, test and learntest should be applied to test data.
                self.assertTrue(example.has_meta(both))
                self.assertTrue(example.has_meta(test))
                self.assertTrue(example.has_meta(learntest))
                # Preprocessor learn should not be applied to test data.
                self.assertFalse(example.has_meta(learn))

    def test_error(self):
        # No data should raise a Value error
        with self.assertRaises(ValueError):
            Orange.evaluation.testing.test_with_indices(self.learners, [], self.indices)

        # If one fold is not represented in indices, cross validation should still execute
        Orange.evaluation.testing.test_with_indices(self.learners, self.examples, [2] + [1]*(len(self.examples)-1))

        # If preprocessors is "broken" (returns no data), error should be  raised
        with self.assertRaises(SystemError):
            Orange.evaluation.testing.test_with_indices(self.learners, self.examples, self.indices, preprocessors=(("L", BrokenPreprocessor()),))
        with self.assertRaises(SystemError):
            Orange.evaluation.testing.test_with_indices(self.learners, self.examples, self.indices, preprocessors=(("T", BrokenPreprocessor()),))

class TestCrossValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def dummy_test_with_indices(*args, **kwds):
            return kwds["indices"], kwds
        Orange.evaluation.testing.test_with_indices = dummy_test_with_indices

    def setUp(self):
        self.examples = Orange.data.Table("iris")
        self.folds = 10
        self.preprocessors = ("P1", "P2", "P3", "P4")
        def callback(): pass
        self.callback = callback
        self.random_generator = 5
        self.store_classifiers = True
        self.store_examples = True

    def test_all_arguments_forwarded(self):
        _, kwargs = Orange.evaluation.testing.cross_validation([], self.examples, self.folds,
                                                                 preprocessors=self.preprocessors,
                                                                 callback=self.callback,
                                                                 store_classifiers=self.store_classifiers,
                                                                 store_examples=self.store_classifiers)
        self.assertIn("preprocessors", kwargs)
        self.assertEqual(kwargs["preprocessors"], self.preprocessors)
        self.assertIn("callback", kwargs)
        self.assertEqual(kwargs["callback"], self.callback)
        self.assertIn("store_classifiers", kwargs)
        self.assertEqual(kwargs["store_classifiers"], self.store_classifiers)
        self.assertIn("store_examples", kwargs)
        self.assertEqual(kwargs["store_examples"], self.store_examples)

    def test_generating_indices(self):
        indices1, _ = Orange.evaluation.testing.cross_validation([], self.examples, self.folds)
        indices2, _ = Orange.evaluation.testing.cross_validation([], self.examples, self.folds)

        # By default, cross_validation generates the same indices every time. (multiple runs of
        # cross-validation produce the same results)
        self.assertEqual(indices1, indices2)

        indices3, _ = Orange.evaluation.testing.cross_validation([], self.examples, self.folds, random_generator=self.random_generator)
        indices4, _ = Orange.evaluation.testing.cross_validation([], self.examples, self.folds, random_generator=self.random_generator)
        # Providing the same random seed should give use the same indices
        self.assertEqual(indices3, indices4)
        # But different from default ones
        self.assertNotEqual(indices1, indices3)

        rg = Orange.core.RandomGenerator()
        indices5, _ = Orange.evaluation.testing.cross_validation([], self.examples, self.folds, random_generator=rg)
        rg.reset()
        indices6, _ = Orange.evaluation.testing.cross_validation([], self.examples, self.folds, random_generator=rg)
        # Using the same random generator and resetting it before calling cross-validation should result
        # in same indices
        self.assertEqual(indices5, indices6)

    def test_stratification(self):
        ds = Orange.data.Table("iris")
        indices1,_ = Orange.evaluation.testing.cross_validation([], ds, 3, stratified=Orange.core.MakeRandomIndices.NotStratified)
        indices2,_ = Orange.evaluation.testing.cross_validation([], ds, 3, stratified=Orange.core.MakeRandomIndices.Stratified)

        # We know that the iris dataset has 150 instances and 3 class values. First 50 examples belong to first class,
        # Next 50 to the second and the rest to the third.
        # When using stratification, class distributions in folds should be about the same (max difference of one
        # instance per class)
        freq = Counter(indices2[:50]), Counter(indices2[50:100]), Counter(indices2[100:]) #Get class value distributions
        frequencies = [[freq[fold][cls] for cls in range(3)] for fold in range(3)]
        for value_counts in frequencies:
            self.assertTrue(max(value_counts)-min(value_counts) <= 1)

        # If stratification is not enabled, class value numbers in different folds usually vary.
        freq = Counter(indices1[:50]), Counter(indices1[50:100]), Counter(indices1[100:]) #Get class value distributions
        frequencies = [[freq[fold][cls] for cls in range(3)] for fold in range(3)]
        for value_counts in frequencies:
            self.assertTrue(max(value_counts)-min(value_counts) > 1)

    @classmethod
    def tearDownClass(cls):
        reload(Orange.evaluation.testing)

class TestLeaveOneOut(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def dummy_test_with_indices(*args, **kwds):
            return kwds["indices"], kwds
        Orange.evaluation.testing.test_with_indices = dummy_test_with_indices

    def setUp(self):
        self.examples = Orange.data.Table("iris")
        self.folds = 10
        self.preprocessors = ("P1", "P2", "P3", "P4")
        def callback(): pass
        self.callback = callback
        self.random_generator = 5
        self.store_classifiers = True
        self.store_examples = True

    def test_all_arguments_forwarded(self):
        indices, kwargs = Orange.evaluation.testing.leave_one_out([], self.examples,
                                                                 preprocessors=self.preprocessors,
                                                                 callback=self.callback,
                                                                 store_classifiers=self.store_classifiers,
                                                                 store_examples=self.store_classifiers)
        self.assertIn("preprocessors", kwargs)
        self.assertEqual(kwargs["preprocessors"], self.preprocessors)
        self.assertIn("callback", kwargs)
        self.assertEqual(kwargs["callback"], self.callback)
        self.assertIn("store_classifiers", kwargs)
        self.assertEqual(kwargs["store_classifiers"], self.store_classifiers)
        self.assertIn("store_examples", kwargs)
        self.assertEqual(kwargs["store_examples"], self.store_examples)
        self.assertItemsEqual(indices, range(len(self.examples)))

    @classmethod
    def tearDownClass(cls):
        reload(Orange.evaluation.testing)

if __name__ == '__main__':
    unittest.main()
