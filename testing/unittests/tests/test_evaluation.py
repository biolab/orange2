import operator, unittest
from collections import Counter

import Orange

class DummyLearner(Orange.classification.majority.MajorityLearner):
    def __init__(self, *args, **kwds):
        self.data = []
        self.classifiers = []
        super(DummyLearner, self).__init__(*args, **kwds)

    def __call__(self, dataset, weight=0):
        self.data.append(dataset)
        cl = super(DummyLearner, self).__call__(dataset, weight)
        cl = DummyClassifier(cl)
        self.classifiers.append(cl)
        return cl

class DummyClassifier(object):
    name="DummyClassifier"
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


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.meta_id = Orange.data.variable.new_meta_id()
        self.examples = create_examples(self.meta_id)

        self.preprocessed_with_both = Orange.data.variable.new_meta_id()
        self.preprocessed_with_learn = Orange.data.variable.new_meta_id()
        self.preprocessed_with_test = Orange.data.variable.new_meta_id()
        self.preprocessed_with_learn_test = Orange.data.variable.new_meta_id()
        self.preprocessors = (("B", DummyPreprocessor(self.preprocessed_with_both)),
                              ("L", DummyPreprocessor(self.preprocessed_with_learn)),
                              ("T", DummyPreprocessor(self.preprocessed_with_test)),
                              ("LT", DummyPreprocessor(self.preprocessed_with_learn_test)))

        self.folds = 3
        examples_in_fold = len(self.examples) // self.folds
        self.indices = [i // examples_in_fold for i in range(len(self.examples))]
        self.callback_calls = 0

        self.evaluation = Orange.evaluation.testing.Evaluation()

    def callback(self):
        self.callback_calls += 1

    def test_with_indices(self):
        # Perform test on MajorityLearner and Iris dataset.
        learner = DummyLearner()
        self.callback_calls = 0
        test_results = self.evaluation.test_with_indices([learner], (self.examples, 0), self.indices, callback=self.callback)
        expected_results = [1]*50+[0]*100
        predicted_classes = map(lambda x:x.classes[0], test_results.results)
        self.assertItemsEqual(expected_results, predicted_classes)

        # Each example should be used for training (n-1)x, where n is the number of folds
        example_cnt = 0
        for fold, data in enumerate(learner.data):
            for example in data:
                # Classifier should be trained on examples where fold is different from current fold
                self.assertNotEqual(self.indices[int(example[self.meta_id])], fold)
                example_cnt += 1
        self.assertEqual(example_cnt, len(self.examples) * (self.folds-1))

        # Each example should be used for testing only once
        example_cnt = 0
        for fold, classifier in enumerate(learner.classifiers):
            for example in classifier.data:
                # Classifier should perform classification on examples with same fold number
                self.assertEqual(self.indices[int(example[self.meta_id])], fold)
                example_cnt += 1
        self.assertEqual(example_cnt, len(self.examples))

        # Callback function should be called once for every fold.
        self.assertEqual(self.callback_calls, self.folds)

    def test_with_indices_can_store_examples_and_classifiers(self):
        learner = DummyLearner()
        # Passing store_examples = True should make examples accessible
        test_results = self.evaluation.test_with_indices([learner], self.examples, self.indices,
                                                        store_examples=True, store_classifiers=True)
        self.assertGreater(len(test_results.examples), 0)
        self.assertGreater(len(test_results.classifiers), 0)
        classifiers = map(operator.itemgetter(0), test_results.classifiers)
        self.assertItemsEqual(learner.classifiers, classifiers)

    def test_with_indices_uses_preprocessors(self):
        # Preprocessors should be applyed to data as specified in their type
        learner = DummyLearner()
        self.evaluation.test_with_indices([learner],
                                     self.examples,
                                     self.indices,
                                     preprocessors=self.preprocessors)

        # Original examples should be left intact
        for example in self.examples:
            self.assertFalse(example.has_meta(self.preprocessed_with_both))
            self.assertFalse(example.has_meta(self.preprocessed_with_learn))
            self.assertFalse(example.has_meta(self.preprocessed_with_test))
            self.assertFalse(example.has_meta(self.preprocessed_with_learn_test))

        for fold, data in enumerate(learner.data):
            for example in data:
                # Preprocessors both, learn and learntest should be applied to learn data.
                self.assertTrue(example.has_meta(self.preprocessed_with_both))
                self.assertTrue(example.has_meta(self.preprocessed_with_learn))
                self.assertTrue(example.has_meta(self.preprocessed_with_learn_test))
                # Preprocessor test should not be applied to learn data.
                self.assertFalse(example.has_meta(self.preprocessed_with_test))

        for fold, classifier in enumerate(learner.classifiers):
            for example in classifier.data:
                # Preprocessors both, test and learntest should be applied to test data.
                self.assertTrue(example.has_meta(self.preprocessed_with_both))
                self.assertTrue(example.has_meta(self.preprocessed_with_test))
                self.assertTrue(example.has_meta(self.preprocessed_with_learn_test))
                # Preprocessor learn should not be applied to test data.
                self.assertFalse(example.has_meta(self.preprocessed_with_learn))

    def test_with_indices_handles_errors(self):
        learner = DummyLearner()
        # No data should raise a Value error
        with self.assertRaises(ValueError):
            self.evaluation.test_with_indices([learner], [], self.indices)

        # If one fold is not represented in indices, cross validation should still execute
        self.evaluation.test_with_indices([learner], self.examples, [2] + [1]*(len(self.examples)-1))

        # If preprocessors is "broken" (returns no data), error should be  raised
        with self.assertRaises(SystemError):
            self.evaluation.test_with_indices([learner], self.examples, self.indices, preprocessors=(("L", BrokenPreprocessor()),))
        with self.assertRaises(SystemError):
            self.evaluation.test_with_indices([learner], self.examples, self.indices, preprocessors=(("T", BrokenPreprocessor()),))


    def test_cross_validation(self):
        def dummy_test_with_indices(*args, **kwargs):
            return kwargs["indices"], kwargs
        self.evaluation.test_with_indices = dummy_test_with_indices


        _, kwargs = self.evaluation.cross_validation([], self.examples, self.folds,
                                                                 preprocessors=self.preprocessors,
                                                                 callback=self.callback,
                                                                 store_classifiers=True,
                                                                 store_examples=True)
        self.assertIn("preprocessors", kwargs)
        self.assertEqual(kwargs["preprocessors"], self.preprocessors)
        self.assertIn("callback", kwargs)
        self.assertEqual(kwargs["callback"], self.callback)
        self.assertIn("store_classifiers", kwargs)
        self.assertEqual(kwargs["store_classifiers"], True)
        self.assertIn("store_examples", kwargs)
        self.assertEqual(kwargs["store_examples"], True)

        indices1, _ = self.evaluation.cross_validation([], self.examples, self.folds)
        indices2, _ = self.evaluation.cross_validation([], self.examples, self.folds)

        # By default, cross_validation generates the same indices every time. (multiple runs of
        # cross-validation produce the same results)
        self.assertEqual(indices1, indices2)

        indices3, _ = self.evaluation.cross_validation([], self.examples, self.folds, random_generator=3145)
        indices4, _ = self.evaluation.cross_validation([], self.examples, self.folds, random_generator=3145)
        # Providing the same random seed should give use the same indices
        self.assertEqual(indices3, indices4)
        # But different from default ones
        self.assertNotEqual(indices1, indices3)

        rg = Orange.core.RandomGenerator()
        indices5, _ = self.evaluation.cross_validation([], self.examples, self.folds, random_generator=rg)
        rg.reset()
        indices6, _ = self.evaluation.cross_validation([], self.examples, self.folds, random_generator=rg)
        # Using the same random generator and resetting it before calling cross-validation should result
        # in same indices
        self.assertEqual(indices5, indices6)

        ds = Orange.data.Table("iris")
        indices1,_ = self.evaluation.cross_validation([], ds, 3, stratified=Orange.core.MakeRandomIndices.NotStratified)
        indices2,_ = self.evaluation.cross_validation([], ds, 3, stratified=Orange.core.MakeRandomIndices.Stratified)

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

    def test_leave_one_out(self):
        def dummy_test_with_indices(*args, **kwargs):
            return kwargs["indices"], kwargs
        self.evaluation.test_with_indices = dummy_test_with_indices
        
        indices, kwargs = self.evaluation.leave_one_out([], self.examples,
                                                                 preprocessors=self.preprocessors,
                                                                 callback=self.callback,
                                                                 store_classifiers=True,
                                                                 store_examples=True)
        self.assertIn("preprocessors", kwargs)
        self.assertEqual(kwargs["preprocessors"], self.preprocessors)
        self.assertIn("callback", kwargs)
        self.assertEqual(kwargs["callback"], self.callback)
        self.assertIn("store_classifiers", kwargs)
        self.assertEqual(kwargs["store_classifiers"], True)
        self.assertIn("store_examples", kwargs)
        self.assertEqual(kwargs["store_examples"], True)
        self.assertItemsEqual(indices, range(len(self.examples)))

    def test_rest(self):
        classifiers = [DummyLearner()(self.examples)]
        learners = [DummyLearner()]
        Orange.evaluation.testing.test_on_data(classifiers, self.examples)
        Orange.evaluation.testing.learn_and_test_on_learn_data(learners, self.examples)
        Orange.evaluation.testing.learn_and_test_on_test_data(learners, self.examples, self.examples)
        Orange.evaluation.testing.learning_curve(learners, self.examples)
        Orange.evaluation.testing.learning_curve_n(learners, self.examples)
        Orange.evaluation.testing.learning_curve_with_test_data(learners, self.examples, self.examples)
        Orange.evaluation.testing.proportion_test(learners, self.examples, 0.7)


if __name__ == '__main__':
    unittest.main()
