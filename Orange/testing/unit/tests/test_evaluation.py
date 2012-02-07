import itertools, operator, unittest
from collections import Counter

import Orange

example_no = Orange.feature.new_meta_id()

class DummyLearner(Orange.classification.majority.MajorityLearner):
    def __init__(self, id=None):
        self.id=id
        self.data = []
        self.classifiers = []
        self.classifier_no = 0
        super(DummyLearner, self).__init__()

    def __call__(self, dataset, weight=0):
        self.data.append(dataset)

        cl = super(DummyLearner, self).__call__(dataset, weight)
        cl = DummyClassifier(cl, self.id, self.classifier_no)
        self.classifier_no += 1
        self.classifiers.append(cl)
        return cl

class DummyClassifier(object):
    name="DummyClassifier"
    def __init__(self, base_class, learner_id, classifier_no):
        self.base_class = base_class
        self.classifier_no = classifier_no
        self.learner_id = learner_id
        self.data = []
        self.predictions = []

    def __call__(self, example, options=None):
        value, probability = self.base_class.__call__(example, options)
        p = [self.learner_id, self.classifier_no, int(example[example_no])]
        self.data.append(example)
        self.predictions.append(p)

        return value, p

class DummyPreprocessor(object):
    def __init__(self, meta_id):
        self.meta_id = meta_id

    def __call__(self, *datasets):
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

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.example_no = example_no
        self.learner = DummyLearner()
        self.examples = Orange.data.Table("iris")
        for i, inst in enumerate(self.examples):
            inst[self.example_no] = i

        self.preprocessed_with_both = Orange.feature.new_meta_id()
        self.preprocessed_with_learn = Orange.feature.new_meta_id()
        self.preprocessed_with_test = Orange.feature.new_meta_id()
        self.preprocessed_with_learn_test = Orange.feature.new_meta_id()
        self.preprocessors = (("B", DummyPreprocessor(self.preprocessed_with_both)),
                              ("L", DummyPreprocessor(self.preprocessed_with_learn)),
                              ("T", DummyPreprocessor(self.preprocessed_with_test)),
                              ("LT", DummyPreprocessor(self.preprocessed_with_learn_test)))

        self.folds = 3
        examples_in_fold = len(self.examples) // self.folds
        self.indices = [i // examples_in_fold for i in range(len(self.examples))]
        self.callback_calls = 0
        reload(Orange.evaluation.testing)
        self.evaluation = Orange.evaluation.testing

    def test_with_indices(self):
        learners = [DummyLearner(id=0), DummyLearner(id=1), DummyLearner(id=2)]
        self.callback_calls = 0

        test_results = self.evaluation.test_with_indices(learners, (self.examples, 0), self.indices, callback=self.callback)

        for l, learner in enumerate(learners):
            predicted_results = [prediction for classifier in learner.classifiers for prediction in classifier.predictions]
            returned_results = [r.probabilities[l] for r in test_results.results]
            self.assertItemsEqual(returned_results, predicted_results)

            # Each example should be used for training (n-1)x, where n is the number of folds
            example_cnt = 0
            for fold, data in enumerate(learner.data):
                for example in data:
                    # Classifier should be trained on examples where fold is different from current fold
                    self.assertNotEqual(self.indices[int(example[self.example_no])], fold)
                    example_cnt += 1
            self.assertEqual(example_cnt, len(self.examples) * (self.folds-1))

            # Each example should be used for testing only once
            example_cnt = 0
            for fold, classifier in enumerate(learner.classifiers):
                for example in classifier.data:
                    # Classifier should perform classification on examples with same fold number
                    self.assertEqual(self.indices[int(example[self.example_no])], fold)
                    example_cnt += 1
            self.assertEqual(example_cnt, len(self.examples))

        # Callback function should be called once for every fold.
        self.assertEqual(self.callback_calls, self.folds)

    def callback(self):
        self.callback_calls += 1

    def test_with_indices_can_store_examples_and_classifiers(self):
        test_results = self.evaluation.test_with_indices([self.learner], self.examples, self.indices,
                                                        store_examples=True, store_classifiers=True)
        self.assertGreater(len(test_results.examples), 0)
        self.assertGreater(len(test_results.classifiers), 0)
        classifiers = map(operator.itemgetter(0), test_results.classifiers)
        self.assertItemsEqual(self.learner.classifiers, classifiers)

    def test_with_indices_uses_preprocessors(self):
        self.evaluation.test_with_indices([self.learner], self.examples, self.indices,
                                           preprocessors=self.preprocessors)
        self.assertPreprocessedCorrectly()

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
        learners = [DummyLearner(id=0), DummyLearner(id=1), DummyLearner(id=2)]
        self.callback_calls = 0
        folds = 10
        test_results = self.evaluation.cross_validation(learners, (self.examples, 0), folds=folds, callback=self.callback)

        for l, learner in enumerate(learners):
            predicted_results = [prediction for classifier in learner.classifiers for prediction in classifier.predictions]
            returned_results = [r.probabilities[l] for r in test_results.results]
            self.assertItemsEqual(returned_results, predicted_results)

            # Each example should be used for training (n-1)x, where n is the number of folds
            example_cnt = 0
            for fold, data in enumerate(learner.data):
                for example in data:
                    example_cnt += 1
            self.assertEqual(example_cnt, len(self.examples) * (folds-1))

            # Each example should be used for testing only once
            example_cnt = 0
            for fold, classifier in enumerate(learner.classifiers):
                for example in classifier.data:
                    example_cnt += 1
            self.assertEqual(example_cnt, len(self.examples))

    def test_leave_one_out(self):
        learners = [DummyLearner(id=0), DummyLearner(id=1), DummyLearner(id=2)]
        self.callback_calls = 0
        
        test_results = self.evaluation.leave_one_out(learners, self.examples)
        for l, learner in enumerate(learners):
            predicted_results = [prediction for classifier in learner.classifiers for prediction in classifier.predictions]
            returned_results = [r.probabilities[l] for r in test_results.results]
            self.assertItemsEqual(returned_results, predicted_results)

            # Each example should be used for training (n-1)x, where n is the number of folds
            example_cnt = 0
            for fold, data in enumerate(learner.data):
                for example in data:
                    example_cnt += 1
            self.assertEqual(example_cnt, len(self.examples) * (len(self.examples)-1))

            # Each example should be used for testing only once
            example_cnt = 0
            for fold, classifier in enumerate(learner.classifiers):
                for example in classifier.data:
                    example_cnt += 1
            self.assertEqual(example_cnt, len(self.examples))

    def test_on_data(self):
        pass

    def test_on_data_can_store_examples_and_classifiers(self):
        learner = DummyLearner()
        classifier = learner(self.examples)
        # Passing store_examples = True should make examples accessible
        test_results = self.evaluation.test_on_data([classifier], self.examples,
                                                    store_examples=True, store_classifiers=True)
        self.assertGreater(len(test_results.examples), 0)
        self.assertGreater(len(test_results.classifiers), 0)
        self.assertItemsEqual(learner.classifiers, test_results.classifiers)

    def test_learn_and_test_on_learn_data(self):
        self.callback_calls = 0
        learner = DummyLearner()

        test_results = self.evaluation.learn_and_test_on_learn_data([learner], self.examples, callback=self.callback,
                                                    store_examples=True, store_classifiers=True)

        self.assertEqual(self.callback_calls, 1)



    def test_learn_and_test_on_learn_data_with_preprocessors(self):
        self.learner = DummyLearner()
        test_results = self.evaluation.learn_and_test_on_learn_data([self.learner], self.examples,
                                                    preprocessors=self.preprocessors,
                                                    callback=self.callback, store_examples=True, store_classifiers=True)
        self.assertPreprocessedCorrectly()

    def test_learn_and_test_on_learn_data_can_store_examples_and_classifiers(self):
        learner = DummyLearner()
        
        test_results = self.evaluation.learn_and_test_on_learn_data([learner], self.examples,
                                                    store_examples=True, store_classifiers=True)
        self.assertGreater(len(test_results.examples), 0)
        self.assertGreater(len(test_results.classifiers), 0)
        self.assertItemsEqual(learner.classifiers, test_results.classifiers)

    def test_learn_and_test_on_test_data(self):
        self.callback_calls = 0
        learner = DummyLearner()

        test_results = self.evaluation.learn_and_test_on_test_data([learner], self.examples, self.examples,
                                                    callback=self.callback, store_examples=True, store_classifiers=True)
        self.assertEqual(self.callback_calls, 1)

    def test_learn_and_test_on_test_data_with_preprocessors(self):
        self.learner = DummyLearner()
        test_results = self.evaluation.learn_and_test_on_test_data([self.learner], self.examples, self.examples,
                                                    preprocessors=self.preprocessors,
                                                    callback=self.callback, store_examples=True, store_classifiers=True)
        self.assertPreprocessedCorrectly()

    def test_learn_and_test_on_test_data_can_store_examples_and_classifiers(self):
        learner = DummyLearner()

        test_results = self.evaluation.learn_and_test_on_test_data([learner], self.examples, self.examples,
                                                    store_examples=True, store_classifiers=True)
        self.assertGreater(len(test_results.examples), 0)
        self.assertGreater(len(test_results.classifiers), 0)
        self.assertItemsEqual(learner.classifiers, test_results.classifiers)

    def test_learning_curve_with_test_data(self):
        learner = DummyLearner()
        times=10
        proportions=Orange.core.frange(0.1)
        test_results = self.evaluation.learning_curve_with_test_data([learner], self.examples, self.examples,
                                                                              times=times, proportions=proportions)
        # We expect the method to return a list of test_results, one instance for each proportion. Each
        # instance should have "times" folds.
        self.assertEqual(len(test_results), len(proportions))
        for test_result in test_results:
            self.assertEqual(test_result.numberOfIterations, times)
            self.assertEqual(len(test_result.results), times*len(self.examples))



    def test_learning_curve_with_test_data_can_store_examples_and_classifiers(self):
        learner = DummyLearner()

        test_results = self.evaluation.learn_and_test_on_test_data([learner], self.examples, self.examples,
                                                                            store_examples=True, store_classifiers=True)
        self.assertGreater(len(test_results.examples), 0)
        self.assertGreater(len(test_results.classifiers), 0)
        self.assertItemsEqual(learner.classifiers, test_results.classifiers)


    def test_proportion_test(self):
        self.callback_calls = 0
        times = 10
        learner = DummyLearner()

        test_results = self.evaluation.proportion_test([learner], self.examples, learning_proportion=.7, times=times,
                                                                callback = self.callback)

        self.assertEqual(self.callback_calls, times)

    def test_learning_curve(self):
        self.callback_calls = 0
        times = 10
        proportions=Orange.core.frange(0.1)
        folds=10
        learner = DummyLearner()

        test_results = self.evaluation.learning_curve([learner], self.examples,
                                                                callback = self.callback)

        # Ensure that each iteration is learned on correct proportion of training examples
        for proportion, data in zip((p for p in proportions for _ in range(10)), learner.data):
            actual_examples = len(data)
            expected_examples = len(self.examples)*proportion*(folds-1)/folds
            self.assertTrue(abs(actual_examples - expected_examples) <= 1)

        # Ensure results are not lost
        predicted_results = [prediction for classifier in learner.classifiers for prediction in classifier.predictions]
        returned_results = [r.probabilities[0] for tr in test_results for r in tr.results]
        self.assertItemsEqual(returned_results, predicted_results)
        
        self.assertEqual(self.callback_calls, folds*len(proportions))

    #TODO: LearningCurveN tests

    def assertPreprocessedCorrectly(self):
        # Original examples should be left intact
        for example in self.examples:
            self.assertFalse(example.has_meta(self.preprocessed_with_both))
            self.assertFalse(example.has_meta(self.preprocessed_with_learn))
            self.assertFalse(example.has_meta(self.preprocessed_with_test))
            self.assertFalse(example.has_meta(self.preprocessed_with_learn_test))
        for fold, data in enumerate(self.learner.data):
            for example in data:
                # Preprocessors both, learn and learntest should be applied to learn data.
                self.assertTrue(example.has_meta(self.preprocessed_with_both))
                self.assertTrue(example.has_meta(self.preprocessed_with_learn))
                self.assertTrue(example.has_meta(self.preprocessed_with_learn_test))
                # Preprocessor test should not be applied to learn data.
                self.assertFalse(example.has_meta(self.preprocessed_with_test))
        for fold, classifier in enumerate(self.learner.classifiers):
            for example in classifier.data:
                # Preprocessors both, test and learntest should be applied to test data.
                self.assertTrue(example.has_meta(self.preprocessed_with_both))
                self.assertTrue(example.has_meta(self.preprocessed_with_test))
                self.assertTrue(example.has_meta(self.preprocessed_with_learn_test))
                # Preprocessor learn should not be applied to test data.
                self.assertFalse(example.has_meta(self.preprocessed_with_learn))

if __name__ == '__main__':
    unittest.main()
