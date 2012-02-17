import itertools, operator
import collections

try:
    import unittest2 as unittest
except:
    import unittest

import Orange

class DummyLearner(Orange.classification.majority.MajorityLearner):
    def __init__(self, id=None):
        self.id = id
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
    name = "DummyClassifier"
    def __init__(self, base_class, learner_id, classifier_no):
        self.base_class = base_class
        self.classifier_no = classifier_no
        self.learner_id = learner_id
        self.data = []
        self.predictions = []

    def __call__(self, example, options=None):
        value, probability = self.base_class.__call__(example, options)
        p = [self.learner_id, self.classifier_no, int(example[ID])]
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

def broken_preprocessor(*args):
    return []

ID = Orange.feature.Descriptor.new_meta_id()
WEIGHT = Orange.feature.Descriptor.new_meta_id()
def prepare_dataset():
    ds = Orange.data.Table("iris")
    for i, inst in enumerate(ds):
        inst[ID] = i
        inst[WEIGHT] = 2 * i
    return ds

class EvaluationTest(object):
    evaluation = None

    def setUp(self):
        self.learner = DummyLearner()
        self.examples = prepare_dataset()

    def test_can_be_run_without_weights(self):
        self.evaluation([self.learner], self.examples)

    def test_can_be_run_with_weights(self):
        self.evaluation([self.learner], (self.examples, 0))

    def test_returns_correct_results(self):
        learners = [DummyLearner(id=0), DummyLearner(id=1), DummyLearner(id=2)]
        test_results = self.evaluation(learners, self.examples)

        for l, learner in enumerate(learners):
            predicted_results = [prediction
                                 for classifier in learner.classifiers
                                 for prediction in classifier.predictions]
            returned_results = [r.probabilities[l]
                                for r in test_results.results]
            self.assertItemsEqual(returned_results, predicted_results)

    def test_can_store_examples(self):
        test_results = self.evaluation([self.learner], self.examples,
            store_examples=True, store_classifiers=True)
        self.assertGreater(len(test_results.examples), 0)

    def test_can_store_classifiers(self):
        test_results = self.evaluation([self.learner], self.examples,
            store_classifiers=True)
        self.assertGreater(len(test_results.classifiers), 0)
        if not isinstance(test_results.classifiers[0], collections.Iterable):
            self.assertItemsEqual(self.learner.classifiers,
                                  test_results.classifiers)
        else:
            classifiers = map(operator.itemgetter(0), test_results.classifiers)
            self.assertItemsEqual(self.learner.classifiers, classifiers)

    def test_uses_preprocessors(self):
        preprocessed_with_both = Orange.feature.Descriptor.new_meta_id()
        preprocessed_with_learn = Orange.feature.Descriptor.new_meta_id()
        preprocessed_with_test = Orange.feature.Descriptor.new_meta_id()
        preprocessed_with_learn_test = Orange.feature.Descriptor.new_meta_id()
        preprocessors = (("B", DummyPreprocessor(preprocessed_with_both)),
                         ("L", DummyPreprocessor(preprocessed_with_learn)),
                         ("T", DummyPreprocessor(preprocessed_with_test)),
                         ("LT", DummyPreprocessor(preprocessed_with_learn_test)))
        self.evaluation([self.learner], self.examples,
            preprocessors=preprocessors)

        # Original examples should be left intact
        for example in self.examples:
            self.assertFalse(example.has_meta(preprocessed_with_both))
            self.assertFalse(example.has_meta(preprocessed_with_learn))
            self.assertFalse(example.has_meta(preprocessed_with_test))
            self.assertFalse(example.has_meta(preprocessed_with_learn_test))
        for fold, data in enumerate(self.learner.data):
            for example in data:
                # Preprocessors both, learn and learntest should be applied to learn data.
                self.assertTrue(example.has_meta(preprocessed_with_both))
                self.assertTrue(example.has_meta(preprocessed_with_learn))
                self.assertTrue(example.has_meta(preprocessed_with_learn_test))
                # Preprocessor test should not be applied to learn data.
                self.assertFalse(example.has_meta(preprocessed_with_test))
        for fold, classifier in enumerate(self.learner.classifiers):
            for example in classifier.data:
                # Preprocessors both, test and learntest should be applied to test data.
                self.assertTrue(example.has_meta(preprocessed_with_both))
                self.assertTrue(example.has_meta(preprocessed_with_test))
                self.assertTrue(example.has_meta(preprocessed_with_learn_test))
                # Preprocessor learn should not be applied to test data.
                self.assertFalse(example.has_meta(preprocessed_with_learn))

class TestCrossValidation(EvaluationTest, unittest.TestCase):
    evaluation = Orange.evaluation.testing.cross_validation

class TestLeaveOneOut(EvaluationTest, unittest.TestCase):
    evaluation = Orange.evaluation.testing.leave_one_out

class TestProportionTest(EvaluationTest, unittest.TestCase):
    evaluation = Orange.evaluation.testing.proportion_test

class TestLearnAndTestOnTrainData(EvaluationTest, unittest.TestCase):
    evaluation = Orange.evaluation.testing.learn_and_test_on_learn_data

class TestLearnAndTestOnTestData(EvaluationTest, unittest.TestCase):
    @property
    def evaluation(self):
        def wrapper(learners, data, *args, **kwargs):
            return Orange.evaluation.testing.learn_and_test_on_test_data\
                (learners, data, self.examples, *args, **kwargs)
        return wrapper

class TestOnData(EvaluationTest, unittest.TestCase):
    @property
    def evaluation(self):
        def wrapper(learners, data, *args, **kwargs):
            classifiers = [l(*data) if isinstance(data, tuple) else l(data)
                           for l in learners]
            return Orange.evaluation.testing.test_on_data\
                (classifiers, data, *args, **kwargs)
        return wrapper

    def test_uses_preprocessors(self):
        # Since this function get classifiers, preprocessors are
        # meaningless
        pass

class TestLearningCurveWithTestData(EvaluationTest, unittest.TestCase):
    @property
    def evaluation(self):
        def wrapper(*args, **kwargs):
            return self.learning_curve(*args, **kwargs)[-1]
        return  wrapper

    @property
    def learning_curve(self):
        def wrapper(learners, data, *args, **kwargs):
            return Orange.evaluation.testing.learning_curve_with_test_data(
                learners, data, self.examples, *args, **kwargs)
        return wrapper

    def test_can_store_classifiers(self):
        self.skipTest("")

    def test_returns_correct_results(self):
        learner = DummyLearner()
        learning_curve_results = self.learning_curve([learner], self.examples)
        for test_results, classifier in zip(learning_curve_results, self.learner.classifiers):
            predicted_results = [prediction
                                 for prediction in classifier.predictions]
            returned_results = [r.probabilities[0]
                                for r in test_results.results]
            self.assertItemsEqual(returned_results, predicted_results)

class TestExperimentResults(unittest.TestCase):
    def test_add_results(self):
        learners = [DummyLearner(id=id) for id in range(3)]
        ds = prepare_dataset()
        results = [Orange.evaluation.testing.learn_and_test_on_learn_data(
                    [learners[i]], ds) for i in range(3)]

        results[0].add(results[1], 0)
        results[0].add(results[2], 0)

if __name__ == '__main__':
    unittest.main()
