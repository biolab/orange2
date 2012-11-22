import itertools

import Orange
from Orange.utils import demangle_examples, getobjectname
from Orange.utils import deprecated_keywords, deprecated_members

#### Data structures

TEST_TYPE_SINGLE = 0
TEST_TYPE_MLC = 1
TEST_TYPE_MULTITARGET = 2

class TestedExample:
    """
    TestedExample stores predictions of different classifiers for a
    single testing data instance.

    .. attribute:: classes

        A list of predictions of type Value, one for each classifier.

    .. attribute:: probabilities

        A list of probabilities of classes, one for each classifier.

    .. attribute:: iteration_number

        Iteration number (e.g. fold) in which the TestedExample was
        created/tested.

    .. attribute actual_class

        The correct class of the example

    .. attribute weight

        Instance's weight; 1.0 if data was not weighted
    """

    @deprecated_keywords({"iterationNumber": "iteration_number",
                          "actualClass": "actual_class"})
    def __init__(self, iteration_number=None, actual_class=None, n=0, weight=1.0):
        """
        :param iteration_number: The iteration number of TestedExample.
        :param actual_class: The actual class of TestedExample.
        :param n: The number of learners.
        :param weight: The weight of the TestedExample.
        """
        self.classes = [None]*n
        self.probabilities = [None]*n
        self.iteration_number = iteration_number
        self.actual_class= actual_class
        self.weight = weight

    def add_result(self, aclass, aprob):
        """Append a new result (class and probability prediction by a single classifier) to the classes and probabilities field."""
    
        if isinstance(aclass, (list, tuple)):
            self.classes.append(aclass)
            self.probabilities.append(aprob)
        elif type(aclass.value)==float:
            self.classes.append(float(aclass))
            self.probabilities.append(aprob)
        else:
            self.classes.append(int(aclass))
            self.probabilities.append(aprob)

    def set_result(self, i, aclass, aprob):
        """Set the result of the i-th classifier to the given values."""
        if isinstance(aclass, (list, tuple)):
            self.classes[i] = aclass
            self.probabilities[i] = aprob
        elif type(aclass.value)==float:
            self.classes[i] = float(aclass)
            self.probabilities[i] = aprob
        else:
            self.classes[i] = int(aclass)
            self.probabilities[i] = aprob

    def __repr__(self):
        return str(self.__dict__)

TestedExample = deprecated_members({"iterationNumber": "iteration_number",
                                    "actualClass": "actual_class"
                                    })(TestedExample)

def mt_vals(vals):
    """
    Substitution for the unpicklable lambda function for multi-target classifiers.
    """
    return [val if val.is_DK() else int(val) if val.variable.var_type == Orange.feature.Type.Discrete
                                            else float(val) for val in vals]

class ExperimentResults(object):
    """
    ``ExperimentResults`` stores results of one or more repetitions of
    some test (cross validation, repeated sampling...) under the same
    circumstances. Instances of this class are constructed by sampling
    and testing functions from module :obj:`Orange.evaluation.testing`
    and used by methods in module :obj:`Orange.evaluation.scoring`.

    .. attribute:: results

        A list of instances of :obj:`TestedExample`, one for each
        example in the dataset.

    .. attribute:: number_of_iterations

        Number of iterations. This can be the number of folds (in
        cross validation) or the number of repetitions of some
        test. :obj:`TestedExample`'s attribute ``iteration_number``
        should be in range ``[0, number_of_iterations-1]``.

    .. attribute:: number_of_learners

        Number of learners. Lengths of lists classes and probabilities
        in each :obj:`TestedExample` should equal
        ``number_of_learners``.

    .. attribute:: classifier_names

        Stores the names of the classifiers.

    .. attribute:: classifiers

        A list of classifiers, one element for each iteration of
        sampling and learning (eg. fold). Each element is a list of
        classifiers, one for each learner. For instance,
        ``classifiers[2][4]`` refers to the 3rd repetition, 5th
        learning algorithm.

        Note that functions from :obj:`~Orange.evaluation.testing`
        only store classifiers it enabled by setting
        ``storeClassifiers`` to ``1``.

    ..
        .. attribute:: loaded

            If the experimental method supports caching and there are no
            obstacles for caching (such as unknown random seeds), this is a
            list of boolean values. Each element corresponds to a classifier
            and tells whether the experimental results for that classifier
            were computed or loaded from the cache.

    .. attribute:: base_class

       The reference class for measures like AUC.

    .. attribute:: class_values

        The list of class values.

    .. attribute:: weights

        A flag telling whether the results are weighted. If ``False``,
        weights are still present in :obj:`TestedExample`, but they
        are all ``1.0``. Clear this flag, if your experimental
        procedure ran on weighted testing examples but you would like
        to ignore the weights in statistics.
    """
    @deprecated_keywords({"classifierNames": "classifier_names",
                          "classValues": "class_values",
                          "baseClass": "base_class",
                          "numberOfIterations": "number_of_iterations",
                          "numberOfLearners": "number_of_learners"})
    def __init__(self, iterations, classifier_names, class_values=None, weights=None, base_class=-1, domain=None, test_type=TEST_TYPE_SINGLE, labels=None, **argkw):
        self.class_values = class_values
        self.classifier_names = classifier_names
        self.number_of_iterations = iterations
        self.number_of_learners = len(classifier_names)
        self.results = []
        self.classifiers = []
        self.loaded = None
        self.base_class = base_class
        self.weights = weights
        self.test_type = test_type
        self.labels = labels

        if domain is not None:
            self.base_class = self.class_values = None
            if test_type == TEST_TYPE_SINGLE:
                if domain.class_var.var_type == Orange.feature.Type.Discrete:
                    self.class_values = list(domain.class_var.values)
                    self.base_class = domain.class_var.base_value
                    self.converter = int
                else:
                    self.converter = float
            elif test_type in (TEST_TYPE_MLC, TEST_TYPE_MULTITARGET):
                self.class_values = [list(cv.values) if cv.var_type == cv.Discrete else None for cv in domain.class_vars]
                self.labels = [var.name for var in domain.class_vars]
                self.converter = mt_vals


        self.__dict__.update(argkw)

    def load_from_files(self, learners, filename):
        raise NotImplementedError("This feature is no longer supported.")

    def save_to_files(self, learners, filename):
        raise NotImplementedError("This feature is no longer supported. Pickle whole class instead.")

    def create_tested_example(self, fold, example):
        actual = example.getclass() if self.test_type == TEST_TYPE_SINGLE \
                                  else example.get_classes()
        return TestedExample(fold,
                             self.converter(actual),
                             self.number_of_learners,
                             example.getweight(self.weights))

    def remove(self, index):
        """remove one learner from evaluation results"""
        for r in self.results:
            del r.classes[index]
            del r.probabilities[index]
        del self.classifier_names[index]
        self.number_of_learners -= 1

    def add(self, results, index, replace=-1):
        """add evaluation results (for one learner)"""
        if len(self.results)!=len(results.results):
            raise SystemError("mismatch in number of test cases")
        if self.number_of_iterations!=results.number_of_iterations:
            raise SystemError("mismatch in number of iterations (%d<>%d)" % \
                  (self.number_of_iterations, results.number_of_iterations))
        if len(self.classifiers) and len(results.classifiers)==0:
            raise SystemError("no classifiers in results")

        if replace < 0 or replace >= self.number_of_learners: # results for new learner
            self.classifier_names.append(results.classifier_names[index])
            self.number_of_learners += 1
            for i,r in enumerate(self.results):
                r.classes.append(results.results[i].classes[index])
                r.probabilities.append(results.results[i].probabilities[index])
            if len(self.classifiers):
                for i in range(self.number_of_iterations):
                    self.classifiers[i].append(results.classifiers[i][index])
        else: # replace results of existing learner
            self.classifier_names[replace] = results.classifier_names[index]
            for i,r in enumerate(self.results):
                r.classes[replace] = results.results[i].classes[index]
                r.probabilities[replace] = results.results[i].probabilities[index]
            if len(self.classifiers):
                for i in range(self.number_of_iterations):
                    self.classifiers[replace] = results.classifiers[i][index]

    def __repr__(self):
        return str(self.__dict__)

ExperimentResults = deprecated_members({"classValues": "class_values",
                                        "classifierNames": "classifier_names",
                                        "baseClass": "base_class",
                                        "numberOfIterations": "number_of_iterations",
                                        "numberOfLearners": "number_of_learners"
})(ExperimentResults)

    
class Evaluation(object):
    """Common methods for learner evaluation."""

    @deprecated_keywords({"pps": "preprocessors",
                          "strat": "stratified",
                          "randseed": "random_generator",
                          "indicesrandseed": "random_generator",
                          "randomGenerator": "random_generator",
                          "storeClassifiers": "store_classifiers",
                          "storeExamples": "store_examples"})
    def cross_validation(self, learners, examples, folds=10,
            stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible,
            preprocessors=(), random_generator=0, callback=None,
            store_classifiers=False, store_examples=False):
        """Cross validation test with specified number of folds.

        :param learners: list of learning algorithms
        :param examples: data instances used for training and testing
        :param folds: number of folds
        :param stratified: tells whether to stratify the sampling
        :param preprocessors: a list of preprocessors to be used on data (obsolete)
        :param random_generator: random seed or generator (see above)
        :param callback: a function that is called after finishing each fold
        :param store_classifiers: if ``True``, classifiers are stored in results
        :param store_examples: if ``True``, examples are stored in results
        :return: :obj:`ExperimentResults`
        """
        (examples, weight) = demangle_examples(examples)

        indices = Orange.core.MakeRandomIndicesCV(examples, folds,
            stratified=stratified, random_generator=random_generator)

        return self.test_with_indices(
            learners=learners,
            examples=(examples, weight),
            indices=indices,
            preprocessors=preprocessors,
            callback=callback,
            store_classifiers=store_classifiers,
            store_examples=store_examples)


    @deprecated_keywords({"pps": "preprocessors",
                          "storeClassifiers": "store_classifiers",
                          "storeExamples": "store_examples"})
    def leave_one_out(self, learners, examples, preprocessors=(),
            callback=None, store_classifiers=False, store_examples=False):
        """Leave-one-out evaluation of learning algorithms.

        :param learners: list of learning algorithms
        :param examples: data instances used for training and testing
        :param preprocessors: a list of preprocessors (obsolete)
        :param callback: a function that is called after finishing each fold
        :param store_classifiers: if ``True``, classifiers are stored in results
        :param store_examples: if ``True``, examples are stored in results
        :return: :obj:`ExperimentResults`
        """
        examples, weight = demangle_examples(examples)
        return self.test_with_indices(
            learners, (examples, weight), indices=range(len(examples)),
            preprocessors=preprocessors, callback=callback,
            store_classifiers=store_classifiers, store_examples=store_examples)

    @deprecated_keywords({"storeExamples": "store_examples",
                          "storeClassifiers": "store_classifiers=True",
                          "pps":"preprocessors"})
    def test_with_indices(self, learners, examples, indices, preprocessors=(),
            callback=None, store_classifiers=False, store_examples=False):
        """
        Perform a cross-validation-like test. Examples for each fold are
        selected based on given indices.

        :param learners: list of learning algorithms
        :param examples: data instances used for training and testing
        :param indices: a list of integer indices that sort examples into folds; each index corresponds to an example from ``examples``
        :param preprocessors: a list of preprocessors (obsolete)
        :param callback: a function that is called after each fold
        :param store_classifiers: if ``True``, classifiers are stored in results
        :param store_examples: if ``True``, examples are stored in results
        :return: :obj:`ExperimentResults`
        """
        examples, weight = demangle_examples(examples)
        if not examples:
            raise ValueError("Test data set with no examples")
        test_type = self.check_test_type(examples, learners)

        niterations = max(indices)+1
        test_result = ExperimentResults(niterations,
                                        classifier_names = [getobjectname(l) for l in learners],
                                        domain=examples.domain,
                                        weights=weight,
                                        test_type=test_type)

        test_result.results = [test_result.create_tested_example(indices[i], example)
                               for i, example in enumerate(examples)]

        if store_examples:
            test_result.examples = examples

        for fold in xrange(niterations):
            results, classifiers = self.one_fold_with_indices(learners, examples, fold, indices, preprocessors, weight)

            for example, learner, result in results:
                test_result.results[example].set_result(learner, *result)

            if store_classifiers:
                test_result.classifiers.append(classifiers)
            if callback:
                callback()

        return test_result


    def one_fold_with_indices(self, learners, examples, fold, indices, preprocessors=(), weight=0):
        """Similar to :obj:`test_with_indices` except that it performs single fold of cross-validation, given by argument ``fold``."""
        learn_set = examples.selectref(indices, fold, negate=1)
        test_set = examples.selectref(indices, fold, negate=0)
        if len(learn_set)==0 or len(test_set)==0:
            return (), ()

        # learning
        learn_set, test_set = self._preprocess_data(learn_set, test_set, preprocessors)
        if not learn_set:
            raise SystemError("no training examples after preprocessing")
        if not test_set:
            raise SystemError("no test examples after preprocessing")

        classifiers = [learner(learn_set, weight) for learner in learners]

        # testing
        testset_ids = (i for i, _ in enumerate(examples) if indices[i] == fold)
        results = self._test_on_data(classifiers, test_set, testset_ids)

        return results, classifiers


    @deprecated_keywords({"storeExamples": "store_examples",
                          "storeClassifiers": "store_classifiers",
                          "pps": "preprocessors"})
    def learn_and_test_on_learn_data(self, learners, examples, preprocessors=(),
                                     callback=None, store_classifiers=False, store_examples=False):
        """
        Train learning algorithms and test them on the same data.

        :param learners: list of learning algorithms
        :param examples: data instances used for training and testing
        :param preprocessors: a list of preprocessors (obsolete)
        :param callback: a function that is called after each learning
        :param store_classifiers: if ``True``, classifiers are stored in results
        :param store_examples: if ``True``, examples are stored in results
        :return: :obj:`ExperimentResults`
        """

        examples, weight = demangle_examples(examples)
        test_type = self.check_test_type(examples, learners)

        # If preprocessors are not used, we use the same dataset for learning and testing. Otherwise we need to
        # clone it.
        if not filter(lambda x:x[0]!="B", preprocessors):
            learn_set, test_set = self._preprocess_data(examples, Orange.data.Table(examples.domain), preprocessors)
            test_set = learn_set
        else:
            learn_set, test_set = self._preprocess_data(examples, Orange.data.Table(examples), preprocessors)

        classifiers = self._train_with_callback(learners, learn_set, weight, callback)

        test_results = ExperimentResults(1,
                                        classifier_names = [getobjectname(l) for l in learners],
                                        test_type = test_type,
                                        domain=examples.domain,
                                        weights=weight)
        test_results.results = [test_results.create_tested_example(0, example)
                               for i, example in enumerate(examples)]

        if store_classifiers:
            test_results.classifiers = classifiers
        if store_examples:
            test_results.examples = test_set

        results = self._test_on_data(classifiers, test_set)
        for example, classifier, result in results:
            test_results.results[example].set_result(classifier, *result)
        return test_results

    @deprecated_keywords({"storeExamples": "store_examples",
                          "storeClassifiers": "store_classifiers",
                          "pps": "preprocessors"})
    def learn_and_test_on_test_data(self, learners, learn_set, test_set, preprocessors=(),
                                    callback=None, store_classifiers=False, store_examples=False):
        """
        Train learning algorithms on one data sets and test them on another.

        :param learners: list of learning algorithms
        :param learn_set: training instances
        :param test_set: testing instances
        :param preprocessors: a list of preprocessors (obsolete)
        :param callback: a function that is called after each learning
        :param store_classifiers: if ``True``, classifiers are stored in results
        :param store_examples: if ``True``, examples are stored in results
        :return: :obj:`ExperimentResults`
        """
        learn_set, learn_weight = demangle_examples(learn_set)
        test_set, test_weight = demangle_examples(test_set)

        test_type = self.check_test_type(learn_set, learners)
        self.check_test_type(test_set, learners)
        
        test_results = ExperimentResults(1,
                                        classifier_names = [getobjectname(l) for l in learners],
                                        domain=test_set.domain,
                                        test_type = test_type,
                                        weights=test_weight)
        test_results.results = [test_results.create_tested_example(0, example)
                               for i, example in enumerate(test_set)]

        classifiers, results = self._learn_and_test_on_test_data(learners, learn_set, learn_weight, test_set, preprocessors, callback)

        if store_classifiers:
            test_results.classifiers = classifiers
        if store_examples:
            test_results.examples = test_set

        for example, classifier, result in results:
            test_results.results[example].set_result(classifier, *result)
        return test_results

    @deprecated_keywords({"storeExamples": "store_examples",
                          "storeClassifiers": "store_classifiers",
                          "learnProp": "learning_proportion",
                          "strat": "stratification",
                          "pps": "preprocessors",
                          "indicesrandseed": "random_generator",
                          "randseed": "random_generator",
                          "randomGenerator": "random_generator"})
    def proportion_test(self, learners, examples, learning_proportion=.7, times=10,
                   stratification=Orange.core.MakeRandomIndices.StratifiedIfPossible,
                   preprocessors=(), random_generator=0,
                   callback=None, store_classifiers=False, store_examples=False):
        """
        Iteratively split the data into training and testing set, and train and test the learnign algorithms.

        :param learners: list of learning algorithms
        :param examples: data instances used for training and testing
        :param learning_proportion: proportion of data used for training
        :param times: number of iterations
        :param stratification: use stratified sampling
        :param preprocessors: a list of preprocessors (obsolete)
        :param random_generator: random seed or generator (see above)
        :param callback: a function that is called after each fold
        :param store_classifiers: if ``True``, classifiers are stored in results
        :param store_examples: if ``True``, examples are stored in results
        :return: :obj:`ExperimentResults`
        """
        pick = Orange.core.MakeRandomIndices2(stratified = stratification, p0 = learning_proportion, randomGenerator = random_generator)

        examples, weight = demangle_examples(examples)

        test_type = self.check_test_type(examples, learners)
        
        test_results = ExperimentResults(times,
                                        classifier_names = [getobjectname(l) for l in learners],
                                        domain=examples.domain,
                                        test_type = test_type,
                                        weights=weight)
        if store_examples:
            test_results.examples = []
        test_results.classifiers = []
        offset=0
        for time in xrange(times):
            indices = pick(examples)
            learn_set = examples.selectref(indices, 0)
            test_set = examples.selectref(indices, 1)
            classifiers, results = self._learn_and_test_on_test_data(learners, learn_set, weight, test_set, preprocessors)
            if store_classifiers:
                test_results.classifiers.append(classifiers)
            if store_examples:
                test_results.examples.append(learn_set)

            test_results.results.extend(test_results.create_tested_example(time, example)
                                        for i, example in enumerate(test_set))
            for example, classifier, result in results:
                test_results.results[offset+example].set_result(classifier, *result)
            offset += len(test_set)

            if callback:
                callback()
        return test_results

    @deprecated_keywords({"storeExamples": "store_examples",
                          "storeClassifiers": "store_classifiers",
                          "learnProp": "learning_proportion",
                          "strat": "stratification",
                          "pps": "preprocessors",
                          "indicesrandseed": "random_generator",
                          "randseed": "random_generator",
                          "randomGenerator": "random_generator"})
    def learning_curve(self, learners, examples, cv_indices=None, proportion_indices=None, proportions=Orange.core.frange(0.1),
                       preprocessors=(), random_generator=0, callback=None):
        """
        Compute a learning curve using multiple cross-validations where
        models are trained on different portions of the training data.

        :param learners: list of learning algorithms
        :param examples: data instances used for training and testing
        :param cv_indices: indices used for cross validation (leave ``None`` for 10-fold CV)
        :param proportion_indices: indices for proportion selection (leave ``None`` to let the function construct the folds)
        :param proportions: list of proportions of data used for training
        :param preprocessors: a list of preprocessors (obsolete)
        :param random_generator: random seed or generator (see above)
        :param callback: a function that is be called after each learning
        :return: list of :obj:`ExperimentResults`
        """
        if cv_indices is None:
            cv_indices = Orange.core.MakeRandomIndicesCV(folds=10, stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible, randomGenerator = random_generator)
        if proportion_indices is None:
            proportion_indices = Orange.core.MakeRandomIndices2(stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible, randomGenerator = random_generator)

        examples, weight = demangle_examples(examples)
        indices = cv_indices(examples)
        
        all_results=[]
        for p in proportions:
            def select_proportion_preprocessor(examples):
                return examples.selectref(proportion_indices(examples, p0=p), 0)

            test_results = self.test_with_indices(learners, examples, indices,
                preprocessors=[("L", select_proportion_preprocessor)] +
                list(preprocessors), callback=callback)
            all_results.append(test_results)
        return all_results


    @deprecated_keywords({"strat": "stratification",
                          "pps": "preprocessors",
                          "indicesrandseed": "random_generator",
                          "randseed": "random_generator",
                          "randomGenerator": "random_generator"})
    def learning_curve_n(self, learners, examples, folds=10,
                       proportions=Orange.core.frange(0.1),
                       stratification = Orange.core.MakeRandomIndices
                       .StratifiedIfPossible,
                       preprocessors=(),
                       random_generator=0, callback=None):
        """
        Compute a learning curve using multiple cross-validations where
        models are trained on different portions of the training data.
        Similar to :obj:`learning_curve` except for simpler arguments.

        :param learners: list of learning algorithms
        :param examples: data instances used for training and testing
        :param folds: number of folds for cross-validation
        :param proportions: list of proportions of data used for training
        :param stratification: use stratified sampling
        :param preprocessors: a list of preprocessors (obsolete)
        :param random_generator: random seed or generator (see above)
        :param callback: a function that is be called after each learning
        :return: list of :obj:`ExperimentResults`
        """

        cv=Orange.core.MakeRandomIndicesCV(folds = folds,
            stratified = stratification, randomGenerator = random_generator)
        pick=Orange.core.MakeRandomIndices2(stratified = stratification,
            randomGenerator = random_generator)
        return learning_curve(learners, examples, cv, pick, proportions,
            preprocessors, callback=callback)
    
    def learning_curve_with_test_data(self, learners, learn_set, test_set,
            times=10, proportions=Orange.core.frange(0.1),
            stratification=Orange.core.MakeRandomIndices.StratifiedIfPossible,
            preprocessors=(), random_generator=0, store_classifiers=False,
            store_examples=False):
        """
        Compute a learning curve given two datasets. Models are learned on
        proportion of the first dataset and then tested on the second.

        :param learners: list of learning algorithms
        :param learn_set: training data
        :param test_set: testing data
        :param times: number of iterations
        :param straitification: use stratified sampling
        :param proportions: a list of proportions of training data to be used
        :param preprocessors: a list of preprocessors (obsolete)
        :param random_generator: random seed or generator (see above)
        :param store_classifiers: if ``True``, classifiers are stored in results
        :param store_examples: if ``True``, examples are stored in results
        :return: list of :obj:`ExperimentResults`
        """
        learn_set, learn_weight = demangle_examples(learn_set)
        test_set, test_weight = demangle_examples(test_set)
        test_type = self.check_test_type(learn_set, learners)
        self.check_test_type(test_set, learners)
        
        indices = Orange.core.MakeRandomIndices2(stratified = stratification, randomGenerator = random_generator)
        
        all_results=[]
        for p in proportions:
            test_results = ExperimentResults(times,
                                        classifier_names = [getobjectname(l) for l in learners],
                                        domain=test_set.domain,
                                        test_type = test_type,
                                        weights=test_weight)
            offset = 0
            for t in xrange(times):
                test_results.results.extend(test_results.create_tested_example(t, example)
                                            for i, example in enumerate(test_set))

                learn_examples = learn_set.selectref(indices(learn_set, p), 0)
                classifiers, results = self._learn_and_test_on_test_data\
                    (learners, learn_examples, learn_weight, test_set,
                    preprocessors=preprocessors)

                for example, classifier, result in results:
                    test_results.results[offset+example].set_result(classifier, *result)
                offset += len(test_set)

                if store_classifiers:
                    test_results.classifiers.append(classifiers)
                if store_examples:
                    test_results.examples = learn_examples

            all_results.append(test_results)
        return all_results


    def test_on_data(self, classifiers, examples, store_classifiers=False, store_examples=False):
        """
        Test classifiers on the given data

        :param classifiers: a list of classifiers
        :param examples: testing data
        :param store_classifiers: if ``True``, classifiers are stored in results
        :param store_examples: if ``True``, examples are stored in results
        """

        examples, weight = demangle_examples(examples)
        test_type = self.check_test_type(examples, classifiers)

        test_results = ExperimentResults(1,
                                        classifier_names = [getobjectname(l) for l in classifiers],
                                        domain=examples.domain,
                                        test_type = test_type,
                                        weights=weight)
        test_results.results = [test_results.create_tested_example(0, example)
                               for i, example in enumerate(examples)]

        if store_examples:
            test_results.examples = examples
        if store_classifiers:
            test_results.classifiers = classifiers

        results = self._test_on_data(classifiers, examples)
        for example, classifier, result in results:
            test_results.results[example].set_result(classifier, *result)
        return test_results


    def _learn_and_test_on_test_data(self, learners, learn_set, learn_weight, test_set,
                                     preprocessors=(), callback=None):
        learn_set, test_set = self._preprocess_data(learn_set, test_set, preprocessors)

        classifiers = self._train_with_callback(learners, learn_set, learn_weight, callback)
        
        results = self._test_on_data(classifiers, test_set)
        return classifiers, results


    def _train_with_callback(self, learners, examples, weight, callback):
        classifiers = []
        for learner in learners:
            classifier = learner(examples, weight)
            classifier.name = getattr(learner, 'name', 'noname')
            classifiers.append(classifier)
            if callback:
                callback()
        return classifiers


    def _test_on_data(self, classifiers, examples, example_ids=None):
        results = []

        if example_ids is None:
            numbered_examples = enumerate(examples)
        else:
            numbered_examples = itertools.izip(example_ids, examples)

        for e, example in numbered_examples:
            for c, classifier in enumerate(classifiers):
                # Hide actual class to prevent cheating
                ex2 = Orange.data.Instance(example)
                if ex2.domain.class_var: ex2.setclass("?")
                if ex2.domain.class_vars: ex2.set_classes(["?" for _ in ex2
                .domain.class_vars])
                result = classifier(ex2, Orange.core.GetBoth)
                results.append((e, c, result))
        return results

    
    def _preprocess_data(self, learn_set, test_set, preprocessors):
        """Apply preprocessors to learn and test dataset (obsolete)"""
        for p_type, preprocessor in preprocessors:
            if p_type == "B":
                learn_set = preprocessor(learn_set)
                test_set = preprocessor(test_set)
        for p_type, preprocessor in preprocessors:
            if p_type == "L":
                learn_set = preprocessor(learn_set)
            elif p_type == "T":
                test_set = preprocessor(test_set)
            elif p_type == "LT":
                (learn_set, test_set) = preprocessor(learn_set, test_set)
        return learn_set, test_set


    def encode_PP(self, pps):
        for pp in pps:
            objname = getobjectname(pp[1])
            if len(objname):
                pps+="_"+objname
            else:
                return "*"
        return pps

    def check_test_type(self, instances, models):
        model_is_mlc = [isinstance(m, Orange.multilabel.MultiLabelLearner) or
                        isinstance(m, Orange.multilabel.MultiLabelClassifier)
                          for m in models]
        multi_label = any(model_is_mlc)
        if multi_label and not all(model_is_mlc):
            raise ValueError("Test on mixed types of learners (MLC and non-MLC) not possible")
        multi_target = instances.domain.class_vars and not multi_label

        if (multi_label or multi_target) and not instances.domain.class_vars:
            raise ValueError("Test data with multiple labels (class vars) expected")
        if not (multi_label or multi_target or instances.domain.class_var):
            raise ValueError("Test data set without class attributes")

        return TEST_TYPE_MLC if multi_label else (
            TEST_TYPE_MULTITARGET if multi_target else TEST_TYPE_SINGLE)
    
default_evaluation = _default_evaluation = Evaluation()

cross_validation = _default_evaluation.cross_validation
leave_one_out = _default_evaluation.leave_one_out
test_with_indices = _default_evaluation.test_with_indices
one_fold_with_indices = _default_evaluation.one_fold_with_indices

learn_and_test_on_learn_data = _default_evaluation.learn_and_test_on_learn_data
learn_and_test_on_test_data = _default_evaluation.learn_and_test_on_test_data
test_on_data = _default_evaluation.test_on_data

learning_curve = _default_evaluation.learning_curve
learning_curve_n = _default_evaluation.learning_curve_n
learning_curve_with_test_data = _default_evaluation.learning_curve_with_test_data

proportion_test = _default_evaluation.proportion_test

encode_PP = _default_evaluation.encode_PP
