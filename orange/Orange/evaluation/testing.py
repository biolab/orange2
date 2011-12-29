import itertools

import Orange
from Orange.misc import demangle_examples, getobjectname, printVerbose, deprecated_keywords, deprecated_members

#### Data structures

TEST_TYPE_SINGLE = 0
TEST_TYPE_MLC = 1

class TestedExample:
    """
    TestedExample stores predictions of different classifiers for a single testing example.

    :var classes: A list of predictions of type Value, one for each classifier.
    :var probabilities: A list of probabilities of classes, one for each classifier.
    :var iterationNumber: Iteration number (e.g. fold) in which the TestedExample was created/tested.
    :var actualClass: The correct class of the example
    :var weight: Example's weight. Even if the example set was not weighted, this attribute is present and equals 1.0.
    """

    @deprecated_keywords({"iterationNumber": "iteration_number",
                          "actualClass": "actual_class"})
    def __init__(self, iteration_number=None, actual_class=None, n=0, weight=1.0):
        """
        :param iteration_number:
        :param actual_class:
        :param n:
        :param weight:
        """
        self.classes = [None]*n
        self.probabilities = [None]*n
        self.iterationNumber = iteration_number
        self.actualClass= actual_class
        self.weight = weight

    def add_result(self, aclass, aprob):
        """Appends a new result (class and probability prediction by a single classifier) to the classes and probabilities field."""
    
        if type(aclass)==list:
            self.classes.append(aclass)
            self.probabilities.append(aprob)
        elif type(aclass.value)==float:
            self.classes.append(float(aclass))
            self.probabilities.append(aprob)
        else:
            self.classes.append(int(aclass))
            self.probabilities.append(list(aprob))

    def set_result(self, i, aclass, aprob):
        """Sets the result of the i-th classifier to the given values."""
        if type(aclass)==list:
            self.classes[i] = aclass
            self.probabilities[i] = aprob
        elif type(aclass.value)==float:
            self.classes[i] = float(aclass)
            self.probabilities[i] = aprob
        else:
            self.classes[i] = int(aclass)
            self.probabilities[i] = list(aprob)

    def __repr__(self):
        return str(self.__dict__)

class ExperimentResults(object):
    """
    ``ExperimentResults`` stores results of one or more repetitions of
    some test (cross validation, repeated sampling...) under the same
    circumstances.

    :var results: A list of instances of TestedExample, one for each example in the dataset.
    :var classifiers: A list of classifiers, one element for each repetition (eg. fold). Each element is a list
      of classifiers, one for each learner. This field is used only if storing is enabled by ``storeClassifiers=1``.
    :var number_of_iterations: Number of iterations. This can be the number of folds (in cross validation)
      or the number of repetitions of some test. ``TestedExample``'s attribute ``iterationNumber`` should
      be in range ``[0, number_of_iterations-1]``.
    :var number_of_learners: Number of learners. Lengths of lists classes and probabilities in each :obj:`TestedExample`
      should equal ``number_of_learners``.
    :var loaded: If the experimental method supports caching and there are no obstacles for caching (such as unknown
      random seeds), this is a list of boolean values. Each element corresponds to a classifier and tells whether the
      experimental results for that classifier were computed or loaded from the cache.
    :var weights: A flag telling whether the results are weighted. If ``False``, weights are still present
      in ``TestedExamples``, but they are all ``1.0``. Clear this flag, if your experimental procedure ran on weighted
      testing examples but you would like to ignore the weights in statistics.
    """
    @deprecated_keywords({"classifierNames": "classifier_names",
                          "classValues": "class_values",
                          "baseClass": "base_class",
                          "numberOfIterations": "number_of_iterations",
                          "numberOfLearners": "number_of_learners"})
    def __init__(self, iterations, classifier_names, class_values=None, weights=None, base_class=-1, domain=None, test_type=TEST_TYPE_SINGLE, **argkw):
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

        if domain is not None:
            self.base_class = self.class_values = None
            if test_type==TEST_TYPE_SINGLE:
                if domain.class_var.var_type == Orange.data.Type.Discrete:
                    self.class_values = list(domain.class_var.values)
                    self.base_class = domain.class_var.base_value
                    self.converter = int
                else:
                    self.converter = float
            elif test_type==TEST_TYPE_MLC:
                self.converter = lambda vals: [int(val) if val.variable.var_type == Orange.data.Type.Discrete
                                               else float(val) for val in vals]

        self.__dict__.update(argkw)

    def load_from_files(self, learners, filename):
        raise NotImplementedError("This feature is no longer supported.")

    def save_to_files(self, learners, filename):
        raise NotImplementedError("This feature is no longer supported. Pickle whole class instead.")

    def create_tested_example(self, fold, example):
        actual = [example.getclass, example.get_classes][self.test_type]()
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

#### Experimental procedures
class Evaluation(object):
    @deprecated_keywords({"pps": "preprocessors",
                          "strat": "stratified",
                          "randseed": "random_generator",
                          "indicesrandseed": "random_generator",
                          "randomGenerator": "random_generator",
                          "storeClassifiers": "store_classifiers",
                          "storeExamples": "store_examples"})
    def cross_validation(self, learners, examples, folds=10, stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible,
                        preprocessors=(), random_generator=0, callback=None, store_classifiers=False, store_examples=False):
        """Perform cross validation with specified number of folds.

        :param learners: list of learners to be tested
        :param examples: data table on which the learners will be tested
        :param folds: number of folds to perform
        :param stratified: sets, whether indices should be stratified
        :param preprocessors: a list of preprocessors to be used on data.
        :param random_generator: random seed or random generator for selection of indices
        :param callback: a function that will be called after each fold is computed.
        :param store_classifiers: if True, classifiers will be accessible in test_results.
        :param store_examples: if True, examples will be accessible in test_results.
        """
        (examples, weight) = demangle_examples(examples)

        indices = Orange.core.MakeRandomIndicesCV(examples, folds, stratified=stratified, random_generator=random_generator)
        return self.test_with_indices(learners=learners, examples=(examples, weight), indices=indices,
                                 preprocessors=preprocessors,
                                 callback=callback, store_classifiers=store_classifiers, store_examples=store_examples)


    @deprecated_keywords({"pps": "preprocessors",
                          "storeClassifiers": "store_classifiers",
                          "storeExamples": "store_examples"})
    def leave_one_out(self, learners, examples, preprocessors=(),
                      callback=None, store_classifiers=False, store_examples=False):
        """Perform leave-one-out evaluation of learners on a data set.

        :param learners: list of learners to be tested
        :param examples: data table on which the learners will be tested
        :param preprocessors: a list of preprocessors to be used on data.
        :param callback: a function that will be called after each fold is computed.
        :param store_classifiers: if True, classifiers will be accessible in test_results.
        :param store_examples: if True, examples will be accessible in test_results.
        """
        return self.test_with_indices(learners, examples, indices=range(len(examples)), preprocessors=preprocessors,
                                 callback=callback, store_classifiers=store_classifiers, store_examples=store_examples)
    
    def check_test_type(self, instances, learners):
        learner_is_mlc = [isinstance(l, Orange.multilabel.MultiLabelLearner)
                          for l in learners]
        multi_label = any(learner_is_mlc)
        if multi_label and not all(learner_is_mlc):
            raise ValueError("Test on mixed types of learners (MLC and non-MLC) not possible")
        
        if multi_label and not instances.domain.class_vars:
            raise ValueError("Test data with multiple labels (class vars) expected")
        if not multi_label and not instances.domain.class_var:
            raise ValueError("Test data set without class attributes")
        
        return [TEST_TYPE_SINGLE, TEST_TYPE_MLC][multi_label]

    
    @deprecated_keywords({"storeExamples": "store_examples",
                          "storeClassifiers": "store_classifiers=True",
                          "pps":"preprocessors"})
    def test_with_indices(self, learners, examples, indices, preprocessors=(),
                          callback=None, store_classifiers=False, store_examples=False, **kwargs):
        """
        Perform a cross-validation-like test. Examples for each fold are selected
        based on given indices.

        :param learners: list of learners to be tested
        :param examples: data table on which the learners will be tested
        :param indices: a list of integers that defines, which examples will be
         used for testing in each fold. The number of indices should be equal to
         the number of examples.
        :param preprocessors: a list of preprocessors to be used on data.
        :param callback: a function that will be called after each fold is computed.
        :param store_classifiers: if True, classifiers will be accessible in test_results.
        :param store_examples: if True, examples will be accessible in test_results.
        """
        examples, weight = demangle_examples(examples)
        if not examples:
            raise ValueError("Test data set with no examples")
        test_type = self.check_test_type(examples, learners)
        if "cache" in kwargs:
            raise ValueError("This feature is no longer supported.")

        niterations = max(indices)+1
        test_result = ExperimentResults(niterations,
                                        classifierNames = [getobjectname(l) for l in learners],
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
        """Perform one fold of cross-validation like procedure using provided indices."""
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
        Perform a test where learners are trained and tested on the same data.

        :param learners: list of learners to be tested
        :param examples: data table on which the learners will be tested
        :param preprocessors: a list of preprocessors to be used on data.
        :param callback: a function that will be called after each fold is computed.
        :param store_classifiers: if True, classifiers will be accessible in test_results.
        :param store_examples: if True, examples will be accessible in test_results.
        """

        examples, weight = demangle_examples(examples)

        # If preprocessors are not used, we use the same dataset for learning and testing. Otherwise we need to
        # clone it.
        if not filter(lambda x:x[0]!="B", preprocessors):
            learn_set, test_set = self._preprocess_data(examples, Orange.data.Table(examples.domain), preprocessors)
            test_set = learn_set
        else:
            learn_set, test_set = self._preprocess_data(examples, Orange.data.Table(examples), preprocessors)

        classifiers = self._train_with_callback(learners, learn_set, weight, callback)

        test_results = ExperimentResults(1,
                                        classifierNames = [getobjectname(l) for l in learners],
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
        Perform a test, where learners are trained on one dataset and tested
        on another.

        :param learners: list of learners to be tested
        :param learn_set: a dataset used for training
        :param test_set: a dataset used for testing
        :param preprocessors: a list of preprocessors to be used on data.
        :param callback: a function that is be called after each classifier is computed.
        :param store_classifiers: if True, classifiers will be accessible in test_results.
        :param store_examples: if True, examples will be accessible in test_results.
        """
        learn_set, learn_weight = demangle_examples(learn_set)
        test_set, test_weight = demangle_examples(test_set)

        test_results = ExperimentResults(1,
                                        classifierNames = [getobjectname(l) for l in learners],
                                        domain=test_set.domain,
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
    def proportion_test(self, learners, examples, learning_proportion, times=10,
                   stratification=Orange.core.MakeRandomIndices.StratifiedIfPossible, preprocessors=(), random_generator=0,
                   callback=None, store_classifiers=False, store_examples=False):
        """
        Perform a test, where learners are trained and tested on different data sets. Training and test sets are
        generated by proportionally splitting examples.

        :param learners: list of learners to be tested
        :param examples: a dataset used for evaluation
        :param learning_proportion: proportion of examples to be used for training
        :param times: number of test repetitions
        :param stratification: use stratification when constructing train and test sets.
        :param preprocessors: a list of preprocessors to be used on data.
        :param callback: a function that is be called after each classifier is computed.
        :param store_classifiers: if True, classifiers will be accessible in test_results.
        :param store_examples: if True, examples will be accessible in test_results.
        """
        pick = Orange.core.MakeRandomIndices2(stratified = stratification, p0 = learning_proportion, randomGenerator = random_generator)

        examples, weight = demangle_examples(examples)

        test_results = ExperimentResults(times,
                                        classifierNames = [getobjectname(l) for l in learners],
                                        domain=examples.domain,
                                        weights=weight)
        test_results.classifiers = []
        offset=0
        for time in xrange(times):
            indices = pick(examples)
            learn_set = examples.selectref(indices, 0)
            test_set = examples.selectref(indices, 1)
            classifiers, results = self._learn_and_test_on_test_data(learners, learn_set, weight, test_set, preprocessors)
            if store_classifiers:
                test_results.classifiers.append(classifiers)

            test_results.results.extend(test_results.create_tested_example(time, example)
                                        for i, example in enumerate(test_set))
            for example, classifier, result in results:
                test_results.results[offset+example].set_result(classifier, *result)
            offset += len(test_set)

            if callback:
                callback()
        return test_results

    def learning_curve(self, learners, examples, cv=None, pick=None, proportions=Orange.core.frange(0.1),
                       preprocessors=(), random_generator=0, callback=None):
        """
        Computes learning curves using a procedure recommended by Salzberg
        (1997). It first prepares data subsets (folds). For each proportion,
        it performs the cross-validation, but taking only a proportion of
        examples for learning.

        Arguments ``cv`` and ``pick`` give the methods for preparing
        indices for cross-validation and random selection of learning
        examples. If they are not given, :obj:`orange.MakeRandomIndicesCV` and
        :obj:`orange.MakeRandomIndices2` are used, both will be stratified and the
        cross-validation will be 10-fold. Proportions is a list of proportions
        of learning examples.

        The function can save time by loading experimental existing data for
        any test that were already conducted and saved. Also, the computed
        results are stored for later use. You can enable this by adding
        a keyword argument ``cache=1``. Another keyword deals with progress
        report. If you add ``verbose=1``, the function will print the proportion
        and the fold number.

        """
        if cv is None:
            cv = Orange.core.MakeRandomIndicesCV(folds=10, stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible, randomGenerator = random_generator)
        if pick is None:
            pick = Orange.core.MakeRandomIndices2(stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible, randomGenerator = random_generator)

        examples, weight = demangle_examples(examples)
        indices = cv(examples)
        
        all_results=[]
        for p in proportions:
            def select_proportion_preprocessor(examples):
                return examples.selectref(pick(examples, p0=p), 0)

            test_results = self.test_with_indices(learners, examples, indices,
                                                  preprocessors=list(preprocessors) + [("L", select_proportion_preprocessor)],
                                                  callback=callback)
            all_results.append(test_results)
        return all_results


    def learning_curve_n(self, learners, examples, folds=10,
                       strat=Orange.core.MakeRandomIndices.StratifiedIfPossible,
                       proportions=Orange.core.frange(0.1), pps=[], **argkw):
        """Construct a learning curve for learners.

        A simpler interface for the function :obj:`learning_curve`. Instead
        of methods for preparing indices, it simply takes the number of folds
        and a flag telling whether we want a stratified cross-validation or
        not. This function does not return a single :obj:`ExperimentResults` but
        a list of them, one for each proportion. ::

            prop = [0.2, 0.4, 0.6, 0.8, 1.0]
            res = Orange.evaluation.testing.learning_curve_n(learners, data, folds = 5, proportions = prop)
            for i, p in enumerate(prop):
                print "%5.3f:" % p,
                printResults(res[i])

        This function basically prepares a random generator and example selectors
        (``cv`` and ``pick``) and calls :obj:`learning_curve`.

        """

        seed = argkw.get("indicesrandseed", -1) or argkw.get("randseed", -1)
        if seed:
            randomGenerator = Orange.core.RandomGenerator(seed)
        else:
            randomGenerator = argkw.get("randomGenerator", Orange.core.RandomGenerator())

        if strat:
            cv=Orange.core.MakeRandomIndicesCV(folds = folds, stratified = strat, randomGenerator = randomGenerator)
            pick=Orange.core.MakeRandomIndices2(stratified = strat, randomGenerator = randomGenerator)
        else:
            cv=Orange.core.RandomIndicesCV(folds = folds, stratified = strat, randomGenerator = randomGenerator)
            pick=Orange.core.RandomIndices2(stratified = strat, randomGenerator = randomGenerator)
        return learning_curve(learners, examples, cv, pick, proportions, pps, **argkw)
    
    def learning_curve_with_test_data(self, learners, learn_set, test_set, times=10,
                                  proportions=Orange.core.frange(0.1),
                                  stratification=Orange.core.MakeRandomIndices.StratifiedIfPossible, pps=[], random_generator=0):
        """
        This function is suitable for computing a learning curve on datasets,
        where learning and testing examples are split in advance. For each
        proportion of learning examples, it randomly select the requested
        number of learning examples, builds the models and tests them on the
        entire testset. The whole test is repeated for the given number of
        times for each proportion. The result is a list of :obj:`ExperimentResults`,
        one for each proportion.

        In the following scripts, examples are pre-divided onto training
        and testing set. Learning curves are computed in which 20, 40, 60,
        80 and 100 percents of the examples in the former set are used for
        learning and the latter set is used for testing. Random selection
        of the given proportion of learning set is repeated for five times.

        .. literalinclude:: code/testing-test.py
            :start-after: Learning curve with pre-separated data
            :end-before: # End


        """
        learn_set, learn_weight = demangle_examples(learn_set)
        test_set, test_weight = demangle_examples(test_set)

        indices = Orange.core.MakeRandomIndices2(stratified = stratification, randomGenerator = random_generator)
        
        all_results=[]
        for p in proportions:
            test_results = ExperimentResults(times,
                                        classifierNames = [getobjectname(l) for l in learners],
                                        domain=test_set.domain,
                                        weights=test_weight)
            offset = 0
            for t in xrange(times):
                test_results.results.extend(test_results.create_tested_example(t, example)
                                            for i, example in enumerate(test_set))

                learn_examples = learn_set.selectref(indices(learn_set, p), 0)
                classifiers, results = self._learn_and_test_on_test_data(learners, learn_examples, learn_weight, test_set)

                for example, classifier, result in results:
                    test_results.results[offset+example].set_result(classifier, *result)
                offset += len(test_set)

                test_results.classifiers.append(classifiers)

            all_results.append(test_results)
        return all_results


    def test_on_data(self, classifiers, examples, store_classifiers=False, store_examples=False):
        """
        Test classifiers on examples

        :param classifiers: classifiers to test
        :param examples: examples to test on
        :param store_classifiers: if True, classifiers will be accessible in test_results.
        :param store_examples: if True, examples will be accessible in test_results.
        """

        examples, weight = demangle_examples(examples)

        test_results = ExperimentResults(1,
                                        classifierNames = [getobjectname(l) for l in classifiers],
                                        domain=examples.domain,
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
                if ex2.domain.class_vars: ex2.set_classes(["?" for cv in ex2.domain.class_vars])
                result = classifier(ex2, Orange.core.GetBoth)
                results.append((e, c, result))
        return results

    
    def _preprocess_data(self, learn_set, test_set, preprocessors):
        """Apply preprocessors to learn and test dataset"""
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
        pps=""
        for pp in pps:
            objname = getobjectname(pp[1], "")
            if len(objname):
                pps+="_"+objname
            else:
                return "*"
        return pps
    
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
