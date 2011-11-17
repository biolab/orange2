import itertools

import Orange
from Orange.misc import demangle_examples, getobjectname, printVerbose, deprecated_keywords


#### Some private stuff



#### Data structures

class TestedExample:
    """
    TestedExample stores predictions of different classifiers for a single testing example.

    .. attribute:: classes

        A list of predictions of type Value, one for each classifier.

    .. attribute:: probabilities

        A list of probabilities of classes, one for each classifier.

    .. attribute:: iterationNumber

        Iteration number (e.g. fold) in which the TestedExample was created/tested.

    .. attribute:: actualClass

        The correct class of the example

    .. attribute:: weight
    
        Example's weight. Even if the example set was not weighted,
        this attribute is present and equals 1.0.

    :param iterationNumber:
    :paramtype iterationNumber: type???
    :param actualClass:
    :paramtype actualClass: type???
    :param n:
    :paramtype n: int
    :param weight:
    :paramtype weight: float

    """

    def __init__(self, iterationNumber=None, actualClass=None, n=0, weight=1.0):
        self.classes = [None]*n
        self.probabilities = [None]*n
        self.iterationNumber = iterationNumber
        self.actualClass= actualClass
        self.weight = weight
    
    def add_result(self, aclass, aprob):
        """Appends a new result (class and probability prediction by a single classifier) to the classes and probabilities field."""
    
        if type(aclass.value)==float:
            self.classes.append(float(aclass))
            self.probabilities.append(aprob)
        else:
            self.classes.append(int(aclass))
            self.probabilities.append(list(aprob))

    def set_result(self, i, aclass, aprob):
        """Sets the result of the i-th classifier to the given values."""
        if type(aclass.value)==float:
            self.classes[i] = float(aclass)
            self.probabilities[i] = aprob
        else:
            self.classes[i] = int(aclass)
            self.probabilities[i] = list(aprob)

class ExperimentResults(object):
    """
    ``ExperimentResults`` stores results of one or more repetitions of
    some test (cross validation, repeated sampling...) under the same
    circumstances.

    .. attribute:: results

        A list of instances of TestedExample, one for each example in
        the dataset.

    .. attribute:: classifiers

        A list of classifiers, one element for each repetition (eg
        fold). Each element is a list of classifiers, one for each
        learner. This field is used only if storing is enabled by
        ``storeClassifiers=1``.

    .. attribute:: numberOfIterations

        Number of iterations. This can be the number of folds
        (in cross validation) or the number of repetitions of some
        test. ``TestedExample``'s attribute ``iterationNumber`` should
        be in range ``[0, numberOfIterations-1]``.

    .. attribute:: numberOfLearners

        Number of learners. Lengths of lists classes and probabilities
        in each :obj:`TestedExample` should equal ``numberOfLearners``.

    .. attribute:: loaded

        If the experimental method supports caching and there are no
        obstacles for caching (such as unknown random seeds), this is a
        list of boolean values. Each element corresponds to a classifier
        and tells whether the experimental results for that classifier
        were computed or loaded from the cache.

    .. attribute:: weights

        A flag telling whether the results are weighted. If ``False``,
        weights are still present in ``TestedExamples``, but they are
        all ``1.0``. Clear this flag, if your experimental procedure
        ran on weighted testing examples but you would like to ignore
        the weights in statistics.

    """
    def __init__(self, iterations, classifierNames, classValues=None, weights=None, baseClass=-1, domain=None, **argkw):
        self.classValues = classValues
        self.classifierNames = classifierNames
        self.numberOfIterations = iterations
        self.numberOfLearners = len(classifierNames)
        self.results = []
        self.classifiers = []
        self.loaded = None
        self.baseClass = baseClass
        self.weights = weights

        if domain is not None:
            if domain.classVar.varType == Orange.data.Type.Discrete:
                self.classValues = list(domain.classVar.values)
                self.baseClass = domain.classVar.base_value
                self.converter = int
            else:
                self.baseClass = self.classValues = None
                self.converter = float

        self.__dict__.update(argkw)

    def load_from_files(self, learners, filename):
        raise NotImplementedError("This feature is no longer supported.")

    def save_to_files(self, learners, filename):
        raise NotImplementedError("This feature is no longer supported. Pickle whole class instead.")

    def create_tested_example(self, fold, example):
        return TestedExample(fold,
                             self.converter(example.getclass()),
                             self.numberOfLearners,
                             example.getweight(self.weights))
        pass

    def remove(self, index):
        """remove one learner from evaluation results"""
        for r in self.results:
            del r.classes[index]
            del r.probabilities[index]
        del self.classifierNames[index]
        self.numberOfLearners -= 1

    def add(self, results, index, replace=-1):
        """add evaluation results (for one learner)"""
        if len(self.results)!=len(results.results):
            raise SystemError("mismatch in number of test cases")
        if self.numberOfIterations!=results.numberOfIterations:
            raise SystemError("mismatch in number of iterations (%d<>%d)" % \
                  (self.numberOfIterations, results.numberOfIterations))
        if len(self.classifiers) and len(results.classifiers)==0:
            raise SystemError("no classifiers in results")

        if replace < 0 or replace >= self.numberOfLearners: # results for new learner
            self.classifierNames.append(results.classifierNames[index])
            self.numberOfLearners += 1
            for i,r in enumerate(self.results):
                r.classes.append(results.results[i].classes[index])
                r.probabilities.append(results.results[i].probabilities[index])
            if len(self.classifiers):
                for i in range(self.numberOfIterations):
                    self.classifiers[i].append(results.classifiers[i][index])
        else: # replace results of existing learner
            self.classifierNames[replace] = results.classifierNames[index]
            for i,r in enumerate(self.results):
                r.classes[replace] = results.results[i].classes[index]
                r.probabilities[replace] = results.results[i].probabilities[index]
            if len(self.classifiers):
                for i in range(self.numberOfIterations):
                    self.classifiers[replace] = results.classifiers[i][index]

#### Experimental procedures


def proportion_test(learners, examples, learnProp, times=10,
                   strat=Orange.core.MakeRandomIndices.StratifiedIfPossible,
                   pps=[], callback=None, **argkw):
    """train-and-test evaluation (train on a subset, test on remaing examples)

    Splits the data with ``learnProp`` of examples in the learning
    and the rest in the testing set. The test is repeated for a given
    number of times (default 10). Division is stratified by default. The
    Function also accepts keyword arguments for randomization and
    storing classifiers.

    100 repetitions of the so-called 70:30 test in which 70% of examples
    are used for training and 30% for testing is done by::

        res = Orange.evaluation.testing.proportion_test(learners, data, 0.7, 100) 

    Note that Python allows naming the arguments; instead of "100" you
    can use "times=100" to increase the clarity (not so with keyword
    arguments, such as ``storeClassifiers``, ``randseed`` or ``verbose``
    that must always be given with a name).

    """
    
    # randomGenerator is set either to what users provided or to orange.RandomGenerator(0)
    # If we left it None or if we set MakeRandomIndices2.randseed, it would give same indices each time it's called
    randomGenerator = argkw.get("indicesrandseed", 0) or argkw.get("randseed", 0) or argkw.get("randomGenerator", 0)
    pick = Orange.core.MakeRandomIndices2(stratified = strat, p0 = learnProp, randomGenerator = randomGenerator)
    
    examples, weight = demangle_examples(examples)
    classVar = examples.domain.classVar
    if classVar.varType == Orange.data.Type.Discrete:
        values = list(classVar.values)
        baseValue = classVar.baseValue
    else:
        baseValue = values = None
    testResults = ExperimentResults(times, [l.name for l in learners], values, weight!=0, baseValue)

    for time in range(times):
        indices = pick(examples)
        learnset = examples.selectref(indices, 0)
        testset = examples.selectref(indices, 1)
        learn_and_test_on_test_data(learners, (learnset, weight), (testset, weight), testResults, time, pps, **argkw)
        if callback: callback()
    return testResults



def learning_curve_n(learners, examples, folds=10,
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
    return apply(learning_curve, (learners, examples, cv, pick, proportions, pps), argkw)


def learning_curve(learners, examples, cv=None, pick=None, proportions=Orange.core.frange(0.1), pps=[], **argkw):
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
    verb = argkw.get("verbose", 0)
    cache = argkw.get("cache", 0)
    callback = argkw.get("callback", 0)

    for pp in pps:
        if pp[0]!="L":
            raise SystemError("cannot preprocess testing examples")

    if not cv or not pick:    
        seed = argkw.get("indicesrandseed", -1) or argkw.get("randseed", -1)
        if seed:
            randomGenerator = Orange.core.RandomGenerator(seed)
        else:
            randomGenerator = argkw.get("randomGenerator", Orange.core.RandomGenerator())
        if not cv:
            cv = Orange.core.MakeRandomIndicesCV(folds=10, stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible, randomGenerator = randomGenerator)
        if not pick:
            pick = Orange.core.MakeRandomIndices2(stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible, randomGenerator = randomGenerator)

    examples, weight = demangle_examples(examples)
    folds = cv(examples)
    ccsum = hex(examples.checksum())[2:]
    ppsp = encode_PP(pps)
    nLrn = len(learners)

    allResults=[]
    for p in proportions:
        printVerbose("Proportion: %5.3f" % p, verb)

        if (cv.randseed<0) or (pick.randseed<0):
            cache = 0
        else:
            fnstr = "{learning_curve}_%s_%s_%s_%s%s-%s" % ("%s", p, cv.randseed, pick.randseed, ppsp, ccsum)
            if "*" in fnstr:
                cache = 0

        conv = examples.domain.classVar.varType == Orange.data.Type.Discrete and int or float
        testResults = ExperimentResults(cv.folds, [l.name for l in learners], examples.domain.classVar.values.native(), weight!=0, examples.domain.classVar.baseValue)
        testResults.results = [TestedExample(folds[i], conv(examples[i].getclass()), nLrn, examples[i].getweight(weight))
                               for i in range(len(examples))]

        if cache and testResults.load_from_files(learners, fnstr):
            printVerbose("  loaded from cache", verb)
        else:
            for fold in range(cv.folds):
                printVerbose("  fold %d" % fold, verb)
                
                # learning
                learnset = examples.selectref(folds, fold, negate=1)
                learnset = learnset.selectref(pick(learnset, p0=p), 0)
                if not len(learnset):
                    continue
                
                for pp in pps:
                    learnset = pp[1](learnset)

                classifiers = [None]*nLrn
                for i in range(nLrn):
                    if not cache or not testResults.loaded[i]:
                        classifiers[i] = learners[i](learnset, weight)

                # testing
                for i in range(len(examples)):
                    if (folds[i]==fold):
                        # This is to prevent cheating:
                        ex = Orange.data.Instance(examples[i])
                        ex.setclass("?")
                        for cl in range(nLrn):
                            if not cache or not testResults.loaded[cl]:
                                cls, pro = classifiers[cl](ex, Orange.core.GetBoth)
                                testResults.results[i].set_result(cl, cls, pro)
                if callback: callback()
            if cache:
                testResults.save_to_files(learners, fnstr)

        allResults.append(testResults)
        
    return allResults


def learning_curve_with_test_data(learners, learnset, testset, times=10,
                              proportions=Orange.core.frange(0.1),
                              strat=Orange.core.MakeRandomIndices.StratifiedIfPossible, pps=[], **argkw):
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
    verb = argkw.get("verbose", 0)

    learnset, learnweight = demangle_examples(learnset)
    testweight = demangle_examples(testset)[1]
    
    randomGenerator = argkw.get("indicesrandseed", 0) or argkw.get("randseed", 0) or argkw.get("randomGenerator", 0)
    pick = Orange.core.MakeRandomIndices2(stratified = strat, randomGenerator = randomGenerator)
    allResults=[]
    for p in proportions:
        printVerbose("Proportion: %5.3f" % p, verb)
        testResults = ExperimentResults(times, [l.name for l in learners],
                                        testset.domain.classVar.values.native(),
                                        testweight!=0, testset.domain.classVar.baseValue)
        testResults.results = []
        
        for t in range(times):
            printVerbose("  repetition %d" % t, verb)
            learn_and_test_on_test_data(learners, (learnset.selectref(pick(learnset, p), 0), learnweight),
                                   testset, testResults, t)

        allResults.append(testResults)
        
    return allResults

def learn_and_test_on_test_data(learners, learnset, testset, testResults=None, iterationNumber=0, pps=(), callback=None, **argkw):
    """
    Perform a test, where learners are learned on one dataset and tested
    on another.

    :param learners: list of learners to be tested
    :param trainset: a dataset used for training
    :param testset: a dataset used for testing
    :param preprocessors: a list of preprocessors to be used on data.
    :param callback: a function that is be called after each classifier is computed.
    :param store_classifiers: if True, classifiers will be accessible in test_results.
    :param store_examples: if True, examples will be accessible in test_results.
    """
    storeclassifiers = argkw.get("storeclassifiers", 0) or argkw.get("storeClassifiers", 0)
    storeExamples = argkw.get("storeExamples", 0)

    learnset, learnweight = demangle_examples(learnset)
    testset, testweight = demangle_examples(testset)
    storeclassifiers = argkw.get("storeclassifiers", 0) or argkw.get("storeClassifiers", 0)

    learnset, testset = preprocess_data(learnset, testset, pps)
            
    classifiers = []
    for learner in learners:
        classifiers.append(learner(learnset, learnweight))
        if callback:
            callback()
    for i in range(len(learners)):
        classifiers[i].name = getattr(learners[i], 'name', 'noname')
    testResults = test_on_data(classifiers, (testset, testweight), testResults, iterationNumber, storeExamples)
    if storeclassifiers:
        testResults.classifiers.append(classifiers)
    return testResults


def learn_and_test_on_learn_data(learners, learnset, testResults=None, iterationNumber=0, pps=[], callback=None, **argkw):
    """
    This function is similar to the above, except that it learns and
    tests on the same data. If first preprocesses the data with ``"B"``
    preprocessors on the whole data, and afterwards any ``"L"`` or ``"T"``
    preprocessors on separate datasets. Then it induces the model from
    the learning set and tests it on the testing set.

    As with :obj:`learn_and_test_on_test_data`, you can pass an already initialized
    :obj:`ExperimentResults` (argument ``results``) and an iteration number to the
    function. In this case, results of the test will be appended with
    the given iteration number.

    """

    storeclassifiers = argkw.get("storeclassifiers", 0) or argkw.get("storeClassifiers", 0)
    storeExamples = argkw.get("storeExamples", 0)

    learnset, learnweight = demangle_examples(learnset)

    hasLorT = 0    
    for pp in pps:
        if pp[0]=="B":
            learnset = pp[1](learnset)
        else:
            hasLorT = 1

    if hasLorT:
        testset = Orange.data.Table(learnset)
        for pp in pps:
            if pp[0]=="L":
                learnset = pp[1](learnset)
            elif pp[0]=="T":
                testset = pp[1](testset)
            elif pp[0]=="LT":
                learnset, testset = pp[1](learnset, testset)
    else:
        testset = learnset    

    classifiers = []
    for learner in learners:
        classifiers.append(learner(learnset, learnweight))
        if callback:
            callback()
    for i in range(len(learners)):
        classifiers[i].name = getattr(learners[i], "name", "noname")
    testResults = test_on_data(classifiers, (testset, learnweight), testResults, iterationNumber, storeExamples)
    if storeclassifiers:
        testResults.classifiers.append(classifiers)
    return testResults


def test_on_data(classifiers, testset, testResults=None, iterationNumber=0, storeExamples = False, **argkw):
    """
    This function gets a list of classifiers, not learners like the other
    functions in this module. It classifies each testing example with
    each classifier. You can pass an existing :obj:`ExperimentResults`
    and iteration number, like in :obj:`learnAndTestWithTestData`
    (which actually calls :obj:`testWithTestData`). If you don't, a new
    :obj:`ExperimentResults` will be created.

    """

    testset, testweight = demangle_examples(testset)

    if not testResults:
        classVar = testset.domain.classVar
        if testset.domain.classVar.varType == Orange.data.Type.Discrete:
            values = classVar.values.native()
            baseValue = classVar.baseValue
        else:
            values = None
            baseValue = -1
        testResults=ExperimentResults(1, [l.name for l in classifiers], values, testweight!=0, baseValue)

    examples = getattr(testResults, "examples", False)
    if examples and len(examples):
        # We must not modify an example table we do not own, so we clone it the
        # first time we have to add to it
        if not getattr(testResults, "examplesCloned", False):
            testResults.examples = Orange.data.Table(testResults.examples)
            testResults.examplesCloned = True
        testResults.examples.extend(testset)
    else:
        # We do not clone at the first iteration - cloning might never be needed at all...
        testResults.examples = testset
    
    conv = testset.domain.classVar.varType == Orange.data.Type.Discrete and int or float
    for ex in testset:
        te = TestedExample(iterationNumber, conv(ex.getclass()), 0, ex.getweight(testweight))

        for classifier in classifiers:
            # This is to prevent cheating:
            ex2 = Orange.data.Instance(ex)
            ex2.setclass("?")
            cr = classifier(ex2, Orange.core.GetBoth)
            te.add_result(cr[0], cr[1])
        testResults.results.append(te)
        
    return testResults

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
        if not examples.domain.classVar:
            raise ValueError("Test data set without class attribute")
        if "cache" in kwargs:
            raise ValueError("This feature is no longer supported.")


        niterations = max(indices)+1
        test_result = ExperimentResults(niterations,
                                        classifierNames = [getobjectname(l) for l in learners],
                                        domain=examples.domain,
                                        weights=weight)

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
        learn_set, test_set = self.preprocess_data(learn_set, test_set, preprocessors)
        if not learn_set:
            raise SystemError("no training examples after preprocessing")
        if not test_set:
            raise SystemError("no test examples after preprocessing")

        classifiers = [learner(learn_set, weight) for learner in learners]

        # testing
        testset_ids = (i for i, _ in enumerate(examples) if indices[i] == fold)
        results = self._test_on_data(classifiers, test_set, testset_ids)

        return results, classifiers


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
                ex2.setclass("?")
                result = classifier(ex2, Orange.core.GetBoth)
                results.append((e, c, result))
        return results

    def preprocess_data(self, learn_set, test_set, preprocessors):
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

preprocess_data = _default_evaluation.preprocess_data
test_with_indices = _default_evaluation.test_with_indices
one_fold_with_indices = _default_evaluation.one_fold_with_indices
cross_validation = _default_evaluation.cross_validation
leave_one_out = _default_evaluation.leave_one_out


encode_PP = _default_evaluation.encode_PP
