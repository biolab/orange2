"""
.. index:: Testing, Sampling

==================================
Sampling and Testing (``testing``)
==================================

This module includes functions for data sampling and splitting, and for
testing learners. It implements cross-validation, leave-one out, random
sampling and learning curves. All functions return their results in the same
format - an instance of :obj:`ExperimentResults`, described at the end of the
page, or, in case of learning curves, a list of :obj:`ExperimentResults`. This
object(s) can be passed to statistical function for model evaluation
(classification accuracy, Brier score, ROC analysis...) available in
module :obj:`Orange.evaluation.scoring`.

Your scripts will thus basically conduct experiments using functions in
:obj:`Orange.evaluation.testing`, covered on this page and then evaluate
the results by functions in :obj:`Orange.evaluation.scoring`. For those
interested in writing their own statistical measures of the quality of
models, description of :obj:`TestedExample` and :obj:`ExperimentResults`
are available at the end of this page.

.. note:: Orange has been "de-randomized". Running the same script twice
    will generally give the same results, unless special care is taken to
    randomize it. This is opposed to the previous versions where special
    care needed to be taken to make experiments repeatable. See arguments
    :obj:`randseed` and :obj:`randomGenerator` for the explanation.

Example scripts in this section suppose that the data is loaded and a
list of learning algorithms is prepared.

part of `testing-test.py`_ (uses `voting.tab`_)

.. literalinclude:: code/testing-test.py
    :start-after: import random
    :end-before: def printResults(res)

Example scripts for multi-label data.
part of `mlc-evaluator.py`_ (uses `multidata.tab`_)

.. literalinclude:: code/mlc-evaluator.py
    :lines: 1-7

After testing is done, classification accuracies can be computed and
printed by the following function.

.. literalinclude:: code/testing-test.py
    :pyobject: printResults

.. _voting.tab: code/voting.tab
.. _testing-test.py: code/testing-test.py

Common Arguments
================

Many function in this module use a set of common arguments, which we define here.

*learners*
    A list of learning algorithms. These can be either pure Orange objects
    (such as :obj:`Orange.classification.bayes.NaiveLearner`) or Python
    classes or functions written in pure Python (anything that can be
    called with the same arguments and results as Orange's classifiers
    and performs similar function).

*examples, learnset, testset*
    Examples, given as an :obj:`Orange.data.Table` (some functions need an undivided
    set of examples while others need examples that are already split
    into two sets). If examples are weighted, pass them as a tuple
    ``(examples, weightID)``. Weights are respected by learning and testing,
    but not by sampling. When selecting 10% of examples, this means 10%
    by number, not by weights. There is also no guarantee that sums
    of example weights will be (at least roughly) equal for folds in
    cross validation.

*strat*
    Tells whether to stratify the random selections. Its default value is
    :obj:`orange.StratifiedIfPossible` which stratifies selections
    if the class variable is discrete and has no unknown values.

*randseed (obsolete: indices_randseed), randomGenerator*
    Random seed (``randseed``) or random generator (``randomGenerator``) for
    random selection of examples. If omitted, random seed of 0 is used and
    the same test will always select the same examples from the example
    set. There are various slightly different ways to randomize it.

    * 
      Set ``randomGenerator`` to :obj:`orange.globalRandom`. The function's
      selection will depend upon Orange's global random generator that
      is reset (with random seed 0) when Orange is imported. The Script's
      output will therefore depend upon what you did after Orange was
      first imported in the current Python session. ::

          res = Orange.evaluation.testing.proportion_test(learners, data, 0.7,
              randomGenerator=orange.globalRandom) 

    * 
      Construct a new :obj:`orange.RandomGenerator`. The code below,
      for instance, will produce different results in each iteration,
      but overall the same results each time it's run.

      .. literalinclude:: code/testing-test.py
        :start-after: but the same each time the script is run
        :end-before: # End

    *
      Set the random seed (argument ``randseed``) to a random
      number. Python has a global random generator that is reset when
      Python is loaded, using the current system time for a seed. With this,
      results will be (in general) different each time the script is run.


      .. literalinclude:: code/testing-test.py
        :start-after: proportionsTest that will give different results each time it is run
        :end-before: # End


      The same module also provides random generators as object, so
      that you can have independent local random generators in case you
      need them.

*pps*
    A list of preprocessors. It consists of tuples ``(c, preprocessor)``,
    where ``c`` determines whether the preprocessor will be applied
    to the learning set (``"L"``), test set (``"T"``) or to both
    (``"B"``). The latter is applied first, when the example set is still
    undivided. The ``"L"`` and ``"T"`` preprocessors are applied on the
    separated subsets. Preprocessing testing examples is allowed only
    on experimental procedures that do not report the TestedExample's
    in the same order as examples in the original set. The second item
    in the tuple, preprocessor can be either a pure Orange or a pure
    Python preprocessor, that is, any function or callable class that
    accepts a table of examples and weight, and returns a preprocessed
    table and weight.

    This example will demonstrate the devastating effect of 100% class
    noise on learning. ::

        classnoise = orange.Preprocessor_addClassNoise(proportion=1.0) 
        res = Orange.evaluation.testing.proportion_test(learners, data, 0.7, 100, pps = [("L", classnoise)]) 

*proportions*
    Gives the proportions of learning examples at which the tests are
    to be made, where applicable. The default is ``[0.1, 0.2, ..., 1.0]``.

*store_classifiers (keyword argument)*
    If this flag is set, the testing procedure will store the constructed
    classifiers. For each iteration of the test (eg for each fold in
    cross validation, for each left out example in leave-one-out...),
    the list of classifiers is appended to the ExperimentResults'
    field classifiers.

    The script below makes 100 repetitions of 70:30 test and store the
    classifiers it induces. ::

        res = Orange.evaluation.testing.proportion_test(learners, data, 0.7, 100, storeClassifier=1)

*verbose (keyword argument)*
    Several functions can report their progress if you add a keyword
    argument ``verbose=1``.

Sampling and Testing Functions
==============================

.. autofunction:: proportion_test
.. autofunction:: leave_one_out
.. autofunction:: cross_validation
.. autofunction:: test_with_indices
.. autofunction:: learning_curve
.. autofunction:: learning_curve_n
.. autofunction:: learning_curve_with_test_data
.. autofunction:: learn_and_test_on_test_data
.. autofunction:: learn_and_test_on_learn_data
.. autofunction:: test_on_data

Classes
=======

Knowing classes :obj:`TestedExample` that stores results of testing
for a single test example and :obj:`ExperimentResults` that stores a list of
TestedExamples along with some other data on experimental procedures
and classifiers used, is important if you would like to write your own
measures of quality of models, compatible the sampling infrastructure
provided by Orange. If not, you can skip the remainder of this page.

.. autoclass:: TestedExample
    :members:

.. autoclass:: ExperimentResults
    :members:

References
==========

Salzberg, S. L. (1997). On comparing classifiers: Pitfalls to avoid
and a recommended approach. Data Mining and Knowledge Discovery 1,
pages 317-328.

"""

import Orange
from Orange.misc import demangleExamples, getobjectname, printVerbose
import exceptions, cPickle, os, os.path
import Orange.multilabel.label as label 

#### Some private stuff

def encode_PP(pps):
    pps=""
    for pp in pps:
        objname = getobjectname(pp[1], "")
        if len(objname):
            pps+="_"+objname
        else:
            return "*"
    return pps

#### Data structures

class TestedExample:
    """
    TestedExample stores predictions of different classifiers for a single testing example.

    .. attribute:: classes

        A list of predictions of type Value, one for each classifier.

    .. attribute:: probabilities
        
        A list of probabilities of classes, one for each classifier.

    .. attribute:: iteration_number

        Iteration number (e.g. fold) in which the TestedExample was created/tested.

    .. attribute:: actual_class

        The correct class of the example

    .. attribute:: weight

        Example's weight. Even if the example set was not weighted,
        this attribute is present and equals 1.0.
        
    .. attribute:: multilabel_flag
        
       Flag for the example to indicate whether it is a multi-label instance. If the flag is 0, it's single-label. Or else, it's multi-label.
    
    :param iteration_number:
    :paramtype iteration_number: int
    :param actual_class:
    :paramtype actual_class: :class:`Orange.data.Value` for single-label classification, and list of :class:`Orange.data.Value` for multi-label classification
    :param n:
    :paramtype n: int
    :param weight:
    :paramtype weight: float
    :param multilabel_flag:
    :paramtype multilabel_flag: int
    """

    def __init__(self, iteration_number=None, actual_class=None, n=0, weight=1.0, multilabel_flag = 0):
        self.classes = [None]*n
        self.probabilities = [None]*n
        self.iteration_number = iteration_number
        self.actual_class= actual_class
        self.weight = weight
        self.multilabel_flag = multilabel_flag
    
    def add_result(self, aclass, aprob):
        """Appends a new result (class and probability prediction by a single classifier) to the classes and probabilities field."""
        if self.multilabel_flag and type(aclass.value)==float:
            self.classes.append(float(aclass))
            self.probabilities.append(aprob)
        else:
            self.classes.append(int(aclass))
            self.probabilities.append(list(aprob))
       
    def set_result(self, i, aclass, aprob):
        """Sets the result of the i-th classifier to the given values."""
        if self.multilabel_flag == 0:
            if  type(aclass.value)==float:
                self.classes[i] = float(aclass)
                self.probabilities[i] = aprob
            else:
                self.classes[i] = int(aclass)
                self.probabilities[i] = list(aprob)
        else:
            self.classes[i] = aclass
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
        ``store_classifiers=1``.

    .. attribute:: number_of_iterations

        Number of iterations. This can be the number of folds
        (in cross validation) or the number of repetitions of some
        test. ``TestedExample``'s attribute ``iteration_number`` should
        be in range ``[0, number_of_iterations-1]``.

    .. attribute:: number_of_learners

        Number of learners. Lengths of lists classes and probabilities
        in each :obj:`TestedExample` should equal ``number_of_learners``.

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
    def __init__(self, iterations, classifier_names, class_values, weights, base_class=-1, **argkw):
        self.class_values = class_values
        self.classifier_names = classifier_names
        self.number_of_iterations = iterations
        self.number_of_learners = len(classifier_names)
        self.results = []
        self.classifiers = []
        self.loaded = None
        self.base_class = base_class
        self.weights = weights
        self.__dict__.update(argkw)

    def load_from_files(self, learners, filename):
        self.loaded = []
      
        for i in range(len(learners)):
            f = None
            try:
                f = open(".\\cache\\"+filename % getobjectname(learners[i], "*"), "rb")
                d = cPickle.load(f)
                for ex in range(len(self.results)):
                    tre = self.results[ex]
                    if (tre.actual_class, tre.iteration_number) != d[ex][0]:
                        raise SystemError, "mismatching example tables or sampling"
                    self.results[ex].set_result(i, d[ex][1][0], d[ex][1][1])
                self.loaded.append(1)
            except exceptions.Exception:
                self.loaded.append(0)
            if f:
                f.close()
                
        return not 0 in self.loaded                
                
    def save_to_files(self, learners, filename):
        """
        Saves and load testing results. ``learners`` is a list of learners and
        ``filename`` is a template for the filename. The attribute loaded is
        initialized so that it contains 1's for the learners whose data
        was loaded and 0's for learners which need to be tested. The
        function returns 1 if all the files were found and loaded,
        and 0 otherwise.

        The data is saved in a separate file for each classifier. The
        file is a binary pickle file containing a list of tuples
        ``((x.actual_class, x.iteration_number), (x.classes[i],
        x.probabilities[i]))`` where ``x`` is a :obj:`TestedExample`
        and ``i`` is the index of a learner.

        The file resides in the directory ``./cache``. Its name consists
        of a template, given by a caller. The filename should contain
        a %s which is replaced by name, shortDescription, description,
        func_doc or func_name (in that order) attribute of the learner
        (this gets extracted by orngMisc.getobjectname). If a learner
        has none of these attributes, its class name is used.

        Filename should include enough data to make sure that it
        indeed contains the right experimental results. The function
        :obj:`learning_curve`, for example, forms the name of the file
        from a string ``"{learning_curve}"``, the proportion of learning
        examples, random seeds for cross-validation and learning set
        selection, a list of preprocessors' names and a checksum for
        examples. Of course you can outsmart this, but it should suffice
        in most cases.

        """

        for i in range(len(learners)):
            if self.loaded[i]:
                continue
            
            fname=".\\cache\\"+filename % getobjectname(learners[i], "*")
            if not "*" in fname:
                if not os.path.isdir("cache"):
                    os.mkdir("cache")
                f=open(fname, "wb")
                pickler=cPickle.Pickler(f, 1)
                pickler.dump([(  (x.actual_class, x.iteration_number), (x.classes[i], x.probabilities[i])  ) for x in self.results])
                f.close()

    def remove(self, index):
        """remove one learner from evaluation results"""
        for r in self.results:
            del r.classes[index]
            del r.probabilities[index]
        del self.classifier_names[index]
        self.number_of_learners -= 1

    def add(self, results, index, replace=-1):
        """add evaluation results (for one learner)"""
        if len(self.results)<>len(results.results):
            raise SystemError, "mismatch in number of test cases"
        if self.number_of_iterations<>results.number_of_iterations:
            raise SystemError, "mismatch in number of iterations (%d<>%d)" % \
                  (self.number_of_iterations, results.number_of_iterations)
        if len(self.classifiers) and len(results.classifiers)==0:
            raise SystemError, "no classifiers in results"

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

#### Experimental procedures

def leave_one_out(learners, examples, pps=[], indices_randseed="*", **argkw):

    """leave-one-out evaluation of learners on a data set

    Performs a leave-one-out experiment with the given list of learners
    and examples. This is equivalent to performing len(examples)-fold
    cross validation. Function accepts additional keyword arguments for
    preprocessing, storing classifiers and verbose output.

    """

    (examples, weight) = demangleExamples(examples)
    return test_with_indices(learners, examples, range(len(examples)), indices_randseed, pps, **argkw)
    # return test_with_indices(learners, examples, range(len(examples)), pps=pps, argkw)

# apply(test_with_indices, (learners, (examples, weight), indices, indices_randseed, pps), argkw)


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
    arguments, such as ``store_classifiers``, ``randseed`` or ``verbose``
    that must always be given with a name).

    """
    
    # randomGenerator is set either to what users provided or to orange.RandomGenerator(0)
    # If we left it None or if we set MakeRandomIndices2.randseed, it would give same indices each time it's called
    randomGenerator = argkw.get("indices_randseed", 0) or argkw.get("randseed", 0) or argkw.get("randomGenerator", 0)
    pick = Orange.core.MakeRandomIndices2(stratified = strat, p0 = learnProp, randomGenerator = randomGenerator)
    
    examples, weight = demangleExamples(examples)
    classVar = examples.domain.classVar
    if classVar.varType == Orange.data.Type.Discrete:
        values = list(classVar.values)
        baseValue = classVar.baseValue
    else:
        baseValue = values = None
    test_results = ExperimentResults(times, [l.name for l in learners], values, weight!=0, baseValue)

    for time in range(times):
        indices = pick(examples)
        learnset = examples.selectref(indices, 0)
        testset = examples.selectref(indices, 1)
        learn_and_test_on_test_data(learners, (learnset, weight), (testset, weight), test_results, time, pps, **argkw)
        if callback: callback()
    return test_results

def cross_validation(learners, examples, folds=10,
                    strat=Orange.core.MakeRandomIndices.StratifiedIfPossible,
                    pps=[], indices_randseed="*", **argkw):
    """cross-validation evaluation of learners

    Performs a cross validation with the given number of folds.

    """
    (examples, weight) = demangleExamples(examples)
    if indices_randseed!="*":
        indices = Orange.core.MakeRandomIndicesCV(examples, folds, randseed=indices_randseed, stratified = strat)
    else:
        randomGenerator = argkw.get("randseed", 0) or argkw.get("randomGenerator", 0)
        indices = Orange.core.MakeRandomIndicesCV(examples, folds, stratified = strat, randomGenerator = randomGenerator)
    return test_with_indices(learners, (examples, weight), indices, indices_randseed, pps, **argkw)


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

    seed = argkw.get("indices_randseed", -1) or argkw.get("randseed", -1)
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
            raise SystemError, "cannot preprocess testing examples"

    if not cv or not pick:    
        seed = argkw.get("indices_randseed", -1) or argkw.get("randseed", -1)
        if seed:
            randomGenerator = Orange.core.RandomGenerator(seed)
        else:
            randomGenerator = argkw.get("randomGenerator", Orange.core.RandomGenerator())
        if not cv:
            cv = Orange.core.MakeRandomIndicesCV(folds=10, stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible, randomGenerator = randomGenerator)
        if not pick:
            pick = Orange.core.MakeRandomIndices2(stratified=Orange.core.MakeRandomIndices.StratifiedIfPossible, randomGenerator = randomGenerator)

    examples, weight = demangleExamples(examples)
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
        test_results = ExperimentResults(cv.folds, [l.name for l in learners], examples.domain.classVar.values.native(), weight!=0, examples.domain.classVar.baseValue)
        test_results.results = [TestedExample(folds[i], conv(examples[i].getclass()), nLrn, examples[i].getweight(weight))
                               for i in range(len(examples))]

        if cache and test_results.load_from_files(learners, fnstr):
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
                    if not cache or not test_results.loaded[i]:
                        classifiers[i] = learners[i](learnset, weight)

                # testing
                for i in range(len(examples)):
                    if (folds[i]==fold):
                        # This is to prevent cheating:
                        ex = Orange.data.Instance(examples[i])
                        ex.setclass("?")
                        for cl in range(nLrn):
                            if not cache or not test_results.loaded[cl]:
                                cls, pro = classifiers[cl](ex, Orange.core.GetBoth)
                                test_results.results[i].set_result(cl, cls, pro)
                if callback: callback()
            if cache:
                test_results.save_to_files(learners, fnstr)

        allResults.append(test_results)
        
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

    learnset, learnweight = demangleExamples(learnset)
    testweight = demangleExamples(testset)[1]
    
    randomGenerator = argkw.get("indices_randseed", 0) or argkw.get("randseed", 0) or argkw.get("randomGenerator", 0)
    pick = Orange.core.MakeRandomIndices2(stratified = strat, randomGenerator = randomGenerator)
    allResults=[]
    for p in proportions:
        printVerbose("Proportion: %5.3f" % p, verb)
        test_results = ExperimentResults(times, [l.name for l in learners],
                                        testset.domain.classVar.values.native(),
                                        testweight!=0, testset.domain.classVar.baseValue)
        test_results.results = []
        
        for t in range(times):
            printVerbose("  repetition %d" % t, verb)
            learn_and_test_on_test_data(learners, (learnset.selectref(pick(learnset, p), 0), learnweight),
                                   testset, test_results, t)

        allResults.append(test_results)
        
    return allResults

   
def test_with_indices(learners, examples, indices, indices_randseed="*", pps=[], callback=None, **argkw):
    """
    Performs a cross-validation-like test. The difference is that the
    caller provides indices (each index gives a fold of an example) which
    do not necessarily divide the examples into folds of (approximately)
    same sizes. In fact, the function :obj:`cross_validation` is actually written
    as a single call to ``test_with_indices``.

    ``test_with_indices`` takes care the ``TestedExamples`` are in the same order
    as the corresponding examples in the original set. Preprocessing of
    testing examples is thus not allowed. The computed results can be
    saved in files or loaded therefrom if you add a keyword argument
    ``cache=1``. In this case, you also have to specify the random seed
    which was used to compute the indices (argument ``indices_randseed``;
    if you don't there will be no caching.

    """

    verb = argkw.get("verbose", 0)
    cache = argkw.get("cache", 0)
    store_classifiers = argkw.get("store_classifiers", 0) or argkw.get("store_classifiers", 0)
    cache = cache and not store_classifiers

    examples, weight = demangleExamples(examples)
    nLrn = len(learners)

    if not examples:
        raise ValueError("Test data set with no examples")
    
    #check if the data is a multi-label data
    multilabel_flag = label.is_multilabel(examples)
    
    if multilabel_flag == 0 and not examples.domain.classVar: #single-label
        raise ValueError("Test data set without class attribute")
    
##    for pp in pps:
##        if pp[0]!="L":
##            raise SystemError, "cannot preprocess testing examples"

    nIterations = max(indices)+1
    if multilabel_flag == 0: #single-label
        if examples.domain.classVar.varType == Orange.data.Type.Discrete:
            values = list(examples.domain.classVar.values)
            basevalue = examples.domain.classVar.baseValue
        else:
            basevalue = values = None
    else: #multi-label
        values = label.get_label_names(examples)
        basevalue = None
    
    test_results = ExperimentResults(nIterations, [getobjectname(l) for l in learners], values, weight!=0, basevalue)
    if multilabel_flag == 0:
        conv = examples.domain.classVar.varType == Orange.data.Type.Discrete and int or float
        test_results.results = [TestedExample(indices[i], conv(examples[i].getclass()), nLrn, examples[i].getweight(weight))
                               for i in range(len(examples))]
    else:
        test_results.results = [TestedExample(indices[i], label.get_labels(examples,examples[i]), nLrn, examples[i].getweight(weight),multilabel_flag)
                           for i in range(len(examples))]
    
    if argkw.get("storeExamples", 0):
        test_results.examples = examples
        
    ccsum = hex(examples.checksum())[2:]
    ppsp = encode_PP(pps)
    fnstr = "{TestWithIndices}_%s_%s%s-%s" % ("%s", indices_randseed, ppsp, ccsum)
    if "*" in fnstr:
        cache = 0

    if cache and test_results.load_from_files(learners, fnstr):
        printVerbose("  loaded from cache", verb)
    else:
        for fold in range(nIterations):
            # learning
            learnset = examples.selectref(indices, fold, negate=1)
            if not len(learnset):
                continue
            testset = examples.selectref(indices, fold, negate=0)
            if not len(testset):
                continue
            
            for pp in pps:
                if pp[0]=="B":
                    learnset = pp[1](learnset)
                    testset = pp[1](testset)

            for pp in pps:
                if pp[0]=="L":
                    learnset = pp[1](learnset)
                elif pp[0]=="T":
                    testset = pp[1](testset)
                elif pp[0]=="LT":
                    (learnset, testset) = pp[1](learnset, testset)

            if not learnset:
                raise SystemError, "no training examples after preprocessing"

            if not testset:
                raise SystemError, "no test examples after preprocessing"

            classifiers = [None]*nLrn
            for i in range(nLrn):
                if not cache or not test_results.loaded[i]:
                    classifiers[i] = learners[i](learnset, weight)
            if store_classifiers:    
                test_results.classifiers.append(classifiers)

            # testing
            tcn = 0
            for i in range(len(examples)):
                if (indices[i]==fold):
                    # This is to prevent cheating:
                    ex = Orange.data.Instance(testset[tcn])
                    if multilabel_flag == 0:
                        ex.setclass("?")
                    tcn += 1
                    for cl in range(nLrn):
                        if not cache or not test_results.loaded[cl]:
                            cr = classifiers[cl](ex, Orange.core.GetBoth)                                      
                            if multilabel_flag == 0 and cr[0].isSpecial():
                                raise "Classifier %s returned unknown value" % (classifiers[cl].name or ("#%i" % cl))
                            test_results.results[i].set_result(cl, cr[0], cr[1])
            if callback:
                callback()
        if cache:
            test_results.save_to_files(learners, fnstr)
        
    return test_results


def learn_and_test_on_test_data(learners, learnset, testset, test_results=None, iteration_number=0, pps=[], callback=None, **argkw):
    """
    This function performs no sampling on its own: two separate datasets
    need to be passed, one for training and the other for testing. The
    function preprocesses the data, induces the model and tests it. The
    order of filters is peculiar, but it makes sense when compared to
    other methods that support preprocessing of testing examples. The
    function first applies preprocessors marked ``"B"`` (both sets), and only
    then the preprocessors that need to processor only one of the sets.

    You can pass an already initialized :obj:`ExperimentResults` (argument
    ``results``) and an iteration number (``iteration_number``). Results
    of the test will be appended with the given iteration
    number. This is because :obj:`learnAndTestWithTestData`
    gets called by other functions, like :obj:`proportion_test` and
    :obj:`learning_curve_with_test_data`. If you omit the parameters, a new
    :obj:`ExperimentResults` will be created.

    """
    store_classifiers = argkw.get("store_classifiers", 0) or argkw.get("store_classifiers", 0)
    storeExamples = argkw.get("storeExamples", 0)

    learnset, learnweight = demangleExamples(learnset)
    testset, testweight = demangleExamples(testset)
    store_classifiers = argkw.get("store_classifiers", 0) or argkw.get("store_classifiers", 0)
    
    for pp in pps:
        if pp[0]=="B":
            learnset = pp[1](learnset)
            testset = pp[1](testset)

    for pp in pps:
        if pp[0]=="L":
            learnset = pp[1](learnset)
        elif pp[0]=="T":
            testset = pp[1](testset)
        elif pp[0]=="LT":
            learnset, testset = pp[1](learnset, testset)
            
    classifiers = []
    for learner in learners:
        classifiers.append(learner(learnset, learnweight))
        if callback:
            callback()
    for i in range(len(learners)):
        classifiers[i].name = getattr(learners[i], 'name', 'noname')
    test_results = test_on_data(classifiers, (testset, testweight), test_results, iteration_number, storeExamples)
    if store_classifiers:
        test_results.classifiers.append(classifiers)
    return test_results


def learn_and_test_on_learn_data(learners, learnset, test_results=None, iteration_number=0, pps=[], callback=None, **argkw):
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

    store_classifiers = argkw.get("store_classifiers", 0) or argkw.get("store_classifiers", 0)
    storeExamples = argkw.get("storeExamples", 0)

    learnset, learnweight = demangleExamples(learnset)

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
    test_results = test_on_data(classifiers, (testset, learnweight), test_results, iteration_number, storeExamples)
    if store_classifiers:
        test_results.classifiers.append(classifiers)
    return test_results


def test_on_data(classifiers, testset, test_results=None, iteration_number=0, storeExamples = False, **argkw):
    """
    This function gets a list of classifiers, not learners like the other
    functions in this module. It classifies each testing example with
    each classifier. You can pass an existing :obj:`ExperimentResults`
    and iteration number, like in :obj:`learnAndTestWithTestData`
    (which actually calls :obj:`testWithTestData`). If you don't, a new
    :obj:`ExperimentResults` will be created.

    """

    testset, testweight = demangleExamples(testset)

    if not test_results:
        classVar = testset.domain.classVar
        if testset.domain.classVar.varType == Orange.data.Type.Discrete:
            values = classVar.values.native()
            baseValue = classVar.baseValue
        else:
            values = None
            baseValue = -1
        test_results=ExperimentResults(1, [l.name for l in classifiers], values, testweight!=0, baseValue)

    examples = getattr(test_results, "examples", False)
    if examples and len(examples):
        # We must not modify an example table we do not own, so we clone it the
        # first time we have to add to it
        if not getattr(test_results, "examplesCloned", False):
            test_results.examples = Orange.data.Table(test_results.examples)
            test_results.examplesCloned = True
        test_results.examples.extend(testset)
    else:
        # We do not clone at the first iteration - cloning might never be needed at all...
        test_results.examples = testset
    
    conv = testset.domain.classVar.varType == Orange.data.Type.Discrete and int or float
    for ex in testset:
        te = TestedExample(iteration_number, conv(ex.getclass()), 0, ex.getweight(testweight))

        for classifier in classifiers:
            # This is to prevent cheating:
            ex2 = Orange.data.Instance(ex)
            ex2.setclass("?")
            cr = classifier(ex2, Orange.core.GetBoth)
            te.add_result(cr[0], cr[1])
        test_results.results.append(te)
        
    return test_results
