"""
.. index:: Testing, Sampling

.. index:: 
   single: multilabel;  Testing for multilabel

===================================
Sampling and Testing for multilabel
===================================

This module includes functions for data sampling and splitting, and for
testing learners. It implements cross-validation, and leave-one out. All 
functions return their results in the same format - an instance of 
:obj:`ExperimentResults`, described at the end of the page. The differences 
between 'TestedExample' in multilabel and Orange/evaluation/testing module 
are that the actual classes in multilabel are list and should has no class attribute.
This object(s)  can be passed to statistical function for model evaluation
(HammingLoss, accuracy, precision, recall...) available in
module :obj:`Orange.multilabel.scoring`.

Your scripts will thus basically conduct experiments using functions in
:obj:`Orange.multilabel.testing`, covered on this page and then evaluate
the results by functions in :obj:`Orange.multilabel.scoring`. For those
interested in writing their own statistical measures of the quality of
models, description of :obj:`TestedExample` and :obj:`ExperimentResults`
are available at the end of this page.

Example scripts in this section suppose that the data is loaded and a
list of learning algorithms is prepared.

part of `ml-evaluator.py`_ (uses `multidata.tab`_)

.. literalinclude:: code/ml-evaluator.py
    :lines: 1-7

After testing is done, classification accuracies can be computed and
printed by the following function.

.. literalinclude:: code/ml-evaluator.py
    :lines: 9-10

.. _multidata.tab: code/multidata.tab
.. _ml-evaluator.py: code/ml-evaluator.py

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

*randseed (obsolete: indicesrandseed), randomGenerator*
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

*storeClassifiers (keyword argument)*
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

.. autofunction:: leave_one_out
.. autofunction:: cross_validation
.. autofunction:: test_with_indices

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
import label 

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

    .. attribute:: iterationNumber

        Iteration number (e.g. fold) in which the TestedExample was created/tested.

    .. attribute:: actualClass

        The correct class of the example

    .. attribute:: weight

        Example's weight. Even if the example set was not weighted,
        this attribute is present and equals 1.0.

    :param iterationNumber:
    :paramtype iterationNumber: int
    :param actualClass:
    :paramtype actualClass: list of :class:`Orange.data.Value`
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
        self.classes[i] = aclass
        self.probabilities[i] = aprob

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
    def __init__(self, iterations, classifierNames, classValues, weights, baseClass=-1, **argkw):
        self.classValues = classValues
        self.classifierNames = classifierNames
        self.numberOfIterations = iterations
        self.numberOfLearners = len(classifierNames)
        self.results = []
        self.classifiers = []
        self.loaded = None
        self.baseClass = baseClass
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
                    if (tre.actualClass, tre.iterationNumber) != d[ex][0]:
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
        ``((x.actualClass, x.iterationNumber), (x.classes[i],
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
                pickler.dump([(  (x.actualClass, x.iterationNumber), (x.classes[i], x.probabilities[i])  ) for x in self.results])
                f.close()

    def remove(self, index):
        """remove one learner from evaluation results"""
        for r in self.results:
            del r.classes[index]
            del r.probabilities[index]
        del self.classifierNames[index]
        self.numberOfLearners -= 1

    def add(self, results, index, replace=-1):
        """add evaluation results (for one learner)"""
        if len(self.results)<>len(results.results):
            raise SystemError, "mismatch in number of test cases"
        if self.numberOfIterations<>results.numberOfIterations:
            raise SystemError, "mismatch in number of iterations (%d<>%d)" % \
                  (self.numberOfIterations, results.numberOfIterations)
        if len(self.classifiers) and len(results.classifiers)==0:
            raise SystemError, "no classifiers in results"

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

def leave_one_out(learners, examples, pps=[], indicesrandseed="*", **argkw):

    """leave-one-out evaluation of learners on a data set

    Performs a leave-one-out experiment with the given list of learners
    and examples. This is equivalent to performing len(examples)-fold
    cross validation. Function accepts additional keyword arguments for
    preprocessing, storing classifiers and verbose output.

    """

    (examples, weight) = demangleExamples(examples)
    return test_with_indices(learners, examples, range(len(examples)), indicesrandseed, pps, **argkw)
    # return test_with_indices(learners, examples, range(len(examples)), pps=pps, argkw)

# apply(test_with_indices, (learners, (examples, weight), indices, indicesrandseed, pps), argkw)

def cross_validation(learners, examples, folds=10,
                    strat=Orange.core.MakeRandomIndices.StratifiedIfPossible,
                    pps=[], indicesrandseed="*", **argkw):
    """cross-validation evaluation of learners

    Performs a cross validation with the given number of folds.

    """
    (examples, weight) = demangleExamples(examples)
    if indicesrandseed!="*":
        indices = Orange.core.MakeRandomIndicesCV(examples, folds, randseed=indicesrandseed, stratified = strat)
    else:
        randomGenerator = argkw.get("randseed", 0) or argkw.get("randomGenerator", 0)
        indices = Orange.core.MakeRandomIndicesCV(examples, folds, stratified = strat, randomGenerator = randomGenerator)
    return test_with_indices(learners, (examples, weight), indices, indicesrandseed, pps, **argkw)


def test_with_indices(learners, examples, indices, indicesrandseed="*", pps=[], callback=None, **argkw):
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
    which was used to compute the indices (argument ``indicesrandseed``;
    if you don't there will be no caching.

    """

    verb = argkw.get("verbose", 0)
    cache = argkw.get("cache", 0)
    storeclassifiers = argkw.get("storeclassifiers", 0) or argkw.get("storeClassifiers", 0)
    cache = cache and not storeclassifiers

    examples, weight = demangleExamples(examples)
    nLrn = len(learners)

    if not examples:
        raise ValueError("Test data set with no examples")
    #TODO: check if the data is a multi-label data 
    #if not examples.domain.classVar:
    #    raise ValueError("Test data set without class attribute")
    
##    for pp in pps:
##        if pp[0]!="L":
##            raise SystemError, "cannot preprocess testing examples"

    nIterations = max(indices)+1
    basevalue = values = None
    values = label.get_label_names(examples)
    #print values
               
    testResults = ExperimentResults(nIterations, [getobjectname(l) for l in learners], values, weight!=0, basevalue)
    testResults.results = [TestedExample(indices[i], label.get_labels(examples,examples[i]), nLrn, examples[i].getweight(weight))
                           for i in range(len(examples))]

    if argkw.get("storeExamples", 0):
        testResults.examples = examples
        
    ccsum = hex(examples.checksum())[2:]
    ppsp = encode_PP(pps)
    fnstr = "{TestWithIndices}_%s_%s%s-%s" % ("%s", indicesrandseed, ppsp, ccsum)
    if "*" in fnstr:
        cache = 0

    if cache and testResults.load_from_files(learners, fnstr):
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
                if not cache or not testResults.loaded[i]:
                    classifiers[i] = learners[i](learnset, weight)
            if storeclassifiers:    
                testResults.classifiers.append(classifiers)

            # testing
            tcn = 0
            for i in range(len(examples)):
                if (indices[i]==fold):
                    # This is to prevent cheating:
                    ex = Orange.data.Instance(testset[tcn])
                    #ex.setclass("?")
                    tcn += 1
                    for cl in range(nLrn):
                        if not cache or not testResults.loaded[cl]:
                            cr = classifiers[cl](ex, Orange.core.GetBoth)                                      
                            for ilable in cr[0]: 
                                if ilable.isSpecial():
                                    raise "Classifier %s returned unknown value" % (classifiers[cl].name or ("#%i" % cl))
                            testResults.results[i].set_result(cl, cr[0], cr[1])
            if callback:
                callback()
        if cache:
            testResults.save_to_files(learners, fnstr)
        
    return testResults
