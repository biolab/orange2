"""\
Orange unit testing
===================

This module contains some classes in common use by Orange unit testing
framework. In particular its most useful feature is the BaseTestOnData
(along with test_on_data function and expand_tests class decorators) 
class for automating tests on datasets.
  
.. todo:: read datasets to include in the run from a file (maybe command line??)

.. todo:: add a flag (maybe also datasets) parameter to test_on_data
          to override the default class member 
"""

import unittest
import os, sys

import orange
import orngTest, orngStat
import itertools
import cPickle

from Orange.preprocess import Preprocessor_discretize, Preprocessor_continuize

TEST_CLASSIFICATION = 1
TEST_REGRESSION = 2
TEST_PICKLE = 4

TEST_ALL = 7
TEST_ALL_CLASSIFICATION = TEST_ALL - TEST_REGRESSION
TEST_ALL_REGRESSION = TEST_ALL - TEST_ALL_CLASSIFICATION

TEST_CLASSLESS = 8
DISCRETIZE_DOMAIN = 16
CONTINUIZE_DOMAIN = 32

datasetsdir = os.path.join(os.path.dirname(orange.__file__), "doc", "datasets")

def openData(name, flags=0):
    dataset = orange.ExampleTable(os.path.join(datasetsdir, name))
    if flags & CONTINUIZE_DOMAIN:
        preprocessor = Preprocessor_continuize()
        dataset = preprocessor(dataset)
    elif flags & DISCRETIZE_DOMAIN:
        preprocessor = Preprocessor_discretize(orange.EquiNDiscretization())
        dataset = preprocessor(dataset)
    dataset.name = name
    return dataset

CLASSIFICATION_DATASETS = ["iris", "brown-selected", "lenses"]
REGRESSION_DATASETS = ["housing", "auto-mpg"]
CLASSLES_DATASETS =  ["water-treatment"]

def classificationDatasets(flags=0):
    """ Return an iterator over classification datasets.
    
    :param flags: `DISCRETIZE_DOMAIN`, `CONTINUIZE_DOMAIN` or 0
    :param flags: int
    
    """
    for name in CLASSIFICATION_DATASETS:
        yield openData(name, flags)
            
def regressionDatasets(flags=0):
    """ Return an iterator over regression datasets.
    
    :param flags: `DISCRETIZE_DOMAIN`, `CONTINUIZE_DOMAIN` or 0
    :type flags: int
    
    """
    for name in REGRESSION_DATASETS:
        yield openData(name, flags)
        
def classlessDatasets(flags=0):
    """ Return an iterator over classless datasets.
    
    :param flags: `DISCRETIZE_DOMAIN`, `CONTINUIZE_DOMAIN` or 0
    :type flags: int
    
    """
    for name in CLASSLES_DATASETS:
        yield openData(name, flags)
        
def datasets(flags=0):
    """ Return an iterator over classification, regression  and classless datasets.
    
    :param flags: `DISCRETIZE_DOMAIN`, `CONTINUIZE_DOMAIN`, `TEST_CLASSIFICATION`, `TEST_REGRESSION` or 0
    :type flags: int
    
    """
    datasets = []
    if flags & TEST_CLASSIFICATION:
        datasets.append(classificationDatasets(flags))
    if flags & TEST_REGRESSION:
        datasets.append(regressionDatasets(flags))
    if flags & TEST_CLASSLESS:
        datasets.append(classlessDatasets(flags))
    return itertools.chain(*datasets) 


def _expanded(func, data):
    """ Return a expanded function name and the function itself.
    """
    from functools import wraps
    @wraps(func)
    def expanded(self):
        return func(self, data)
    newname = func.__name__ + "_" + data.name.replace("-", "_")
    expanded.__name__ = newname
    return newname, expanded
                

def test_on_data(test_func):
    """ Decorator for test member of unittest.TestCase, signaling that it
    wants to be expanded (replicated) for each test data-set. The actual
    expanding is done by `expand_tests` class decorator.
    
    Example ::
    
        @test_on_data
        class Test(BaseTestOnData):
            FLAGS = TEST_CLASSIFICATION
            @test_on_data
            def test_on(self, data)
                ''' This will be a separate test case for each data-set
                instance.
                '''
                print data.name
                
    .. note:: Within the unittest framework `test_on` test will be expanded to `test_on_iris`,
              `test_on_lenses` ... for each dataset in the testing run indicating by the `FLAGS`
              class attribute
              
    ... note:: You can run individual tests from the command line ::
    
        python -m unittest mymodule.Test.test_on_iris
        
    """
    test_func._expand_data = True
    return test_func

def expand_tests(cls):
    """ A class decorator that expands BaseTestOnData subclass
    methods decorated with `test_on_data` decorator. 
    """ 
    FLAGS = getattr(cls, "FLAGS", TEST_ALL)
    for name in dir(cls):
        val = getattr(cls, name)
        if getattr(val, "_expand_data", False):
            for data in datasets(FLAGS):
                newname, expanded = _expanded(val, data)
                setattr(cls, newname, expanded)
            setattr(cls, name, None)
            setattr(cls, "_" + name, val)
    return cls
            
            
class BaseTestOnData(unittest.TestCase):
    """ Base class for tests which use data (orange example tables)
    """
    
    FLAGS = TEST_ALL
    """ A bitwise or of module level flags.
    """
    
    def classificationDatasets(self):
        return classificationDatasets(self.FLAGS)
    
    def regressionDatasets(self):
        return regressionDatasets(self.FLAGS)
    
    def classlessDatasets(self):
        return classlessDatasets(self.FLAGS)
        
    def datasets(self):
        return datasets(self.FLAGS)
    
    
class LearnerTestCase(BaseTestOnData):
    """ A basic test class for orange learner class. Must define
    class variable `LEARNER` in a subclass or define the proper
    setUp method.
    
    """ 
    
    LEARNER = None
    FLAGS = TEST_ALL_CLASSIFICATION
    
    def setUp(self):
        """ Set up the learner for the test
        """
        self.learner = self.LEARNER
        
    @test_on_data
    def test_learner_on(self, dataset):
        """ Test learner {LEARNER!r} '{NAME}' on {DATANAME}.
        """
        indices = orange.MakeRandomIndices2(p0=20)(dataset)
        learn = dataset.select(indices, 1)
        test = dataset.select(indices, 0)
        
        classifier = self.learner(learn)
        
        # Test for classVar 
        self.assertTrue(hasattr(classifier, "classVar"))
        self.assertTrue(classifier.classVar is not None)
        
        res = orngTest.testOnData([classifier], test)
        
        for ex in test:
            self.assertIsInstance(classifier(ex, orange.GetValue), orange.Value)
            self.assertIsInstance(classifier(ex, orange.GetProbabilities), orange.Distribution)
            
            value, dist = classifier(ex, orange.GetBoth)
            
            self.assertIsInstance(value, orange.Value)
            self.assertIsInstance(dist, orange.Distribution)
            
            if isinstance(dist, orange.ContDistribution):
                dist_sum = sum(dist.values())
            else:
                dist_sum = sum(dist)
                
            self.assertGreater(dist_sum, 0.0)
            self.assertLess(abs(dist_sum - 1.0), 1e-3)
            
#            # just for fun also test this
#            self.assertLess(abs(dist_sum - dist.abs), 1e-3)
#            # not fun because it fails

        # Store classifier for possible use in subclasses
        self.classifier = classifier

    def test_pickling(self):
        """ Test learner {LEARNER!r} '{NAME}' pickling.
        """
        if not self.FLAGS & TEST_PICKLE:
            return 
        
        datasets = []
        if self.FLAGS & TEST_CLASSIFICATION:
            data = iter(self.classificationDatasets()).next()
            classifier = self.learner(data)
            
            import cPickle
            s = cPickle.dumps(classifier)
            classifier_clone = cPickle.loads(s)
            
            indices = orange.MakeRandomIndices2(p0=20)(data)
            test = data.select(indices, 0)
            
            for ex in test:
                if classifier(ex, orange.GetValue) != classifier_clone(ex, orange.GetValue):
                    print classifier(ex, orange.GetBoth) , classifier_clone(ex, orange.GetBoth)
                    print classifier(ex, orange.GetValue) , classifier_clone(ex, orange.GetValue)
                self.assertEqual(classifier(ex, orange.GetValue), classifier_clone(ex, orange.GetValue), "Pickled and original classifier return a different value!")
            self.assertTrue(all(classifier(ex, orange.GetValue) == classifier_clone(ex, orange.GetValue) for ex in test))
            
    def shortDescription(self):
        """ Do some magic for prettier output on test failures.
        """
        doc = self._testMethodDoc or self._testMethodName
        return doc.format(LEARNER=getattr(self, "learner", self.LEARNER), 
                          NAME=getattr(self.LEARNER, "name", ""),
                          DATANAME=self._testMethodName.split("_on_", 1)[-1])


class TestMeasureAttribute(BaseTestOnData):
    """ Test orange MeasureAttribute subclass.
    
    .. todo:: Test if measures respect `handlesDiscrete`, `handlesContinuous`
        `computesThresholds`, `needs` (raise the appropriate exception). Test 
        `thresholdFunction`.
    """
    FLAGS = TEST_ALL_CLASSIFICATION
    MEASURE = None
    """ MEASURE must be defined in the subclass
    """
            
    @test_on_data
    def test_measure_attribute_on(self, data):
        """ Test {MEASURE!r} on {DATANAME}
        """
        scores = []
        for attr in data.domain.attributes:
            score = self.MEASURE(attr, data)
            self.assertTrue(score >= 0.0) # Can some scores be negative?
            scores.append(score)
        # any scores actually non zero
        self.assertTrue(any(score > 0.0 for score in scores))
            
        
    def test_pickle(self):
        """ Test {MEASURE!r} pickling
        """
        if self.FLAGS & TEST_PICKLE:
            import cPickle
            s = cPickle.dumps(self.MEASURE)
            measure = cPickle.loads(s)
            # TODO: make sure measure computes the same scores as measure
    
    def shortDescription(self):
        doc = self._testMethodDoc or self._testMethodName
        return doc.format(MEASURE=self.MEASURE, DATANAME=self._testMethodName.split("_on_", 1)[-1])
         

class TestPreprocessor(BaseTestOnData):
    """ Test orange.Preprocessor subclass
    
    """ 
    PREPROCESSOR = None

    @test_on_data
    def test_preprocessor_on(self, dataset):
        """ Test {PREPROCESSOR!r} on data {NAME}
        """
        newdata = self.PREPROCESSOR(dataset)
        
    def test_pickle(self):
        """ Test {PREPROCESSOR!r} pickling
        """
        if self.FLAGS & TEST_PICKLE:
            if isinstance(self.PREPROCESSOR, type):
                prep = self.PREPROCESSOR() # Test the default constructed
                s = cPickle.dumps(prep)
                prep = cPickle.loads(s)
                
            s = cPickle.dumps(self.PREPROCESSOR)
            prep = cPickle.loads(s)
            data = iter(self.datasets()).next()
            prep(data)
            
    def shortDescription(self):
        doc = self._testMethodDoc or self._testMethodName
        return doc.format(PREPROCESSOR=self.PREPROCESSOR, NAME=self._testMethodName.split("_on_", 1)[-1])
            
            
def test_case_script(path):
    """ Return a TestCase instance from a script in `path`.
    The script will be run in the directory it is in.
    
    :param path: The path to the script to test
    :type path: str
    """
    dirname = os.path.dirname(os.path.realpath(path))
    _dir = {}
    def setUp():
        _dir["cwd"] = os.path.realpath(os.curdir)
        os.chdir(dirname)
    def tearDown():
        os.chdir(_dir["cwd"])
        
    def runScript():
        execfile(path, {})
        
    runScript.__name__ = "runScript %s" % os.path.basename(path)
    return unittest.FunctionTestCase(runScript, setUp=setUp, tearDown=tearDown)


def test_suite_scripts(path):
    """ Return a TestSuite for testing all scripts in a directory `path`
    
    :param path: Directory path
    :type path: str 
    """
    import glob
    return unittest.TestSuite([test_case_script(os.path.join(path, name)) for name in glob.glob1(path, "*.py")])
    

_default_run = unittest.TestCase.run
def enable_pdb():
    """ Enable the python pdb postmortem debugger to handle any
    raised exception during the test for interactive debugging.
    
    For example you can examine excaptions in tests from ipython -pdb ::
    
        In [1]: import Orange.misc.testing as testing
        In [2]: testing.enable_pdb()
        In [3]: run tests/test_preprocessors.py
        ---...
        KernelException...
        ipdb> 
        
    .. warning:: This modifies the unittest.TestCase.run method
    
    """  
    
    def run(self, result=None):
        if result is None:
            result = self.defaultTestResult()
        result.startTest(self)
        testMethod = getattr(self, self._testMethodName)
        try:
            try:
                self.setUp()
                testMethod()
                result.addSuccess(self)
            except self.failureException:
                result.addFailure(self, self._exc_info())
            except KeyboardInterrupt:
                raise
            finally:
                self.tearDown()
        finally:
            result.stopTest(self)
            
    unittest.TestCase.run = run
    
def disable_pdb():
    """ Disables the python pdb postmortem debugger to handle
    exceptions raised during test run.
    
    """
    unittest.TestCase.run = _default_run
    
try:
    __IPYTHON__  #We are running tests from ipython
    if __IPYTHON__.shell.call_pdb: # Is pdb enabled
        enable_pdb()
except NameError:
    pass
    
    
def test_module(module):
    """ A helper function to run all tests from a module.  
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)
    runner = unittest.TextTestRunner()
    return runner.run(suite)
