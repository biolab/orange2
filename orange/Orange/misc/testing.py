"""\
Orange unit testing
===================

This module contains some classes in common use by Orange unit testing
framework. In particular its most useful feature is the BaseTestOnData
(along with ``test_on_data`` function and ``datasets_driven`` class decorators) 
class for automating data driven tests.
 
          
Example of use ::

    from Orange.misc import testing
    import unittest
    
    data = [("one", 1),
            ("two", 2)]
    
    # Data driven with data_iter argument
    # data must be reiterable multiple times if more than one test member defined
    @data_driven(data_iter=data)
    class TestDemo(unittest.TestCase):
        @test_on_data
        def test_instance_on(self, arg):
            self.assertIsInstance(arg, int)
            
        @test_on_data
        def test_add(self, arg):
            res = arg + arg
            
    # data_driven without argument
    @data_driven
    class TestDemo1(unittest.TestCase):
        @test_on_data(data_iter=data)
        def test_instance_on(self, arg):
            self.assertIsInstance(arg, int)
            
        @test_on_data(data_iter=data)
        def test_add(self, arg):
            res = arg + arg
    
    # data_driven without arg, using a static data_iter method
    @data_driven
    class TestDemo1(unittest.TestCase):
        @test_on_data
        def test_instance_on(self, arg):
            self.assertIsInstance(arg, int)
            
        @test_on_data
        def test_add(self, arg):
            res = arg + arg
            
        @staticmethod
        def data_iter():
            yield "iris", Orange.data.Table("doc:iris")
        
    #@data_driven(data_iter=testing.datasets_iter(testing.CLASSIFICATION_DATASETS | testing.CLASSLES_DATASETS))
    @datasets_driven(data_iter=testing.CLASSIFICATION_DATASETS |\
                     testing.CLASSLESS_DATASETS)
    class TestDefaultLearner(unittest.TestCase):
        @test_on_data
        def test_learner_on(self, dataset):
            import Orange
            Orange.classifcation.majority.MajorityLearner(dataset)
            
        # this overloads the class decorator's flags 
        @test_on_datasets(testing.CLASSLES_DATASETS)
        def test_raise_missing_class_on(self, dataset):
            import Orange
            Orange.classifcation.majority.MajorityLearner(dataset)
        
"""
from __future__ import absolute_import
import unittest
import os, sys

import itertools
import cPickle
from functools import partial

import orange
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

def open_data(name, flags=0):
    """ Open a named data-set return it. 
    """
    dataset = orange.ExampleTable(os.path.join(datasetsdir, name))
    if flags & CONTINUIZE_DOMAIN:
        preprocessor = Preprocessor_continuize()
        dataset = preprocessor(dataset)
    elif flags & DISCRETIZE_DOMAIN:
        preprocessor = Preprocessor_discretize(method=orange.EquiNDiscretization(),
                                               discretize_class=False)
        dataset = preprocessor(dataset)
    dataset.name = name
    return dataset

CLASSIFICATION_DATASETS = ["iris", "brown-selected", "lenses", "monks-1"]
REGRESSION_DATASETS = ["housing", "auto-mpg"]
CLASSLES_DATASETS =  ["water-treatment"]
ALL_DATASETS  = CLASSIFICATION_DATASETS + REGRESSION_DATASETS + CLASSLES_DATASETS


from collections import namedtuple
ExtraArgs = namedtuple("ExtraArgs", "args kwargs")


def _expanded(func, name, extra_args):
    """ Return an expanded function name and the function itself.
    """
    from functools import wraps
    if isinstance(extra_args, ExtraArgs):
        extra_args, extra_kwargs = extra_args
    else:
        extra_kwargs = {}
    @wraps(func)
    def expanded(*fixed_args, **fixed_kwargs):
        call = partial(partial(func, *fixed_args, **fixed_kwargs), *extra_args, **extra_kwargs)
        return call()
#    expanded = partial(func, args, kwargs)
#    expanded = wraps(func)(expanded)
    newname = func.__name__ + "_" + name.replace("-", "_")
    expanded.__name__ = newname
    return newname, expanded

def _expanded_lazy(func, name, args_getter):
    """ Return an expanded function name and the function itself.
    arge_getter must return the expanded arguments when called.
    
    """
    from functools import wraps
    @wraps(func)
    def expanded(*fixed_args, **kwargs):
        kwargs = kwargs.copy()
        extra_args = args_getter()
        if isinstance(extra_args, ExtraArgs):
            extra_args, extra_kwargs = extra_args
        else:
            extra_kwargs = {}
        call = partial(partial(func, fixed_args, kwargs), extra_args, extra_kwargs)
        return call()
    
    newname = func.__name__ + "_" + name.replace("-", "_")
    expanded.__name__ = newname
    return newname, expanded

                
def _data_driven_cls_decorator(cls, data_iter=None, lazy=False):
    """ A class decorator that expands TestCase subclass
    methods decorated with `test_on_data` or `data_driven`
    decorator.
    
    """ 
    if data_iter is None:
        data_iter = getattr(cls, "data_iter", None) # data_iter should be a staticmethod or classmethod
        if data_iter is not None:
            data_iter = data_iter()
            
    if data_iter is not None:
        data_iter = list(data_iter) # Because it needs to be iterated multiple times (each member not overriding it)
    
    for test_name in dir(cls):
        val = getattr(cls, test_name)
        if hasattr(val, "_data_iter"):
            member_data_iter = val._data_iter
            if member_data_iter is None or member_data_iter == (None, False):
                member_data_iter, lazy_iter = data_iter, lazy
            else:
                if isinstance(member_data_iter, tuple):
                    member_data_iter, lazy_iter = member_data_iter
                else:
                    lazy_iter = lazy
                    
            assert(member_data_iter is not None)
            for name, expand_args in iter(member_data_iter):
                if lazy:
                    newname, expanded = _expanded_lazy(val, name, expand_args)
                else:
                    newname, expanded = _expanded(val, name, expand_args)
                setattr(cls, newname, expanded)
            setattr(cls, test_name, None)
            setattr(cls, "__" + test_name, val)
    return cls

def data_driven(cls=None, data_iter=None):
    """ Class decorator for building data driven test cases.
    
    :param data_iter: An iterator supplying the names and arguments for
        the expanded test.
    
    Example ::
    
        data_for_tests = [("one", (1, )), ("two", (2, ))]
        
        @data_driven(data_iter=data_for_tests)
        class MyTestCase(unittest.TestCase):
            @test_on_data
            def test_add_on(self, number):
                number + number
                
    The tests are then accessible from the command line ::
    
        python -m unittest MyTestCase.MyTestCase.test_add_on_one
        
    """
    if data_iter is not None:
        #Used as
        # @data_driven(data_iter=...)
        # class ...
        return partial(_data_driven_cls_decorator, data_iter=data_iter)
    elif cls is not None:
        #Used as
        # @data_driven
        # class ...
        return _data_driven_cls_decorator(cls)
     


def data_driven_lazy(cls=None, data_iter=None):
    if lazy_data_iter is not None: 
        #Used as
        # @data_driven_lazy(data_iter= ...)
        # class ...
        return partial(_data_driven_cls_decorator, data_iter=data_iter, lazy=True)
    elif cls is not None:
        #Used as
        # @data_driven_lazy
        # class ...
        return _data_driven_cls_decorator(cls, lazy=True)
    
def test_on_data(test_func=None, data_iter=None):
    """ Decorator for test member of unittest.TestCase, signaling that it
    wants to be expanded (replicated) on each test's data case. This decorator
    accepts an optional parameter (an data case iterator, see 
    `Data Iterators`_) which overrides the iterator passed to 
    :obj:`data_driven` decorator.
    
    Example ::
    
        @data_driven
        class MyTestCase(TestCase):
            @test_on_data(datasets_iterator())
            def test_on(self, data)
                ''' This will be a separate test case for each data-set
                instance.
                '''
                print data.name
                
    .. note:: The actual expanding is done by `data_driven` class decorator.
    
    .. note:: Within the unittest framework `test_on` test will be expanded
        to `test_on_iris`, `test_on_lenses` ... for each dataset returned
        by :obj:`datasets_iterator`. You can then run individual tests from
        the command line (requires Python 2.7) ::
                   
           python -m unittest mymodule.MyTestCase.test_on_iris
    
    """
    def set_iter(func):
        func._data_iter = data_iter, False
        return func
    
    if data_iter is not None:
        return set_iter
    else:
        return set_iter(test_func)
    
    
def test_on_data_lazy(test_func=None, data_iter=None):
    """ Same as :func:`test_on_data` except the ``data_iter`` is 
    interpreted as a lazy data iterator (see `Data Iterators`_).
    
    """
    def set_iter(func):
        func._data_iter = data_iter, True
        return func
    
    if data_iter is not None:
        return set_iter
    else:
        return set_iter(test_func)
    
    
def datasets_iter(datasets=ALL_DATASETS, preprocess=0):
    for name in datasets:
        data = open_data(name, flags=preprocess)
        name = name.replace("-", "_")
        yield name, (data,)
        
        
def datasets_iter_lazy(datasets=ALL_DATASETS, preprocess=0):
    for name in datasets:
        data = lambda : (open_data(name, flags=preprocess), )
        name = name.replace("-", "_")
        yield name, data
    

def test_on_datasets(test_func=None, datasets=ALL_DATASETS):
    """ same as ``test_on_data(data_iter=datasests_iter(datasets))``
    """
    if test_func is None:
        return test_on_data(data_iter=datasets_iter(datasets))
    else:
        return test_on_data(data_iter=datasets_iter(datasets))(test_func)


def datasets_driven(cls=None, datasets=ALL_DATASETS, preprocess=0):
    """ same as ```data_driven(data_iter=datasets_iter(datasets)```
    """
    if  cls is None:
        return data_driven(data_iter=datasets_iter(datasets, preprocess))
    else:
        return data_driven(data_iter=datasets_iter(datasets, preprocess))(cls)
    

class DataTestCase(unittest.TestCase):
    """ Base class for data driven tests.
    """
    
import Orange
from Orange.evaluation import testing as _testing
from Orange.evaluation import scoring as _scoring
from Orange.core import MakeRandomIndices2 as _MakeRandomIndices2


class LearnerTestCase(DataTestCase):
    """ A basic test class for orange learner class. Must define
    class variable `LEARNER` in a subclass or define the proper
    setUp method which sets ``self.learner``.
    
    """ 
    
    LEARNER = None
    
    def setUp(self):
        """ Set up the learner for the test from the ``LEARNER`` class member.
        """
        self.learner = self.LEARNER
        
    @test_on_data
    def test_learner_on(self, dataset):
        """ Default test case for Orange learners.
        """
        indices = _MakeRandomIndices2(p0=20)(dataset)
        learn = dataset.select(indices, 1)
        test = dataset.select(indices, 0)
        
        classifier = self.learner(learn)
        
        # Test for classVar 
        self.assertTrue(hasattr(classifier, "class_var"))
        self.assertTrue(classifier.class_var is not None)
        
        res = _testing.test_on_data([classifier], test)
        
        for ex in test:
            self.assertIsInstance(classifier(ex, Orange.core.GetValue), Orange.core.Value)
            self.assertIsInstance(classifier(ex, Orange.core.GetProbabilities), Orange.core.Distribution)
            
            value, dist = classifier(ex, Orange.core.GetBoth)
            
            self.assertIsInstance(value, Orange.core.Value)
            self.assertIsInstance(dist, Orange.core.Distribution)
            
            if isinstance(dist, Orange.core.ContDistribution):
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

    @test_on_data
    def test_pickling_on(self, dataset):
        """ Test learner and classifier pickling.
        """
        classifier = self.learner(dataset)
        
        import cPickle
        s = cPickle.dumps(classifier)
        classifier_clone = cPickle.loads(s)
        
        indices = orange.MakeRandomIndices2(p0=20)(dataset)
        test = dataset.select(indices, 0)
        
        for ex in test:
            if isinstance(dataset.domain.class_var, Orange.data.variable.Continuous):
                self.assertAlmostEqual(classifier(ex, orange.GetValue).native(),
                                       classifier_clone(ex, orange.GetValue).native(),
                                       dataset.domain.class_var.number_of_decimals + 3,
                                       "Pickled and original classifier return a different value!")
            else:
                self.assertEqual(classifier(ex, orange.GetValue), classifier_clone(ex, orange.GetValue), "Pickled and original classifier return a different value!")

class MeasureAttributeTestCase(DataTestCase):
    """ Test orange MeasureAttribute subclass.
    
    .. todo:: Test if measures respect `handlesDiscrete`, `handlesContinuous`
        `computesThresholds`, `needs` (raise the appropriate exception). Test 
        `thresholdFunction`.
    """
    MEASURE = None
    """ MEASURE must be defined in the subclass
    """
            
    @test_on_data
    def test_measure_attribute_on(self, data):
        """ Default test for attribute measures.
        """
        scores = []
        for attr in data.domain.attributes:
            score = self.MEASURE(attr, data)
#            self.assertTrue(score >= 0.0)
            scores.append(score)
        # any scores actually non zero
        self.assertTrue(any(score > 0.0 for score in scores))
            
        
    def test_pickle(self):
        """ Test attribute measure pickling support.
        """
        import cPickle
        s = cPickle.dumps(self.MEASURE)
        measure = cPickle.loads(s)
        # TODO: make sure measure computes the same scores as measure
         

class PreprocessorTestCase(DataTestCase):
    """ Test orange.Preprocessor subclass
    
    """ 
    PREPROCESSOR = None

    @test_on_data
    def test_preprocessor_on(self, dataset):
        """ Test preprocessor on dataset 
        """
        newdata = self.PREPROCESSOR(dataset)
        
    def test_pickle(self):
        """ Test preprocessor pickling
        """
        if isinstance(self.PREPROCESSOR, type):
            prep = self.PREPROCESSOR() # Test the default constructed
            s = cPickle.dumps(prep)
            prep = cPickle.loads(s)
                
        s = cPickle.dumps(self.PREPROCESSOR)
        prep = cPickle.loads(s)
        
        
from Orange.distance.instances import distance_matrix
from Orange.misc import member_set

class DistanceTestCase(DataTestCase):
    """ Test orange.ExamplesDistance/Constructor
    """
    DISTANCE_CONSTRUCTOR = None
    
    def setUp(self):
        self.distance_constructor = self.DISTANCE_CONSTRUCTOR
        
    @test_on_data
    def test_distance_on(self, dataset):
        import numpy
        indices = orange.MakeRandomIndices2(dataset, min(20, len(dataset)))
        dataset = dataset.select(indices, 0)
        with member_set(self.distance_constructor, "ignore_class", True):
            mat = distance_matrix(dataset, self.distance_constructor)
        
        m = numpy.array(list(mat))
        self.assertTrue((m >= 0.0).all())
        
        if dataset.domain.class_var:
            with member_set(self.distance_constructor, "ignore_class", False):
                mat = distance_matrix(dataset, self.distance_constructor)
            m1 = numpy.array(list(mat))
            self.assertTrue((m1 != m).all() or dataset, "%r does not seem to respect the 'ignore_class' flag")
        
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
    
    For example you can examine exceptions in tests from ipython -pdb ::
    
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
#            except self.failureException:
#                result.addFailure(self, self._exc_info())
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
    if getattr(__IPYTHON__.shell, "call_pdb", None): # Is pdb enabled
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
