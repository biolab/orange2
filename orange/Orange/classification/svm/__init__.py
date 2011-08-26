"""
.. index:: classification, support vector machines (SVM)

*********************************
Support vector machines (``svm``)
*********************************

A collection of classes that wrap the `LibSVM library
<http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_, a library for `support vector
machines <http://en.wikipedia.org/wiki/Support_vector_machine>`_ (SVM). In this
way SVM learners from LibSVM behave like ordinary Orange learners and can be
used as Python objects in training, classification and evaluation tasks. The
implementation supports the implementation of Python-based kernels, that can be
plugged-in into LibSVM implementations.

.. note:: On some data-sets SVM can perform very poorly. SVM can be very
          sensitive to the proper choice of the parameters. If you are having
          problems with the learner's accuracy try scaling the data and using
          different parameters or choose an easier approach and use the \
          :obj:`SVMLearnerEasy` class which does this automatically (it is
          similar to the `svm-easy.py`_ script in the LibSVM distribution).
          
SVM learners
============

.. autoclass:: Orange.classification.svm.SVMLearner
   :members:
   
.. autoclass:: Orange.classification.svm.SVMLearnerSparse
   :members:
   
.. autoclass:: Orange.classification.svm.SVMLearnerEasy
   :members:
   
.. autoclass:: Orange.classification.svm.LinearLearner
   :members:
   
Utility functions
-----------------

.. automethod:: Orange.classification.svm.maxNu

.. automethod:: Orange.classification.svm.getLinearSVMWeights

.. automethod:: Orange.classification.svm.tableToSVMFormat

SVM-derived feature weights
---------------------------

.. autoclass:: Orange.classification.svm.MeasureAttribute_SVMWeights
   :members:

.. _kernel-wrapper:


Kernel wrappers
===============

.. autoclass:: Orange.classification.svm.kernels.KernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.DualKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.RBFKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.PolyKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.AdditionKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.MultiplicationKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.CompositeKernelWrapper
   :members:

.. autoclass:: Orange.classification.svm.kernels.SparseLinKernel
   :members:

.. autoclass:: Orange.classification.svm.kernels.BagOfWords
   :members:

Example (`svm-custom-kernel.py`_ uses: `iris.tab`_)

.. literalinclude:: code/svm-custom-kernel.py



SVM-based recursive feature elimination
=======================================

.. autoclass:: Orange.classification.svm.RFE
   :members:


.. _svm-linear-weights.py: code/svm-linear-weights.py
.. _svm-custom-kernel.py: code/svm-custom-kernel.py
.. _svm-easy.py: code/svm-easy.py
.. _brown-selected.tab: code/brown-selected.tab
.. _iris.tab: code/iris.tab
.. _vehicle.tab: code/vehicle.tab
"""

import math

from collections import defaultdict

import Orange.core
import Orange.data
import kernels
import warnings

from Orange.core import SVMLearner as _SVMLearner
from Orange.core import SVMLearnerSparse as _SVMLearnerSparse
from Orange.core import LinearClassifier, \
                        LinearLearner, \
                        SVMClassifier, \
                        SVMClassifierSparse

# ORANGE Support Vector Machines
# This module was written by Ales Erjavec
# and supersedes an earlier one written by Alex Jakulin (jakulin@acm.org),
# based on: Chih-Chung Chang and Chih-Jen Lin's
# LIBSVM : a library for support vector machines
# (http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz)

#from Orange.misc import _orange__new__

def _orange__new__(base=Orange.core.Learner):
    """Return an orange 'schizofrenic' __new__ class method.
    
    :param base: base orange class (default Orange.core.Learner)
    :type base: type
         
    Example::
        class NewOrangeLearner(Orange.core.Learner):
            __new__ = _orange__new(Orange.core.Learner)
        
    """
    from functools import wraps
    @wraps(base.__new__)
    def _orange__new_wrapped(cls, data=None, **kwargs):
        self = base.__new__(cls, **kwargs)
        if data:
            self.__init__(**kwargs)
            return self.__call__(data)
        else:
            return self
    return _orange__new_wrapped

def maxNu(examples):
    """Return the maximum nu parameter for Nu_SVC support vector learning
     for the given data table. 
    
    """
    nu = 1.0
    dist = list(Orange.core.Distribution(examples.domain.classVar, examples))
    def pairs(seq):
        for i, n1 in enumerate(seq):
            for n2 in seq[i+1:]:
                yield n1, n2
    return min([2.0 * min(n1, n2) / (n1 + n2) for n1, n2 in pairs(dist) \
                if n1 != 0 and n2 !=0] + [nu])
    
class SVMLearner(_SVMLearner):
    """:param svm_type: defines the type of SVM (can be C_SVC, Nu_SVC 
        (default), OneClass, Epsilon_SVR, Nu_SVR)
    :type svm_type: SVMLearner.SVMType
    :param kernel_type: defines the type of a kernel to use for learning
        (can be kernels.RBF (default), kernels.Linear, kernels.Polynomial, 
        kernels.Sigmoid, kernels.Custom)
    :type kernel_type: kernel function, see :ref:`kernel-wrapper`
    :param degree: kernel parameter (for Polynomial) (default 3)
    :type degree: int
    :param gamma: kernel parameter (Polynomial/RBF/Sigmoid)
        (default 1/number_of_instances)
    :type gamma: float
    :param coef0: kernel parameter (Polynomial/Sigmoid) (default 0)
    :type coef0: int
    :param kernelFunc: function that will be called if `kernel_type` is
        `Custom`. It must accept two :obj:`Orange.data.Instance` arguments and
        return a float (the distance between the instances).
    :type kernelFunc: callable function
    :param C: C parameter for C_SVC, Epsilon_SVR, Nu_SVR
    :type C: float
    :param nu: Nu parameter for Nu_SVC, Nu_SVR and OneClass (default 0.5)
    :type nu: float
    :param p: epsilon in loss-function for Epsilon_SVR
    :type p: float
    :param cache_size: cache memory size in MB (default 100)
    :type cache_size: int
    :param eps: tolerance of termination criterion (default 0.001)
    :type eps: float
    :param probability: determines if a probability model should be build
        (default False)
    :type probability: bool
    :param shrinking: determines whether to use shrinking heuristics 
        (default True)
    :type shrinking: bool
    :param weights: a list of class weights
    :type weights: list
    
    """
    __new__ = _orange__new__(_SVMLearner)
    
    C_SVC = _SVMLearner.C_SVC
    Nu_SVC = _SVMLearner.Nu_SVC
    OneClass = _SVMLearner.OneClass
    Nu_SVR = _SVMLearner.Nu_SVR
    Epsilon_SVR = _SVMLearner.Epsilon_SVR
            
    def __init__(self, svm_type=Nu_SVC, kernel_type=kernels.RBF, 
                 kernelFunc=None, C=1.0, nu=0.5, p=0.1, gamma=0.0, degree=3, 
                 coef0=0, shrinking=True, probability=True, verbose=False, 
                 cache_size=200, eps=0.001, normalization=True,
                 weight=[], **kwargs):
        self.svm_type = SVMLearner.Nu_SVC
        self.kernel_type = kernel_type
        self.kernelFunc = kernelFunc
        self.C = C
        self.nu = nu
        self.p = p
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.verbose = verbose
        self.cache_size = cache_size
        self.eps = eps
        self.normalization = normalization
        for key, val in kwargs.items():
            setattr(self, key, val)
#        self.__dict__.update(kwargs)
        self.learner = Orange.core.SVMLearner(**kwargs)
        self.weight = weight

    maxNu = staticmethod(maxNu)

    def __call__(self, examples, weight=0):
        """Construct a SVM classifier
        
        :param examples: data table with continuous features
        :type examples: Orange.data.Table
        :param weight: refer to `LibSVM documentation 
            <http://http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_
        
        """
        examples = Orange.core.Preprocessor_dropMissingClasses(examples)
        if len(examples) == 0:
            raise ValueError("Example table is without any defined classes")
        if self.svm_type in [0,1] and \
        examples.domain.classVar.varType!=Orange.data.Type.Discrete:
            self.svm_type+=3
            #raise AttributeError, "Cannot learn a discrete classifier from non descrete class data. Use EPSILON_SVR or NU_SVR for regression"
        if self.svm_type in [3,4] and \
        examples.domain.classVar.varType==Orange.data.Type.Discrete:
            self.svm_type-=3
            #raise AttributeError, "Cannot do regression on descrete class data. Use C_SVC or NU_SVC for classification"
        if self.kernel_type==4 and not self.kernelFunc:
            raise AttributeError, "Custom kernel function not supplied"
        ##################################################
#        if self.kernel_type==4:     #There is a bug in svm. For some unknown reason only the probability model works with custom kernels
#            self.probability=True
        ##################################################
        nu = self.nu
        if self.svm_type == SVMLearner.Nu_SVC: #is nu feasibile
            maxNu= self.maxNu(examples)
            if self.nu > maxNu:
                if getattr(self, "verbose", 0):
                    import warnings
                    warnings.warn("Specified nu %.3f is infeasible. \
                    Setting nu to %.3f" % (self.nu, maxNu))
                nu = max(maxNu - 1e-7, 0.0)
            
        for name in ["svm_type", "kernel_type", "kernelFunc", "C", "nu", "p", 
                     "gamma", "degree", "coef0", "shrinking", "probability", 
                     "verbose", "cache_size", "eps"]:
            setattr(self.learner, name, getattr(self, name))
        self.learner.nu = nu
        self.learner.setWeights(self.weight)
        return self.learnClassifier(examples)

    def learnClassifier(self, examples):
        if self.normalization:
            examples = self._normalize(examples)
            svm = self.learner(examples)
#            if self.:
#                return SVMClassifierWrapper(svm)
#            else:
            return SVMClassifierWrapper(svm)
        return self.learner(examples)

    def tuneParameters(self, examples, parameters=None, folds=5, verbose=0, 
                       progressCallback=None):
        """Tune the parameters of the SVMLearner on given instances using 
        cross validation.
        
        :param examples: data table on which to tune the parameters
        :type examples: Orange.data.Table 
        :param parameters: if not set defaults to ["nu", "C", "gamma"]
        :type parameters: list of strings
        :param folds: number of folds used for cross validation
        :type folds: int
        :param verbose: default False
        :type verbose: bool
        :param progressCallback: report progress
        :type progressCallback: callback function
            
        Example::
            >>> svm = SVMLearner()
            >>> svm.tuneParameters(examples, parameters=["gamma"], folds=3)
            
        This code tunes the `gamma` parameter on `examples` using 3-fold cross 
        validation  
        
        """
        import orngWrap
        parameters = ["nu", "C", "gamma"] if parameters == None else parameters
        searchParams = []
        normalization = self.normalization
        if normalization:
            examples = self._normalize(examples)
            self.normalization = False
        if self.svm_type == SVMLearner.Nu_SVC and "nu" in parameters:
            numOfNuValues=9
            maxNu = max(self.maxNu(examples) - 1e-7, 0.0)
            searchParams.append(("nu", [i/10.0 for i in range(1, 9) if \
                                        i/10.0 < maxNu] + [maxNu]))
        elif "C" in parameters:
            searchParams.append(("C", [2**a for a in  range(-5,15,2)]))
        if self.kernel_type==2 and "gamma" in parameters:
            searchParams.append(("gamma", [2**a for a in range(-5,5,2)]+[0]))
        tunedLearner = orngWrap.TuneMParameters(object=self,
                            parameters=searchParams, 
                            folds=folds, 
                            returnWhat=orngWrap.TuneMParameters.returnLearner, 
                            progressCallback=progressCallback 
                            if progressCallback else lambda i:None)
        tunedLearner(examples, verbose=verbose)
        if normalization:
            self.normalization = normalization

    def _normalize(self, examples):
        dc = Orange.core.DomainContinuizer()
        dc.classTreatment = Orange.core.DomainContinuizer.Ignore
        dc.continuousTreatment = Orange.core.DomainContinuizer.NormalizeBySpan
        dc.multinomialTreatment = Orange.core.DomainContinuizer.NValues
        newdomain = dc(examples)
        return examples.translate(newdomain)

class SVMClassifierWrapper(Orange.core.SVMClassifier):
    def __new__(cls, wrapped):
        return Orange.core.SVMClassifier.__new__(cls, name=wrapped.name)
    
    def __init__(self, wrapped):
        self.wrapped = wrapped
        for name, val in wrapped.__dict__.items():
            self.__dict__[name] = val
        
    def __call__(self, example, what=Orange.core.GetValue):
        example = Orange.data.Instance(self.wrapped.domain, example)
        return self.wrapped(example, what)
    
    def classDistribution(self, example):
        example = Orange.data.Instance(self.wrapped.domain, example)
        return self.wrapped.classDistribution(example)
    
    def getDecisionValues(self, example):
        example = Orange.data.Instance(self.wrapped.domain, example)
        return self.wrapped.getDecisionValues(example)
    
    def getModel(self):
        return self.wrapped.getModel()
    
    def __reduce__(self):
        return SVMClassifierWrapper, (self.wrapped,), dict([(name, val) \
            for name, val in self.__dict__.items() \
            if name not in self.wrapped.__dict__])
            
class SVMLearnerSparse(SVMLearner):
    
    """Same as SVMLearner except that it learns from the 
        :obj:`Orange.data.Table` meta attributes.
    
    .. note:: Note that meta attributes don't need to be registered with
        the data-set domain, or present in all the instances. Use this if you
        are learning from large sparse data-sets.
    
    """
    
    def __init__(self, **kwds):
        SVMLearner.__init__(self, **kwds)
        self.learner=Orange.core.SVMLearnerSparse(**kwds)

class SVMLearnerEasy(SVMLearner):
    
    """Same as :obj:`SVMLearner` except that it will automatically scale the 
    data and perform parameter optimization using the :obj:`tuneParameters`
    method similar to the easy.py script in LibSVM package. Use this if the
    SVMLearner performs badly.
    
    Example (`svm-easy.py`_ uses: `vehicle.tab`_)
    
    .. literalinclude:: code/svm-easy.py
    
    """
    
    def __init__(self, **kwds):
        self.folds=4
        self.verbose=0
        SVMLearner.__init__(self, **kwds)
        self.learner = SVMLearner(**kwds)
        
    def learnClassifier(self, examples):
        transformer=Orange.core.DomainContinuizer()
        transformer.multinomialTreatment=Orange.core.DomainContinuizer.NValues
        transformer.continuousTreatment= \
            Orange.core.DomainContinuizer.NormalizeBySpan
        transformer.classTreatment=Orange.core.DomainContinuizer.Ignore
        newdomain=transformer(examples)
        newexamples=examples.translate(newdomain)
        #print newexamples[0]
        params={}
        parameters = []
        self.learner.normalization = False ## Normalization already done
        
        if self.svm_type in [1,4]:
            numOfNuValues=9
            if self.svm_type == SVMLearner.Nu_SVC:
                maxNu = max(self.maxNu(newexamples) - 1e-7, 0.0)
            else:
                maxNu = 1.0
            parameters.append(("nu", [i/10.0 for i in range(1, 9) \
                                      if i/10.0 < maxNu] + [maxNu]))
        else:
            parameters.append(("C", [2**a for a in  range(-5,15,2)]))
        if self.kernel_type==2:
            parameters.append(("gamma", [2**a for a in range(-5,5,2)]+[0]))
        import orngWrap
        tunedLearner = orngWrap.TuneMParameters(object=self.learner, 
                                                parameters=parameters, 
                                                folds=self.folds)
        
        return SVMClassifierWrapper(tunedLearner(newexamples,
                                                 verbose=self.verbose))

class SVMLearnerSparseClassEasy(SVMLearnerEasy, SVMLearnerSparse):
    def __init__(self, **kwds):
        SVMLearnerSparse.__init__(self, **kwds)

class LinearLearner(Orange.core.LinearLearner):
    
    """A wrapper around Orange.core.LinearLearner with a default
    solver_type == L2Loss_SVM_Dual 
    
    .. note:: The default in Orange.core.LinearLearner is L2_LR
    
    """
    
    def __new__(cls, data=None, weightId=0, **kwargs):
        self = Orange.core.LinearLearner.__new__(cls, **kwargs)
        if data:
            self.__init__(**kwargs)
            return self.__call__(data, weightId)
        else:
            return self
        
    def __init__(self, **kwargs):
        if kwargs.get("solver_type", None) in [Orange.core.LinearLearner.L2_LR, 
                                               None]:
            kwargs = dict(kwargs)
            kwargs["solver_type"] = Orange.core.LinearLearner.L2Loss_SVM_Dual
        for name, val in kwargs.items():
            setattr(self, name, val)

def getLinearSVMWeights(classifier, sum=True):
    """Extract attribute weights from the linear svm classifier.
    
    .. note:: For multi class classification the weights are square-summed
        over all binary one vs. one classifiers. If you want weights for
        each binary classifier pass `sum=False` flag (In this case the order
        of reported weights are for class1 vs class2, class1 vs class3 ... 
        class2 vs class3 ... classifiers).
        
    Example (`svm-linear-weights.py`_, uses: `brown-selected.tab`_)
        .. literalinclude:: code/svm-linear-weights.py
        
    """
    def updateWeights(w, key, val, mul):
        if key in w:
            w[key]+=mul*val
        else:
            w[key]=mul*val
            
    def to_float(val):
        return float(val) if not val.isSpecial() else 0.0 
            
    SVs=classifier.supportVectors
    weights=[]
    classes=classifier.supportVectors.domain.classVar.values
    classSV=dict([(value, filter(lambda sv: sv.getclass()==value, \
                                 classifier.supportVectors)) \
                                 for value in classes])
    svRanges=[(0, classifier.nSV[0])]
    for n in classifier.nSV[1:]:
        svRanges.append((svRanges[-1][1], svRanges[-1][1]+n))
    for i in range(len(classes)-1):
        for j in range(i+1, len(classes)):
            w={}
            coefInd=j-1
            for svInd in apply(range, svRanges[i]):
                attributes = SVs.domain.attributes + \
                SVs[svInd].getmetas(False, Orange.data.variable.Variable).keys()
                for attr in attributes:
                    if attr.varType==Orange.data.Type.Continuous:
                        updateWeights(w, attr, to_float(SVs[svInd][attr]), \
                                      classifier.coef[coefInd][svInd])
            coefInd=i
            for svInd in apply(range, svRanges[j]):
                attributes = SVs.domain.attributes + \
                SVs[svInd].getmetas(False, Orange.data.variable.Variable).keys()
                for attr in attributes:
                    if attr.varType==Orange.data.Type.Continuous:
                        updateWeights(w, attr, to_float(SVs[svInd][attr]), \
                                      classifier.coef[coefInd][svInd])
            weights.append(w)
            
    if sum:
        scores = defaultdict(float)
        
        for w in weights:
            for attr, wAttr in w.items():
                scores[attr] += wAttr**2
        for key in scores:
            scores[key] = math.sqrt(scores[key])
        return scores
    else:
        return weights

def exampleWeightedSum(example, weights):
    sum=0
    for attr, w in weights.items():
        sum+=float(example[attr])*w
    return sum
        
class MeasureAttribute_SVMWeights(Orange.core.MeasureAttribute):
    
    """Measure attribute relevance by training a linear SVM classifier on
    provided instances, using a squared sum of weights (of each binary
    classifier) as the returned measure.
        
    Example::
        >>> measure = MeasureAttribute_SVMWeights()
        >>> for attr in table.domain.attributes:
            ...   print "%15s: %.3f" % (attr.name, measure(attr, data))
          
    """
    
    def __new__(cls, attr=None, examples=None, weightId=None, **kwargs):
        self = Orange.core.MeasureAttribute.__new__(cls, **kwargs)
        if examples is not None and attr is not None:
            self.__init__(**kwargs)
            return self.__call__(attr, examples, weightId)
        else:
            return self
        
    def __reduce__(self):
        return MeasureAttribute_SVMWeights, (), {"learner": self.learner}
    
    def __init__(self, learner=None, **kwargs):
        """:param learner: Learner used for weight esstimation 
            (default LinearLearner(solver_type=L2Loss_SVM_Dual))
        :type learner: Orange.core.Learner 
        
        """
        if learner:
            self.learner = learner 
        else:
            self.learner = LinearLearner(solver_type=
                                         LinearLearner.L2Loss_SVM_Dual)
             
        self._cached_examples = None
        
    def __call__(self, attr, examples, weightId=None):
        if examples is self._cached_examples:
            weights = self._cached_weights
        else:
            classifier = self.learner(examples, weightId)
            self._cached_examples = examples
            import numpy
            weights = numpy.array(classifier.weights)
            weights = numpy.sum(weights ** 2, axis=0)
            weights = dict(zip(examples.domain.attributes, weights))
            self._cached_weights = weights
        return weights.get(attr, 0.0)

class RFE(object):
    
    """Recursive feature elimination using linear svm derived attribute 
    weights.
    
    Example::
    
        >>> rfe = RFE(SVMLearner(kernel_type=kernels.Linear, 
        normalization=False)) # normalization=False -> SVM Learner should 
        not change the domain 
        >>> data_with_removed_features = rfe(data, 5) # returns an example 
        table with only 5 best attributes
        
    """
    
    def __init__(self, learner=None):
        self.learner = learner or SVMLearner(kernel_type=
                            kernels.Linear, normalization=False)

    def getAttrScores(self, data, stopAt=0, progressCallback=None):
        """Return a dict mapping attributes to scores (scores are not scores 
        in a general meaning; they represent the step number at which they 
        were removed from the recursive evaluation).
        
        """
        iter = 1
        attrs = data.domain.attributes
        attrScores = {}
        
        while len(attrs) > stopAt:
            weights = getLinearSVMWeights(self.learner(data), sum=False)
            if progressCallback:
                progressCallback(100. * iter / (len(attrs) - stopAt))
            score = dict.fromkeys(attrs, 0)
            for w in weights:
                for attr, wAttr in w.items():
                    score[attr] += wAttr**2
            score = score.items()
            score.sort(lambda a,b:cmp(a[1],b[1]))
            numToRemove = max(int(len(attrs)*1.0/(iter+1)), 1)
            for attr, s in  score[:numToRemove]:
                attrScores[attr] = len(attrScores)
            attrs = [attr for attr, s in score[numToRemove:]]
            if attrs:
                data = data.select(attrs + [data.domain.classVar])
            iter += 1
        return attrScores
        
    def __call__(self, data, numSelected=20, progressCallback=None):
        """Return a new dataset with only `numSelected` best scoring attributes
        
        :param data: Data
        :type data: Orange.data.Table
        :param numSelected: number of features to preserve
        :type numSelected: int
        
        """
        scores = self.getAttrScores(data, progressCallback=progressCallback)
        scores = sorted(scores.items(), key=lambda item: item[1])
        
        scores = dict(scores[-numSelected:])
        attrs = [attr for attr in data.domain.attributes if attr in scores]
        domain = Orange.data.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        data = Orange.data.Table(domain, data)
        return data

def exampleTableToSVMFormat(examples, file):
    warnings.warn("Deprecated. Use tableToSVMFormat", DeprecationWarning)
    tableToSVMFormat(examples, file)

def tableToSVMFormat(examples, file):
    """Save :obj:`Orange.data.Table` to a format used by LibSVM."""
    attrs = examples.domain.attributes + examples.domain.getmetas().values()
    attrs = [attr for attr in attrs if attr.varType 
             in [Orange.data.Type.Continuous, 
                 Orange.data.Type.Discrete]]
    cv = examples.domain.classVar
    
    for ex in examples:
        if cv.varType == Orange.data.Type.Discrete:
            file.write(str(int(ex[cv])))  
        else:
            file.write(str(float(ex[cv])))
             
        for i, attr in enumerate(attrs):
            if not ex[attr].isSpecial():
                file.write(" "+str(i+1)+":"+str(ex[attr]))
        file.write("\n")
      
