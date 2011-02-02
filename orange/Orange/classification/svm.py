""" 
.. index:: svm

=======================
Support Vector Machines
=======================

.. index:: Support Vector Machines Classification

Interface to the LibSVM library (LIBSVM : a library for support vector machines
(http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz)

.. note:: On some data-sets SVM can perform very badly. It is a known fact that
          SVM's can be very sensitive to the proper choice of the parameters.
          If you are having problems with learner's accuracy try scaling the
          data and using different parameters or choose an easier approach
          and use the `SVMLearnerEasy` class which does this automatically
          (it is similar to the easy.py script in the LibSVM distribution).

.. autoclass:: Orange.classify.svm.SVMLearner
   :members:
   
.. autoclass:: Orange.classify.svm.SVMLearnerSparse
   :members:
   
.. autoclass:: Orange.classify.svm.SVMLearnerEasy
   :members:
   
Usefull functions
=================

.. automethod:: Orange.classify.svm.maxNu

.. automethod:: Orange.classify.svm.getLinearSVMWeights

.. automethod:: Orange.classify.svm.exampleTableToSVMFormat


Kernel Wrappers
---------------

.. autoclass:: Orange.classify.svm.KernelWrapper
   :members:

.. autoclass:: Orange.classify.svm.DualKernelWrapper
   :members:

.. autoclass:: Orange.classify.svm.RBFKernelWrapper
   :members:

.. autoclass:: Orange.classify.svm.PolyKernelWrapper
   :members:

.. autoclass:: Orange.classify.svm.AdditionKernelWrapper
   :members:

.. autoclass:: Orange.classify.svm.MultiplicationKernelWrapper
   :members:

.. autoclass:: Orange.classify.svm.CompositeKernelWrapper
   :members:

.. autoclass:: Orange.classify.svm.SparseLinKernel
   :members:

.. autoclass:: Orange.classify.svm.BagOfWords
   :members:

Example (`svm-custom-kernel.py`_ uses: `iris.tab`_)

.. literalinclude:: code/svm-custom-kernel.py


SVM derived feature weights
---------------------------

.. autoclass:: Orange.classify.svm.MeasureAttribute_SVMWeights
   :members:


SVM based Recursive Feature Elimination
---------------------------------------

.. autoclass:: Orange.classify.svm.RFE
   :members:


.. _svm-linear-weights.py: code/svm-linear-weights.py
.. _svm-custom-kernel.py: code/svm-custom-kernel.py
.. _svm-easy.py: code/svm-easy.py
.. _brown-selected.tab: code/brown-selected.tab
.. _iris.tab: code/iris.tab
.. _vehicle.tab: code/vehicle.tab

"""

import orange
from orange import SVMLearner as _SVMLearner
from orange import SVMLearnerSparse as _SVMLearnerSparse
from orange import KernelFunc
from orange import LinearClassifier, \
         LinearLearner, \
         SVMClassifier, \
              SVMClassifierSparse
              
Linear = _SVMLearner.Linear
Polynomial = _SVMLearner.Polynomial
RBF = _SVMLearner.RBF
Sigmoid = _SVMLearner.Sigmoid
Custom = _SVMLearner.Custom

C_SVC = _SVMLearner.C_SVC
Nu_SVC = _SVMLearner.Nu_SVC
OneClass = _SVMLearner.OneClass
Nu_SVR = _SVMLearner.Nu_SVR
Epsilon_SVR = _SVMLearner.Epsilon_SVR


# ORANGE Support Vector Machines
# This module was written by Ales Erjavec
# and supersedes an earlier one written by Alex Jakulin (jakulin@acm.org),
# based on: Chih-Chung Chang and Chih-Jen Lin's
# LIBSVM : a library for support vector machines
#  (http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz)

#from Orange.misc import _orange__new__

def _orange__new__(base=orange.Learner):
    """ Return an orange 'schizofrenic' __new__ class method.
    
    :param base: base orange class (default orange.Learner)
    :type base: type
         
    Example::
        class NewOrangeLearner(orange.Learner):
            __new__ = _orange__new(orange.Learner)
        
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

import orange
def maxNu(examples):
    """ Return the maximum nu parameter for Nu_SVC support vector learning
     for the given example table. 
    """
    nu = 1.0
    dist = list(orange.Distribution(examples.domain.classVar, examples))
    def pairs(seq):
        for i, n1 in enumerate(seq):
            for n2 in seq[i+1:]:
                yield n1, n2
    return min([2.0 * min(n1, n2) / (n1 + n2) for n1, n2 in pairs(dist) if n1 != 0 and n2 !=0] + [nu])

class SVMLearner(_SVMLearner):
    """ 
    """
    __new__ = _orange__new__(_SVMLearner)
        
    def __init__(self, svm_type=Nu_SVC, kernel_type=RBF, kernelFunc=None, C=1.0,
                 nu=0.5, p=0.1, gamma=0.0, degree=3, coef0=0, shrinking=True,
                 probability=True, verbose = False, cache_size=200, eps=0.001,
                 normalization = True, weight = [], **kwargs):
        """
        :param svm_type: Defines the type of SVM (can be C_SVC, Nu_SVC (default),
            OneClass, Epsilon_SVR, Nu_SVR)
        :type svm_type: SVMLearner.SVMType
        :param kernel_type: Defines the type of a kernel to use for learning
            (can be RBF (default), Linear, Polynomial, Sigmoid, Custom)
        :type kernel_type: SVMLearner.Kernel
        :param degree: Kernel parameter (for Polynomial) (default 3)
        :type degree: int
        :param gamma: Kernel parameter (Polynomial / RBF/Sigmoid)
            (default 1/number_of_examples)
        :type gamma: float
        :param coef0: Kernel parameter (Polynomial/Sigmoid) (default 0)
        :type coef0: int
        :param kernelFunc: Function that will be called if `kernel_type` is
            `Custom`. It must accept two `Orange.data.Example` arguments and
            return a float (the distance between the examples).
        :type kernelFunc: callable function
        :param C: C parameter for C_SVC, Epsilon_SVR, Nu_SVR
        :type C: float
        :param nu: Nu parameter for Nu_SVC, Nu_SVR and OneClass (default 0.5)
        :type nu: float
        :param p: Epsilon in loss-function for Epsilon_SVR
        :type p: float
        :param cache_size: Cache memory size in MB (default 100)
        :type cache_size: int
        :param eps: Tolerance of termination criterion (default 0.001)
        :type eps: float
        :param probability: Determines if a probability model should be build
            (default False)
        :type probability: bool
        :param shrinking: Determines whether to use shrinking heuristics 
            (default True)
        :type shrinking: bool
        :param weights: a list of class weights
        :type weights: list
        
        """
        self.svm_type=orange.SVMLearner.Nu_SVC
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
        self.learner = orange.SVMLearner(**kwargs)
        self.weight = weight

    maxNu = staticmethod(maxNu)

    def __call__(self, examples, weight=0):
        examples = orange.Preprocessor_dropMissingClasses(examples)
        if len(examples) == 0:
            raise ValueError("Example table is without any defined classes")
        if self.svm_type in [0,1] and examples.domain.classVar.varType!=orange.VarTypes.Discrete:
            self.svm_type+=3
            #raise AttributeError, "Cannot learn a discrete classifier from non descrete class data. Use EPSILON_SVR or NU_SVR for regression"
        if self.svm_type in [3,4] and examples.domain.classVar.varType==orange.VarTypes.Discrete:
            self.svm_type-=3
            #raise AttributeError, "Cannot do regression on descrete class data. Use C_SVC or NU_SVC for classification"
        if self.kernel_type==4 and not self.kernelFunc:
            raise AttributeError, "Custom kernel function not supplied"
        ##################################################
#        if self.kernel_type==4:     #There is a bug in svm. For some unknown reason only the probability model works with custom kernels
#            self.probability=True
        ##################################################
        nu = self.nu
        if self.svm_type == orange.SVMLearner.Nu_SVC: #check nu feasibility
            maxNu= self.maxNu(examples)
            if self.nu > maxNu:
                if getattr(self, "verbose", 0):
                    import warnings
                    warnings.warn("Specified nu %.3f is infeasible. Setting nu to %.3f" % (self.nu, maxNu))
                nu = max(maxNu - 1e-7, 0.0)
            
        for name in ["svm_type", "kernel_type", "kernelFunc", "C", "nu", "p", "gamma", "degree",
                "coef0", "shrinking", "probability", "verbose", "cache_size", "eps"]:
            setattr(self.learner, name, getattr(self, name))
        self.learner.nu = nu
        self.learner.setWeights(self.weight)
        return self.learnClassifier(examples)

    def learnClassifier(self, examples):
        if self.normalization:
            examples = self._normalize(examples)
            return SVMClassifierWrapper(self.learner(examples), examples.domain)
        return self.learner(examples)

    def tuneParameters(self, examples, parameters=None, folds=5, verbose=0, progressCallback=None):
        """ Tune the parameters of the SVMLearner on given examples using cross validation.
        
        :param examples: ExampleTable on which to tune the parameters 
        :param parameters: if not set defaults to ["nu", "C", "gamma"]
        :param folds: number of folds used for cross validation
        :param verbose:
        :param progressCallback: a callback function to report progress
            
        Example::
            >>> svm = SVMLearner()
            >>> svm.tuneParameters(examples, parameters=["gamma"], folds=3)
            
        This code tunes the `gamma` parameter on `examples` using 3-fold cross validation  
        
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
            searchParams.append(("nu", [i/10.0 for i in range(1, 9) if i/10.0 < maxNu] + [maxNu]))
        elif "C" in parameters:
            searchParams.append(("C", [2**a for a in  range(-5,15,2)]))
        if self.kernel_type==2 and "gamma" in parameters:
            searchParams.append(("gamma", [2**a for a in range(-5,5,2)]+[0]))
        tunedLearner = orngWrap.TuneMParameters(object=self, parameters=searchParams, folds=folds, 
                                                returnWhat=orngWrap.TuneMParameters.returnLearner, 
                                                progressCallback=progressCallback if progressCallback else lambda i:None)
        tunedLearner(examples, verbose=verbose)
        if normalization:
            self.normalization = normalization

    def _normalize(self, examples):
        dc = orange.DomainContinuizer()
        dc.classTreatment = orange.DomainContinuizer.Ignore
        dc.continuousTreatment = orange.DomainContinuizer.NormalizeBySpan
        dc.multinomialTreatment = orange.DomainContinuizer.NValues
        newdomain = dc(examples)
        return examples.translate(newdomain)

class SVMClassifierWrapper(object):
    
    def __init__(self, classifier=None, domain=None):
        self.classifier = classifier
        self.domain = domain
        
    def __getattr__(self, name):
        try:
            return getattr(self.__dict__["classifier"], name)
        except (KeyError, AttributeError):
            raise AttributeError(name)

    def __call__(self, example, what=orange.GetValue):
        example = orange.Example(self.domain, example)
        return self.classifier(example, what)
        
class SVMLearnerSparse(SVMLearner):
    """ Same as SVMLearner except that it learns from the examples meta
    attributes.
    
    .. note:: Note that meta attributes don't need to be registered with
        the data-set domain, or present in all the examples. Use this if you
        are learning from large sparse data-sets.
    
    """
    def __init__(self, **kwds):
        SVMLearner.__init__(self, **kwds)
        self.learner=orange.SVMLearnerSparse(**kwds)

    
class SVMLearnerEasy(SVMLearner):
    """ Same as `SVMLearner` except that it will automatically scale the data
    and perform parameter optimization using the `tuneParameters` method
    similar to the easy.py script in LibSVM package. Use this if the
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
        transformer=orange.DomainContinuizer()
        transformer.multinomialTreatment=orange.DomainContinuizer.NValues
        transformer.continuousTreatment=orange.DomainContinuizer.NormalizeBySpan
        transformer.classTreatment=orange.DomainContinuizer.Ignore
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
            parameters.append(("nu", [i/10.0 for i in range(1, 9) if i/10.0 < maxNu] + [maxNu]))
        else:
            parameters.append(("C", [2**a for a in  range(-5,15,2)]))
        if self.kernel_type==2:
            parameters.append(("gamma", [2**a for a in range(-5,5,2)]+[0]))
        import orngWrap
        tunedLearner = orngWrap.TuneMParameters(object=self.learner, parameters=parameters, folds=self.folds)
        
        return SVMClassifierClassEasyWrapper(tunedLearner(newexamples, verbose=self.verbose), newdomain, examples)

class SVMLearnerSparseClassEasy(SVMLearnerEasy, SVMLearnerSparse):
    def __init__(self, **kwds):
        SVMLearnerSparse.__init__(self, **kwds)
        
class SVMClassifierClassEasyWrapper:
    def __init__(self, classifier, domain=None, oldexamples=None):
        self.classifier=classifier
        self.domain=domain
        self.oldexamples=oldexamples
    def __call__(self,example, getBoth=orange.GetValue):
        example=orange.ExampleTable([example]).translate(self.domain)[0] #orange.Example(self.domain, example)
        return self.classifier(example, getBoth)
    def __getattr__(self, name):
        if name in ["supportVectors", "nSV", "coef", "rho", "examples", "kernelFunc"]:
            return getattr(self.__dict__["classifier"], name)
        else:
            raise AttributeError(name)
    def __setstate__(self, state):
        print state
        self.__dict__.update(state)
        transformer=orange.DomainContinuizer()
        transformer.multinominalTreatment=orange.DomainContinuizer.NValues
        transformer.continuousTreatment=orange.DomainContinuizer.NormalizeBySpan
        transformer.classTreatment=orange.DomainContinuizer.Ignore
        print self.examples
        self.domain=transformer(self.oldexamples)

from collections import defaultdict

def getLinearSVMWeights(classifier, sum=True):
    """ Extract attribute weights from the linear svm classifier.
    
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
    classSV=dict([(value, filter(lambda sv: sv.getclass()==value, classifier.supportVectors)) for value in classes])
    svRanges=[(0, classifier.nSV[0])]
    for n in classifier.nSV[1:]:
        svRanges.append((svRanges[-1][1], svRanges[-1][1]+n))
    for i in range(len(classes)-1):
        for j in range(i+1, len(classes)):
            w={}
            coefInd=j-1
            for svInd in apply(range, svRanges[i]):
                for attr in SVs.domain.attributes+SVs[svInd].getmetas(False, orange.Variable).keys():
                    if attr.varType==orange.VarTypes.Continuous:
                        updateWeights(w, attr, to_float(SVs[svInd][attr]), classifier.coef[coefInd][svInd])
            coefInd=i
            for svInd in apply(range, svRanges[j]):
                for attr in SVs.domain.attributes+SVs[svInd].getmetas(False, orange.Variable).keys():
                    if attr.varType==orange.VarTypes.Continuous:
                        updateWeights(w, attr, to_float(SVs[svInd][attr]), classifier.coef[coefInd][svInd])
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

import math
class KernelWrapper(object):
    """ A base class for kernel function wrappers 
    """
    def __init__(self, wrapped):
        """
        :param wrapped: a function to wrap
        
        """
        self.wrapped=wrapped
    def __call__(self, example1, example2):
        return self.wrapped(example1, example2)

class DualKernelWrapper(KernelWrapper):
    """ A base class for kernel wrapper that wrap two other kernel functions.
    """
    def __init__(self, wrapped1, wrapped2):
        """
        :param wrapped1:
        :param wrapped2:
        
        """
        self.wrapped1=wrapped1
        self.wrapped2=wrapped2
        
class RBFKernelWrapper(KernelWrapper):
    """ A Kernel wrapper that uses a wrapped kernel function in a RBF
    (Radial Basis Function)
    
    """
    def __init__(self, wrapped, gamma=0.5):
        """
        :param wrapped: a function to wrap
        :param gamma: the gamma of the RBF
        
        """
        KernelWrapper.__init__(self, wrapped)
        self.gamma=gamma
        
    def __call__(self, example1, example2):
        """ Return 
        
            .. math::  exp(-gamma * wrapped(example1, example2) ^ 2)
        
        """
        return math.exp(-self.gamma*math.pow(self.wrapped(example1, example2),2))

class PolyKernelWrapper(KernelWrapper):
    def __init__(self, wrapped, degree=3.0):
        """
        :param wrapped: a function to wrap
        :param degree: degree of the polinomial
        
        """
        KernelWrapper.__init__(self, wrapped)
        self.degree=degree
    def __call__(self, example1, example2):
        """ Return
        
            .. math:: wrapped(example1, example2) ^ d
        
        """
        return math.pow(self.wrapped(example1, example2), self.degree)

class AdditionKernelWrapper(DualKernelWrapper):
    def __call__(self, example1, example2):
        """ Return
        
            .. math:: wrapped1(example1, example2) + wrapped2(example1, example2)
            
        """
        return self.wrapped1(example1, example2)+self.wrapped2(example1, example2)

class MultiplicationKernelWrapper(DualKernelWrapper):
    def __call__(self, example1, example2):
        """ Return
        
            .. math:: wrapped1(example1, example2) * wrapped2(example1, example2)
            
        """
        return self.wrapped1(example1, example2)*self.wrapped2(example1, example2)

class CompositeKernelWrapper(DualKernelWrapper):
    def __init__(self, wrapped1, wrapped2, l=0.5):
        DualKernelWrapper.__init__.__doc__ + """\
        :param l:
        
        """
        DualKernelWrapper.__init__(self, wrapped1, wrapped2)
        self.l=l
    def __call__(self, example1, example2):
        """ Return
        
            .. math:: l * wrapped1(example1, example2) + (1 - l) * wrapped2(example1, example2)
            
        """
        return self.l*self.wrapped1(example1, example2) + (1-self.l)*self.wrapped2(example1,example2)

class SparseLinKernel(object):
    def __call__(self, example1, example2):
        """ Computes a linear kernel function using the examples meta attributes (need to be floats)
        """
        s=set(example1.getmetas().keys()+example2.getmetas().keys())
        sum=0
        getmeta=lambda e: e.hasmeta(key) and float(e[key]) or 0.0
        for key in s:
            sum+=pow(getmeta(example2)-getmeta(example1), 2)
        return pow(sum, 0.5)

class BagOfWords(object):
    def __call__(self, example1, example2):
        """ Computes a BOW kernel function
         
        .. math:: \sum_{i=1}^n example1_i * example2_i
        
        using the examples meta attributes (need to be floats)
        """
        s=set(example1.getmetas().keys()).intersection(set(example2.getmetas().keys()))
        sum=0
        for key in s:
            sum+=float(example2[key])*float(example1[key])
        return sum
    
class MeasureAttribute_SVMWeights(orange.MeasureAttribute):
    """ Measure attribute relevance by training an linear SVM classifier on
    provided examples and using a squared sum of weights (of each binary
    classifier) as the returned measure.
        
    Example::
        >>> measure = MeasureAttribute_SVMWeights()
        >>> for attr in data.domain.attributes:
        ...   print "%15s: %.3f" % (attr.name, measure(attr, data))
          
    """
    def __new__(cls, attr=None, examples=None, weightId=None, **kwargs):
        self = orange.MeasureAttribute.__new__(cls, **kwargs)
        if examples is not None and attr is not None:
            self.__init__(**kwargs)
            return self.__call__(attr, examples, weightId)
        else:
            return self
        
    def __reduce__(self):
        return MeasureAttribute_SVMWeights, (), {"learner": self.learner}
    
    def __init__(self, learner=None, **kwargs):
        """
        :param learner: Learner used for weight esstimation (default LinearLearner(solver_type=L2Loss_SVM_Dual))
        :type learner: orange.Learner 
        """
        self.learner = LinearLearner(solver_type=LinearLearner.L2Loss_SVM_Dual) if learner is None else learner
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
    """ Recursive feature elimination using linear svm derived attribute weights.
    
    Example::
    
        >>> rfe = RFE(SVMLearner(kernel_type=SVMLearner.Linear, normalization=False)) # normalization=False -> SVM Learner should not change the domain 
        >>> data_with_removed_features = rfe(data, 5) # returns an example table with only 5 best attributes
        
    """
    def __init__(self, learner=None):
        self.learner = learner or SVMLearner(kernel_type=orange.SVMLearner.Linear, normalization=False)

    def getAttrScores(self, data, stopAt=0, progressCallback=None):
        """ Return a dict mapping attributes to scores (scores are not scores in a general
        meaning they represent the step number at which they were removed from the recursive
        evaluation).
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
        """ Return a new dataset with only `numSelected` best scoring attributes.
        
        :param data: Data
        :type data: orange.ExampleTable
        :param numSelected: number of features to preserve
        :type numSelected: int
         
        """
        scores = self.getAttrScores(data, progressCallback=progressCallback)
        scores = sorted(scores.items(), key=lambda item: item[1])
        
        scores = dict(scores[-numSelected:])
        attrs = [attr for attr in data.domain.attributes if attr in scores]
        domain = orange.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        data = orange.ExampleTable(domain, data)
        return data

def exampleTableToSVMFormat(examples, file):
    """ Save an example table in svm format as used by LibSVM
    """
    attrs = examples.domain.attributes + examples.domain.getmetas().values()
    attrs = [attr for attr in attrs if attr.varType in [orange.VarTypes.Continuous, orange.VarTypes.Discrete]]
    cv = examples.domain.classVar
    
    for ex in examples:
        file.write(str(int(ex[cv]) if cv.varType == orange.VarTypes.Discrete else float(ex[cv])))
        for i, attr in enumerate(attrs):
            if not ex[attr].isSpecial():
                file.write(" "+str(i+1)+":"+str(ex[attr]))
        file.write("\n")
            
class LinearLearner(orange.LinearLearner):
    """ A wrapper around orange.LinearLearner with a default
    solver_type == L2Loss_SVM_Dual 
    
    .. note:: The default in orange.LinearLearner is L2_LR
    
    """
    def __new__(cls, data=None, weightId=0, **kwargs):
        self = orange.LinearLearner.__new__(cls, **kwargs)
        if data:
            self.__init__(**kwargs)
            return self.__call__(data, weightId)
        else:
            return self
        
    def __init__(self, **kwargs):
        if kwargs.get("solver_type", None) in [orange.LinearLearner.L2_LR, None]:
            kwargs = dict(kwargs)
            kwargs["solver_type"] = orange.LinearLearner.L2Loss_SVM_Dual
        for name, val in kwargs.items():
            setattr(self, name, val)
