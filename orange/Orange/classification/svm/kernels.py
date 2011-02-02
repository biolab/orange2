import math

from Orange.core import SVMLearner as _SVMLearner
from Orange.core import KernelFunc

Linear = _SVMLearner.Linear
Polynomial = _SVMLearner.Polynomial
RBF = _SVMLearner.RBF
Sigmoid = _SVMLearner.Sigmoid
Custom = _SVMLearner.Custom

class KernelWrapper(object):
    
    """A base class for kernel function wrappers"""
    
    def __init__(self, wrapped):
        """:param wrapped: a function to wrap"""
        self.wrapped=wrapped
        
    def __call__(self, example1, example2):
        return self.wrapped(example1, example2)
 
class DualKernelWrapper(KernelWrapper):
    
    """A base class for kernel wrapper that wrap two other kernel functions."""
    
    def __init__(self, wrapped1, wrapped2):
        """:param wrapped1:
        :param wrapped2:
        
        """
        self.wrapped1=wrapped1
        self.wrapped2=wrapped2
        
class RBFKernelWrapper(KernelWrapper):
    
    """AKernel wrapper that uses a wrapped kernel function in a RBF
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
        """Return 
        
        .. math::  exp(-gamma * wrapped(example1, example2) ^ 2)
        
        """
        return math.exp(-self.gamma*math.pow(self.wrapped(example1, 
                                                          example2),2))
            
class PolyKernelWrapper(KernelWrapper):
    def __init__(self, wrapped, degree=3.0):
        """:param wrapped: a function to wrap
        :param degree: degree of the polinomial
        
        """
        KernelWrapper.__init__(self, wrapped)
        self.degree=degree
    def __call__(self, example1, example2):
        """Return
        
        .. math:: wrapped(example1, example2) ^ d
        
        """
        return math.pow(self.wrapped(example1, example2), self.degree)

class AdditionKernelWrapper(DualKernelWrapper):
    def __call__(self, example1, example2):
        """Return
        
        .. math:: wrapped1(example1, example2) + wrapped2(example1, example2)
            
        """
        return self.wrapped1(example1, example2) + \
                                            self.wrapped2(example1, example2)

class MultiplicationKernelWrapper(DualKernelWrapper):
    def __call__(self, example1, example2):
        """Return
        
        .. math:: wrapped1(example1, example2) * wrapped2(example1, example2)
            
        """
        return self.wrapped1(example1, example2) * \
                                            self.wrapped2(example1, example2)

class CompositeKernelWrapper(DualKernelWrapper):
    def __init__(self, wrapped1, wrapped2, l=0.5):
        DualKernelWrapper.__init__.__doc__ + """\
        :param l:
        
        """
        DualKernelWrapper.__init__(self, wrapped1, wrapped2)
        self.l=l
    def __call__(self, example1, example2):
        """Return
        
        .. math:: l * wrapped1(example1, example2) + (1 - l) * 
        wrapped2(example1, example2)
            
        """
        return self.l * self.wrapped1(example1, example2) + (1-self.l) * \
                                            self.wrapped2(example1,example2)

class SparseLinKernel(object):
    def __call__(self, example1, example2):
        """Computes a linear kernel function using the examples meta attributes
        (need to be floats)
        
        """
        s=set(example1.getmetas().keys()+example2.getmetas().keys())
        sum=0
        getmeta=lambda e: e.hasmeta(key) and float(e[key]) or 0.0
        for key in s:
            sum+=pow(getmeta(example2)-getmeta(example1), 2)
        return pow(sum, 0.5)

class BagOfWords(object):
    def __call__(self, example1, example2):
        """Computes a BOW kernel function
         
        .. math:: \sum_{i=1}^n example1_i * example2_i
        
        using the examples meta attributes (need to be floats)
        
        """
        s = set(example1.getmetas().keys()) & set(example2.getmetas().keys())
        sum = 0
        for key in s:
            sum += float(example2[key]) * float(example1[key])
        return sum
