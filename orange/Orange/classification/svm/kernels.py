import math

from Orange.core import SVMLearner as _SVMLearner
from Orange.core import KernelFunc

Linear = _SVMLearner.Linear
Polynomial = _SVMLearner.Polynomial
RBF = _SVMLearner.RBF
Sigmoid = _SVMLearner.Sigmoid
Custom = _SVMLearner.Custom

class KernelWrapper(object):
    
    """A base class for kernel function wrappers.
    
    :param wrapped: a function to wrap
    :type wrapped: function(:class:`Orange.data.Instance`, :class:`Orange.data.Instance`)
    
    """
    
    def __init__(self, wrapped):
        self.wrapped=wrapped
        
    def __call__(self, example1, example2):
        return self.wrapped(example1, example2)
 
class DualKernelWrapper(KernelWrapper):
    
    """A base class for kernel wrapper that wraps two other kernel functions.
    
    :param wrapped1:  a function to wrap
    :type wrapped1: function(:class:`Orange.data.Instance`, :class:`Orange.data.Instance`)
    :param wrapped2:  a function to wrap
    :type wrapped2: function(:class:`Orange.data.Instance`, :class:`Orange.data.Instance`)
    
    """
    
    def __init__(self, wrapped1, wrapped2):
        self.wrapped1=wrapped1
        self.wrapped2=wrapped2
        
class RBFKernelWrapper(KernelWrapper):
    
    """A Kernel wrapper that uses a wrapped kernel function in a RBF
    (Radial Basis Function).
    
    :param wrapped: a function to wrap
    :type wrapped: function(:class:`Orange.data.Instance`, :class:`Orange.data.Instance`)
    :param gamma: the gamma of the RBF
    :type gamma: double
    
    """
    
    def __init__(self, wrapped, gamma=0.5):
        KernelWrapper.__init__(self, wrapped)
        self.gamma=gamma
        
    def __call__(self, example1, example2):
        """:math:`exp(-gamma * wrapped(example1, example2) ^ 2)` 
        
        """
        
        return math.exp(-self.gamma*math.pow(self.wrapped(example1, 
                                                          example2),2))
            
class PolyKernelWrapper(KernelWrapper):
    
    """Polynomial kernel wrapper.
    
    :param wrapped: a function to wrap
    :type wrapped: function(:class:`Orange.data.Instance`, :class:`Orange.data.Instance`)
    :param degree: degree of the polynomial
    :type degree: double
    
    """
    
    def __init__(self, wrapped, degree=3.0):
        KernelWrapper.__init__(self, wrapped)
        self.degree=degree
        
    def __call__(self, example1, example2):
        """:math:`wrapped(example1, example2) ^ d`"""
        
        return math.pow(self.wrapped(example1, example2), self.degree)

class AdditionKernelWrapper(DualKernelWrapper):
    
    """Addition kernel wrapper."""
    
    def __call__(self, example1, example2):
        """:math:`wrapped1(example1, example2) + wrapped2(example1, example2)`
            
        """
        
        return self.wrapped1(example1, example2) + \
                                            self.wrapped2(example1, example2)

class MultiplicationKernelWrapper(DualKernelWrapper):
    
    """Multiplication kernel wrapper."""
    
    def __call__(self, example1, example2):
        """:math:`wrapped1(example1, example2) * wrapped2(example1, example2)`
            
        """
        
        return self.wrapped1(example1, example2) * \
                                            self.wrapped2(example1, example2)

class CompositeKernelWrapper(DualKernelWrapper):
    
    """Composite kernel wrapper.
    
    :param wrapped1:  a function to wrap
    :type wrapped1: function(:class:`Orange.data.Instance`, :class:`Orange.data.Instance`)
    :param wrapped2:  a function to wrap
    :type wrapped2: function(:class:`Orange.data.Instance`, :class:`Orange.data.Instance`)
    :param l: coefficient
    :type l: double
        
    """
    
    def __init__(self, wrapped1, wrapped2, l=0.5):
        DualKernelWrapper.__init__(self, wrapped1, wrapped2)
        self.l=l
        
    def __call__(self, example1, example2):
        """:math:`l*wrapped1(example1,example2)+(1-l)*wrapped2(example1,example2)`
            
        """
        return self.l * self.wrapped1(example1, example2) + (1-self.l) * \
                                            self.wrapped2(example1,example2)

class SparseLinKernel(object):
    def __call__(self, example1, example2):
        """Computes a linear kernel function using the examples meta attributes
        (need to be floats).
        
        """
        s = set(example1.getmetas().keys()) & set(example2.getmetas().keys())
        sum = 0
        for key in s:
            sum += float(example2[key]) * float(example1[key])
        return sum

BagOfWords = SparseLinKernel
