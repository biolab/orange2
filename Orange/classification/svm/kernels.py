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
    
    :param wrapped: a kernel function to wrap
    
    """
    
    def __init__(self, wrapped):
        self.wrapped=wrapped
        
    def __call__(self, inst1, inst2):
        return self.wrapped(inst1, inst2)
 
class DualKernelWrapper(KernelWrapper):
    
    """A base class for kernel wrapper that wraps two kernel functions.
    
    :param wrapped1:  first kernel function
    :param wrapped2:  second kernel function
    
    """
    
    def __init__(self, wrapped1, wrapped2):
        self.wrapped1=wrapped1
        self.wrapped2=wrapped2
        
class RBFKernelWrapper(KernelWrapper):
    
    """A Kernel wrapper that wraps the given function into RBF
    
    :param wrapped: a kernel function
    :param gamma: the gamma of the RBF
    :type gamma: double
    
    """
    
    def __init__(self, wrapped, gamma=0.5):
        KernelWrapper.__init__(self, wrapped)
        self.gamma=gamma
        
    def __call__(self, inst1, inst2):
        """Return :math:`exp(-gamma * wrapped(inst1, inst2) ^ 2)` 
        """
        
        return math.exp(
            -self.gamma*math.pow(self.wrapped(inst1, inst2), 2))
            
class PolyKernelWrapper(KernelWrapper):
    
    """Polynomial kernel wrapper.
    
    :param wrapped: a kernel function

    :param degree: degree of the polynomial
    :type degree: float
    
    """
    
    def __init__(self, wrapped, degree=3.0):
        KernelWrapper.__init__(self, wrapped)
        self.degree=degree
        
    def __call__(self, inst1, inst2):
        """Return :math:`wrapped(inst1, inst2) ^ d`"""
        
        return math.pow(self.wrapped(inst1, inst2), self.degree)

class AdditionKernelWrapper(DualKernelWrapper):
    
    """
    Addition kernel wrapper.

    :param wrapped1:  first kernel function
    :param wrapped2:  second kernel function

    """
    
    def __call__(self, inst1, inst2):
        """Return :math:`wrapped1(inst1, inst2) + wrapped2(inst1, inst2)`
            
        """
        
        return self.wrapped1(inst1, inst2) + self.wrapped2(inst1, inst2)

class MultiplicationKernelWrapper(DualKernelWrapper):
    
    """
    Multiplication kernel wrapper.

    :param wrapped1:  first kernel function
    :param wrapped2:  second kernel function
"""
    
    def __call__(self, inst1, inst2):
        """Return :math:`wrapped1(inst1, inst2) * wrapped2(inst1, inst2)`
            
        """
        
        return self.wrapped1(inst1, inst2) * self.wrapped2(inst1, inst2)

class CompositeKernelWrapper(DualKernelWrapper):
    
    """Composite kernel wrapper.

    :param wrapped1:  first kernel function
    :param wrapped2:  second kernel function
    :param l: coefficient
    :type l: double
        
    """
    
    def __init__(self, wrapped1, wrapped2, l=0.5):
        DualKernelWrapper.__init__(self, wrapped1, wrapped2)
        self.l=l
        
    def __call__(self, inst1, inst2):
        """Return :math:`l*wrapped1(inst1, inst2) + (1-l)*wrapped2(inst1, inst2)`
            
        """
        return self.l * self.wrapped1(inst1, inst2) + \
            (1-self.l) * self.wrapped2(inst1, inst2)

class SparseLinKernel(object):
    def __call__(self, inst1, inst2):
        """
        Compute a linear kernel function using the instances' meta attributes.
        The meta attributes' values must be floats.
        
        """
        s = set(inst1.getmetas().keys()) & set(inst2.getmetas().keys())
        sum = 0
        for key in s:
            sum += float(inst2[key]) * float(inst1[key])
        return sum

BagOfWords = SparseLinKernel
