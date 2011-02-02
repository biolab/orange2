"""
Data instances in Orange can contain four types of features: `discrete`_,
continuous_, strings_ and Python_; the latter represent arbitrary Python objects.
The names, types, values (where applicable), functions for computing the
feature value from other features, and other properties of the
features are stored in descriptors contained in this module.

Feature descriptors can be constructed directly, using constructors, or by a
factory function :obj:`make`, which either retrieves an existing descriptor or
constructs a new one.

.. class:: Feature

    An abstract base class for feature descriptors.

    .. attribute:: name
    
    Each feature has a name. The names do not need to be unique since two
    features are considered the same only if they have the same descriptor
    (e.g. even multiple features in the same table can have the same name).
    This should however be avoided since it may result in unpredictable
    behaviour.
    
    .. attribute:: varType
       
    Stores the feature type; it can be Orange.data.Type.Discrete,
    Orange.data.Type.Continuous, Orange.data.Type.String or
    Orange.data.Type.Other.  

    .. attribute:: getValueFrom
    A function (an instance of `Orange.core.Clasifier`) which computes a
    value of the feature from values of one or more other features. This is
    used, for instance, in discretization where the features describing the 
    discretized feature are computed from the original feature. 

    .. attribute:: ordered
    A flag telling whether the values of a discrete feature are ordered. At the 
    moment, no builtin method treats ordinal features differently than nominal.
    
    .. attribute:: distributed
    A flag telling whether the values of this features are distributions.
    As for flag ordered, no methods treat such features in any special manner.
    
    .. attribute:: randomGenerator
    A local random number generator used by method :obj:`Feature.randomvalue`.
    
    .. attribute:: defaultMetaId
    A proposed (but not guaranteed) meta id to be used for that feature. This is 
    used, for instance, by the data loader for tab-delimited file format instead 
    of assigning an arbitrary new value, or by `Orange.core.newmetaid` if the
    feature is passed as an argument. 
        

    .. method:: __call__(obj)
       Convert a string, number or other suitable object into a feature value.
       :param obj: An object to be converted into a feature value
       :type o: any suitable
       :rtype: :class:`Orange.data.Value`
       
    .. method:: randomvalue()
       Return a random value of the feature
       :rtype: :class:`Orange.data.Value`
       
    .. method:: computeValue()
       Calls getValueFrom through a mechanism that prevents deadlocks by circular calls.
       
"""
from orange import Variable as Feature
from orange import EnumVariable as Discrete
from orange import FloatVariable as Continuous
from orange import PythonVariable as Python
from orange import StringVariable as String

from orange import VarList as Features

import orange
make = orange.Variable.make
retrieve = orange.Variable.getExisting
del orange