"""
.. autoclass:: Preprocessor_discretizeEntropy

.. autoclass:: Preprocessor_removeContinuous

.. autoclass:: Preprocessor_continuize

.. autoclass:: Preprocessor_removeDiscrete

.. autoclass:: Preprocessor_impute

.. autoclass:: Preprocessor_featureSelection

.. autofunction:: bestP

.. autofunction:: bestN

.. autofunction:: selectNRandom

.. autofunction:: selectPRandom

.. autoclass:: Preprocessor_RFE

.. autoclass:: Preprocessor_sample

.. autoclass:: Preprocessor_preprocessorList

.. class:: RemoveUnusedValues(variable, data, remove_one_valued=False)

    Removes unused values and reduces the variable, if a variable
    declares values that do not appear in the data.

    :param variable: :class:`Orange.feature.Descriptor`
    :param data: :class:`Orange.data.Table`
    :param remove_one_valued: Decides whether to remove or to retain
        the attributes with only one value defined (default: False).
    
    Example:
    
    .. literalinclude:: code/unusedValues.py    

    There are four possible outcomes:
    
    1. The variable does not have any used values in the data - value
    of this variable is undefined for all examples. The variable is
    thus useless and the class returns None.

    2. The variable has only one used value (or, possibly, only one
    value at all). Such a variable is in fact useless, and can
    probably be removed without harm. Nevertheless, its fate is
    decided by the flag remove_one_valued which is False by default,
    so such variables are retained unless explicitly specified
    otherwise.

    3. All variable's values occur in the data (and the variable has more
    than one value; otherwise the above case applies). The original variable
    is returned.

    4. There are some unused values. A new variable is constructed and the
    unused values are omitted. The value of the new variable is computed
    automatically from the value of the original variable 
    :class:`Orange.classification.lookup.ClassifierByLookupTable` is used
    for mapping.
    
    Results of example:
    
    .. literalinclude:: code/unusedValues.res
    
    Variables a and y are OK and are left alone. In b, value 1 is not used
    and is removed (not in the original variable, of course; a new variable
    is created). c is useless and is removed altogether. d is retained since
    remove_one_valued was left at False; if we set it to True, this variable
    would be removed as well.

"""

from orange import \
     DomainContinuizer, \
    VariableFilterMap, \
    ValueFilter, \
         ValueFilter_continuous, \
         ValueFilter_discrete, \
         ValueFilter_string, \
         ValueFilter_stringList, \
    ValueFilterList, \
    TransformValue, \
         Discrete2Continuous, \
         Discretizer, \
              BiModalDiscretizer, \
              EquiDistDiscretizer, \
              IntervalDiscretizer, \
              ThresholdDiscretizer, \
         MapIntValue, \
         NormalizeContinuous, \
         Ordinal2Continuous, \
         TransformValue_IsDefined, \
    TableAverager, \
    Preprocessor, \
         Preprocessor_addCensorWeight, \
         Preprocessor_addClassNoise, \
         Preprocessor_addClassWeight, \
         Preprocessor_addGaussianClassNoise, \
         Preprocessor_addGaussianNoise, \
         Preprocessor_addMissing, \
         Preprocessor_addMissingClasses, \
         Preprocessor_addNoise, \
         Preprocessor_discretize, \
         Preprocessor_drop, \
         Preprocessor_dropMissing, \
         Preprocessor_dropMissingClasses, \
         Preprocessor_filter, \
         Preprocessor_ignore, \
         Preprocessor_imputeByLearner, \
         Preprocessor_removeDuplicates, \
         Preprocessor_select, \
         Preprocessor_shuffle, \
         Preprocessor_take, \
         Preprocessor_takeMissing, \
         Preprocessor_takeMissingClasses, \
    Imputer, \
         Imputer_asValue, \
         Imputer_defaults, \
         Imputer_model, \
         Imputer_random, \
    ImputerConstructor, \
         ImputerConstructor_asValue, \
         ImputerConstructor_average, \
         ImputerConstructor_maximal, \
         ImputerConstructor_minimal, \
         ImputerConstructor_model, \
         ImputerConstructor_random, \
    FilterList, \
    Filter, \
         Filter_conjunction, \
         Filter_disjunction, \
         Filter_hasClassValue, \
         Filter_hasMeta, \
         Filter_hasSpecial, \
         Filter_isDefined, \
         Filter_random, \
         Filter_sameValue, \
         Filter_values, \
    Discretization, \
         BiModalDiscretization, \
         EntropyDiscretization, \
         EquiDistDiscretization, \
         EquiNDiscretization, \
    DomainTransformerConstructor, \
    RemoveRedundant, \
         RemoveRedundantByInduction, \
         RemoveRedundantByQuality, \
         RemoveRedundantOneValue, \
    RemoveUnusedValues

import math

import orange
from Orange.misc import _orange__new__, _orange__reduce__

class Preprocessor_discretizeEntropy(Preprocessor_discretize):
    """ An discretizer that uses orange.EntropyDiscretization method but,
    unlike Preprocessor_discretize class, also removes unused attributes
    from the domain.
    
    """
    
    __new__ = _orange__new__(Preprocessor_discretize)
    __reduce__ = _orange__reduce__
    
    def __init__(self, method=orange.EntropyDiscretization()):
        self.method = method
        assert(isinstance(method, orange.EntropyDiscretization))
        
    def __call__(self, data, wightId=0):
        newattr_list = []
        for attr in data.domain.attributes:
            if attr.varType == orange.VarTypes.Continuous:
                newattr = self.method(attr, data)
                if newattr.getValueFrom.transformer.points:
                    newattr_list.append(newattr)
            else:
                newattr_list.append(attr)
        newdomain = orange.Domain(newattr_list, data.domain.classVar)
        newdomain.addmetas(data.domain.getmetas())
        return orange.ExampleTable(newdomain, data)
    
class Preprocessor_removeContinuous(Preprocessor_discretize):
    """ A preprocessor that removes all continuous features.
    """
    __new__ = _orange__new__(Preprocessor_discretize)
    __reduce__ = _orange__reduce__
    
    def __call__(self, data, weightId=None):
        attrs = [attr for attr in data.domain.attributes if attr.varType == orange.VarTypes.Discrete]
        domain = orange.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        return orange.ExampleTable(domain, data)
                
class Preprocessor_continuize(orange.Preprocessor):
    """ A preprocessor that continuizes a discrete domain (and optionally normalizes it).
    See :obj:`Orange.data.continuization.DomainContinuizer` for list of
    accepted arguments.
    
    """
    __new__ = _orange__new__(orange.Preprocessor)
    __reduce__ = _orange__reduce__
    
    def __init__(self, zeroBased=True, multinomialTreatment=orange.DomainContinuizer.NValues,
                 continuousTreatment=orange.DomainContinuizer.Leave,
                 classTreatment=orange.DomainContinuizer.Ignore,
                 **kwargs):
        self.zeroBased = zeroBased
        self.multinomialTreatment = multinomialTreatment
        self.continuousTreatment = continuousTreatment
        self.classTreatment = classTreatment
            
    def __call__(self, data, weightId=0):
        continuizer = orange.DomainContinuizer(zeroBased=self.zeroBased,
                                               multinomialTreatment=self.multinomialTreatment,
                                               continuousTreatment=self.continuousTreatment,
                                               classTreatment=self.classTreatment)
        c_domain = continuizer(data, weightId)
        return data.translate(c_domain)
    
class Preprocessor_removeDiscrete(Preprocessor_continuize):
    """ A Preprocessor that removes all discrete attributes from the domain.
    """
    __new__ = _orange__new__(Preprocessor_continuize)
    
    def __call__(self, data, weightId=None):
        attrs = [attr for attr in data.domain.attributes if attr.varType == orange.VarTypes.Continuous]
        domain = orange.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        return orange.ExampleTable(domain, data)
         
class Preprocessor_impute(orange.Preprocessor):
    """ A preprocessor that imputes unknown values using a learner.
    
    :param model: a learner class.
    
    """
    __new__ = _orange__new__(orange.Preprocessor)
    __reduce__ = _orange__reduce__
    
    def __init__(self, model=None, **kwargs):
        self.model = orange.MajorityLearner() if model is None else model
        
    def __call__(self, data, weightId=0):
        return orange.Preprocessor_imputeByLearner(data, learner=self.model)

def bestN(attrMeasures, N=10):
    """ Return best N attributes 
    """
    return attrMeasures[-N:]

def bestP(attrMeasures, P=10):
    """ Return best P percent of attributes
    """
    count = len(attrMeasures)
    return  attrMeasures[-max(int(math.ceil(count * P / 100.0)), 1):]

class Preprocessor_featureSelection(orange.Preprocessor):
    """ A preprocessor that runs feature selection using an feature scoring function.
    
    :param measure: a scoring function (default: orange.MeasureAttribute_relief)
    :param filter: a filter function to use for selection (default Preprocessor_featureSelection.bestN)
    :param limit: the limit for the filter function (default 10)
        
    """
    __new__ = _orange__new__(orange.Preprocessor)
    __reduce__ = _orange__reduce__
    
    bestN = staticmethod(bestN)
    bestP = staticmethod(bestP)
    
    def __init__(self, measure=orange.MeasureAttribute_relief(), filter=None, limit=10):
        self.measure = measure
        self.filter = filter if filter is not None else self.bestN
        self.limit = limit
    
    def attrScores(self, data):
        """ Return a list of computed scores for all attributes in `data`. 
        """
        measures = sorted([(self.measure(attr, data), attr) for attr in data.domain.attributes])
        return measures
         
    def __call__(self, data, weightId=None):
        measures = self.attrScores(data)
        attrs = [attr for _, attr in self.filter(measures, self.limit)]
        domain = orange.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        return orange.ExampleTable(domain, data)
    
class Preprocessor_RFE(Preprocessor_featureSelection):
    """ A preprocessor that runs RFE(Recursive Feature Elimination) using
    linear SVM derived attribute weights.
    
    :param filter: a filter function to use for selection (default
                   Preprocessor_featureSelection.bestN)
    :param limit: the limit for the filter function (default 10)
        
    """
    __new__ = _orange__new__(Preprocessor_featureSelection)
    __reduce__ = _orange__reduce__
    def __init__(self, filter=None, limit=10):
        self.limit = limit
        self.filter = filter if filter is not None else self.bestN
        
    def __call__(self, data, weightId=None):
        from Orange.classification.svm import RFE
        rfe = RFE()
        filtered = self.filter(range(len(data)), self.limit)
        return rfe(data, len(filtered))
    
def selectNRandom(examples, N=10):
    """ Select N random examples.
    """
    import random
    return random.sample(examples, N)

def selectPRandom(examples, P=10):
    """ Select P percent random examples.
    """
    import random
    count = len(examples)
    return random.sample(examples, max(int(math.ceil(count * P / 100.0)), 1))

class Preprocessor_sample(orange.Preprocessor):
    """ A preprocessor that samples a subset of the data.
    
    :param filter: a filter function to use for selection (default
                   Preprocessor_sample.selectNRandom)
    :param limit: the limit for the filter function (default 10)
    
    """
    __new__ = _orange__new__(orange.Preprocessor)
    __reduce__ = _orange__reduce__

    selectNRandom = staticmethod(selectNRandom)
    selectPRandom = staticmethod(selectPRandom)
    
    def __init__(self, filter=None, limit=10):
        self.filter = filter if filter is not None else self.selectNRandom
        self.limit = limit
        
    def __call__(self, data, weightId=None):
        return orange.ExampleTable(data.domain, self.filter(data, self.limit))
    

class Preprocessor_preprocessorList(orange.Preprocessor):
    """ A preprocessor wrapping a sequence of other preprocessors.
    
    :param preprocessors: a list of :obj:`Preprocessor` instances
    
    """
    
    __new__ = _orange__new__(orange.Preprocessor)
    __reduce__ = _orange__reduce__
    
    def __init__(self, preprocessors=[]):
        self.preprocessors = preprocessors
        
    def __call__(self, data, weightId=None):
        import orange
        hadWeight = hasWeight = weightId is not None
        for preprocessor in self.preprocessors:
            t = preprocessor(data, weightId) if hasWeight else preprocessor(data)
            if isinstance(t, tuple):
                data, weightId = t
                hasWeight = True
            else:
                data = t
        if hadWeight:
            return data, weightId
        else:
            return data
        
