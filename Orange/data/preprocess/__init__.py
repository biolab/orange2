from Orange import core

DomainContinuizer = core.DomainContinuizer
VariableFilterMap = core.VariableFilterMap

ValueFilter = core.ValueFilter
ValueFilter_continuous = core.ValueFilter_continuous
ValueFilter_discrete = core.ValueFilter_discrete
ValueFilter_string = core.ValueFilter_string
ValueFilter_stringList = core.ValueFilter_stringList
ValueFilterList = core.ValueFilterList

TransformValue = core.TransformValue
Discrete2Continuous = core.Discrete2Continuous
Discretizer = core.Discretizer
BiModalDiscretizer = core.BiModalDiscretizer
EquiDistDiscretizer = core.EquiDistDiscretizer
IntervalDiscretizer = core.IntervalDiscretizer
ThresholdDiscretizer = core.ThresholdDiscretizer
MapIntValue = core.MapIntValue
NormalizeContinuous = core.NormalizeContinuous
Ordinal2Continuous = core.Ordinal2Continuous
TransformValue_IsDefined = core.TransformValue_IsDefined

TableAverager = core.TableAverager

Preprocessor = core.Preprocessor
AddCensorWeight = core.Preprocessor_addCensorWeight
AddClassNoise = core.Preprocessor_addClassNoise
AddClassWeight = core.Preprocessor_addClassWeight
AddGaussianClassNoise = core.Preprocessor_addGaussianClassNoise
AddGaussianNoise = core.Preprocessor_addGaussianNoise
AddMissing = core.Preprocessor_addMissing
AddMissingClasses = core.Preprocessor_addMissingClasses
AddNoise = core.Preprocessor_addNoise

Discretize = core.Preprocessor_discretize
Drop = core.Preprocessor_drop
DropMissing = core.Preprocessor_dropMissing
DropMissingClasses = core.Preprocessor_dropMissingClasses
Filter = core.Preprocessor_filter
Ignore = core.Preprocessor_ignore
ImputeByLearner = core.Preprocessor_imputeByLearner
RemoveDuplicates = core.Preprocessor_removeDuplicates
Select = core.Preprocessor_select
Shuffle = core.Preprocessor_shuffle
Take = core.Preprocessor_take
TakeMissing = core.Preprocessor_takeMissing
TakeMissingClasses = core.Preprocessor_takeMissingClasses

Imputer = core.Imputer
Imputer_asValue = core.Imputer_asValue
Imputer_defaults = core.Imputer_defaults
Imputer_model = core.Imputer_model
Imputer_random = core.Imputer_random

ImputerConstructor = core.ImputerConstructor
ImputerConstructor_asValue = core.ImputerConstructor_asValue
ImputerConstructor_average = core.ImputerConstructor_average
ImputerConstructor_maximal = core.ImputerConstructor_maximal
ImputerConstructor_minimal = core.ImputerConstructor_minimal
ImputerConstructor_model = core.ImputerConstructor_model
ImputerConstructor_random = core.ImputerConstructor_random

FilterList = core.FilterList
Filter = core.Filter
Filter_conjunction = core.Filter_conjunction
Filter_disjunction = core.Filter_disjunction
Filter_hasClassValue = core.Filter_hasClassValue
Filter_hasMeta = core.Filter_hasMeta
Filter_hasSpecial = core.Filter_hasSpecial
Filter_isDefined = core.Filter_isDefined
Filter_random = core.Filter_random
Filter_sameValue = core.Filter_sameValue
Filter_values = core.Filter_values

Discretization = core.Discretization
BiModalDiscretization = core.BiModalDiscretization
EntropyDiscretization = core.EntropyDiscretization
EquiDistDiscretization = core.EquiDistDiscretization
EquiNDiscretization = core.EquiNDiscretization

DomainTransformerConstructor = core.DomainTransformerConstructor

RemoveRedundant = core.RemoveRedundant
RemoveRedundantByInduction = core.RemoveRedundantByInduction
RemoveRedundantByQuality = core.RemoveRedundantByQuality
RemoveRedundantOneValue = core.RemoveRedundantOneValue
RemoveUnusedValues = core.RemoveUnusedValues

import math

import Orange
import Orange.classification.majority
import Orange.data
import Orange.feature
import Orange.feature.discretization
import Orange.feature.scoring
from Orange.utils import _orange__new__, _orange__reduce__

class DiscretizeEntropy(Discretize):
    """ An discretizer that uses orange.EntropyDiscretization method but,
    unlike Preprocessor_discretize class, also removes unused attributes
    from the domain.
    """
    
    __new__ = _orange__new__(Discretize)
    __reduce__ = _orange__reduce__
    
    def __init__(self, method=Orange.feature.discretization.Entropy()):
        self.method = method
        assert(isinstance(method, Orange.feature.discretization.Entropy))
        
    def __call__(self, data, weightId=0):
        newattr_list = []
        for attr in data.domain.attributes:
            if attr.varType == Orange.feature.Type.Continuous:
                newattr = self.method(attr, data)
                if newattr.getValueFrom.transformer.points:
                    newattr_list.append(newattr)
            else:
                newattr_list.append(attr)
        newdomain = Orange.data.Domain(newattr_list, data.domain.classVar)
        newdomain.addmetas(data.domain.getmetas())
        return Orange.data.Table(newdomain, data)
    
class RemoveContinuous(Discretize):
    """ A preprocessor that removes all continuous features.
    """
    __new__ = _orange__new__(Discretize)
    __reduce__ = _orange__reduce__
    
    def __call__(self, data, weightId=None):
        attrs = [attr for attr in data.domain.attributes if attr.varType == Orange.feature.Type.Discrete]
        domain = Orange.data.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        return Orange.data.Table(domain, data)
                
class Continuize(Preprocessor):
    """ A preprocessor that continuizes a discrete domain (and optionally normalizes it).
    See :obj:`Orange.data.continuization.DomainContinuizer` for list of
    accepted arguments.
    
    """
    __new__ = _orange__new__(Preprocessor)
    __reduce__ = _orange__reduce__
    
    def __init__(self, zeroBased=True, multinomialTreatment=DomainContinuizer.NValues,
                 continuousTreatment=DomainContinuizer.Leave,
                 classTreatment=DomainContinuizer.Ignore,
                 **kwargs):
        self.zeroBased = zeroBased
        self.multinomialTreatment = multinomialTreatment
        self.continuousTreatment = continuousTreatment
        self.classTreatment = classTreatment
            
    def __call__(self, data, weightId=0):
        continuizer = DomainContinuizer(zeroBased=self.zeroBased,
                                        multinomialTreatment=self.multinomialTreatment,
                                        continuousTreatment=self.continuousTreatment,
                                        classTreatment=self.classTreatment)
        c_domain = continuizer(data, weightId)
        return data.translate(c_domain)
    
class RemoveDiscrete(Continuize):
    """ A Preprocessor that removes all discrete attributes from the domain.
    """
    __new__ = _orange__new__(Continuize)
    
    def __call__(self, data, weightId=None):
        attrs = [attr for attr in data.domain.attributes if attr.varType == Orange.feature.Type.Continuous]
        domain = Orange.data.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        return Orange.data.Table(domain, data)
         
class Impute(Preprocessor):
    """ A preprocessor that imputes unknown values using a learner.
    
    :param model: a learner class.
    
    """
    __new__ = _orange__new__(Preprocessor)
    __reduce__ = _orange__reduce__
    
    def __init__(self, model=None, **kwargs):
        self.model = Orange.classification.majority.MajorityLearner() if model is None else model
        
    def __call__(self, data, weightId=0):
        return ImputeByLearner(data, learner=self.model)

def bestN(attrMeasures, N=10):
    """ Return best N attributes 
    """
    return attrMeasures[-N:]
_bestN = bestN

def bestP(attrMeasures, P=10):
    """ Return best P percent of attributes
    """
    count = len(attrMeasures)
    return  attrMeasures[-max(int(math.ceil(count * P / 100.0)), 1):]
_bestP = bestP

class FeatureSelection(Preprocessor):
    """ A preprocessor that runs feature selection using an feature scoring function.
    
    :param measure: a scoring function (default: orange.MeasureAttribute_relief)
    :param filter: a filter function to use for selection (default Preprocessor_featureSelection.bestN)
    :param limit: the limit for the filter function (default 10)
        
    """
    __new__ = _orange__new__(Preprocessor)
    __reduce__ = _orange__reduce__
    
    bestN = staticmethod(_bestN)
    bestP = staticmethod(_bestP)
    
    def __init__(self, measure=Orange.feature.scoring.Relief(), filter=None, limit=10):
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
        domain = Orange.data.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        return Orange.data.Table(domain, data)
    
class RFE(FeatureSelection):
    """ A preprocessor that runs RFE(Recursive Feature Elimination) using
    linear SVM derived attribute weights.
    
    :param filter: a filter function to use for selection (default
                   Preprocessor_featureSelection.bestN)
    :param limit: the limit for the filter function (default 10)
        
    """
    __new__ = _orange__new__(FeatureSelection)
    __reduce__ = _orange__reduce__

    def __init__(self, filter=None, limit=10):
        super(RFE, self).__init__(filter=filter, limit=limit)
        
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
_selectNRandom = selectNRandom

def selectPRandom(examples, P=10):
    """ Select P percent random examples.
    """
    import random
    count = len(examples)
    return random.sample(examples, max(int(math.ceil(count * P / 100.0)), 1))
_selectPRandom = selectPRandom

class Sample(Preprocessor):
    """ A preprocessor that samples a subset of the data.
    
    :param filter: a filter function to use for selection (default
                   Preprocessor_sample.selectNRandom)
    :param limit: the limit for the filter function (default 10)
    
    """
    __new__ = _orange__new__(Preprocessor)
    __reduce__ = _orange__reduce__

    selectNRandom = staticmethod(_selectNRandom)
    selectPRandom = staticmethod(_selectPRandom)
    
    def __init__(self, filter=None, limit=10):
        self.filter = filter if filter is not None else self.selectNRandom
        self.limit = limit
        
    def __call__(self, data, weightId=None):
        return Orange.data.Table(data.domain, self.filter(data, self.limit))
    

class PreprocessorList(Preprocessor):
    """ A preprocessor wrapping a sequence of other preprocessors.
    
    :param preprocessors: a list of :obj:`Preprocessor` instances
    
    """
    
    __new__ = _orange__new__(Preprocessor)
    __reduce__ = _orange__reduce__
    
    def __init__(self, preprocessors=()):
        self.preprocessors = preprocessors
        
    def __call__(self, data, weightId=None):
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
        
