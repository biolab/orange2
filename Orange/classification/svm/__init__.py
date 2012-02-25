import math

from collections import defaultdict

import Orange.core
import Orange.data
import Orange.misc
import Orange.feature

import kernels
import warnings

from Orange.core import SVMLearner as _SVMLearner
from Orange.core import SVMLearnerSparse as _SVMLearnerSparse
from Orange.core import LinearClassifier, \
                        LinearLearner, \
                        SVMClassifier, \
                        SVMClassifierSparse

from Orange.preprocess import Preprocessor_impute, \
                              Preprocessor_continuize, \
                              Preprocessor_preprocessorList, \
                              DomainContinuizer

from Orange import feature as variable

from Orange.misc import _orange__new__

def max_nu(data):
    """
    Return the maximum nu parameter for the given data table for
    Nu_SVC learning.
    
    :param data: Data with discrete class variable
    :type data: Orange.data.Table
    
    """
    nu = 1.0
    dist = list(Orange.core.Distribution(data.domain.classVar, data))
    def pairs(seq):
        for i, n1 in enumerate(seq):
            for n2 in seq[i + 1:]:
                yield n1, n2
    return min([2.0 * min(n1, n2) / (n1 + n2) for n1, n2 in pairs(dist) \
                if n1 != 0 and n2 != 0] + [nu])

maxNu = max_nu

class SVMLearner(_SVMLearner):
    """
    :param svm_type: the SVM type
    :type svm_type: SVMLearner.SVMType
    :param kernel_type: the kernel type
    :type kernel_type: SVMLearner.Kernel
    :param degree: kernel parameter (only for ``Polynomial``)
    :type degree: int
    :param gamma: kernel parameter; if 0, it is set to 1.0/#features (for ``Polynomial``, ``RBF`` and ``Sigmoid``)
    :type gamma: float
    :param coef0: kernel parameter (for ``Polynomial`` and ``Sigmoid``)
    :type coef0: int
    :param kernel_func: kernel function if ``kernel_type`` is
        ``kernels.Custom``
    :type kernel_func: callable object
    :param C: C parameter (for ``C_SVC``, ``Epsilon_SVR`` and ``Nu_SVR``)
    :type C: float
    :param nu: Nu parameter (for ``Nu_SVC``, ``Nu_SVR`` and ``OneClass``)
    :type nu: float
    :param p: epsilon parameter (for ``Epsilon_SVR``)
    :type p: float
    :param cache_size: cache memory size in MB
    :type cache_size: int
    :param eps: tolerance of termination criterion
    :type eps: float
    :param probability: build a probability model
    :type probability: bool
    :param shrinking: use shrinking heuristics 
    :type shrinking: bool
    :param weight: a list of class weights
    :type weight: list

    Example:
    
        >>> import Orange
        >>> from Orange.classification import svm
        >>> from Orange.evaluation import testing, scoring
        >>> data = Orange.data.Table("vehicle.tab")
        >>> learner = svm.SVMLearner()
        >>> results = testing.cross_validation([learner], data, folds=5)
        >>> print scoring.CA(results)[0]
        0.789613644274
    
    """
    __new__ = _orange__new__(_SVMLearner)

    C_SVC = _SVMLearner.C_SVC
    Nu_SVC = _SVMLearner.Nu_SVC
    OneClass = _SVMLearner.OneClass
    Nu_SVR = _SVMLearner.Nu_SVR
    Epsilon_SVR = _SVMLearner.Epsilon_SVR

    @Orange.misc.deprecated_keywords({"kernelFunc": "kernel_func"})
    def __init__(self, svm_type=Nu_SVC, kernel_type=kernels.RBF,
                 kernel_func=None, C=1.0, nu=0.5, p=0.1, gamma=0.0, degree=3,
                 coef0=0, shrinking=True, probability=True, verbose=False,
                 cache_size=200, eps=0.001, normalization=True,
                 weight=[], **kwargs):
        self.svm_type = svm_type
        self.kernel_type = kernel_type
        self.kernel_func = kernel_func
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
        self.learner = Orange.core.SVMLearner(**kwargs)
        self.weight = weight

    max_nu = staticmethod(max_nu)

    def __call__(self, data, weight=0):
        """Construct a SVM classifier
        
        :param table: data with continuous features
        :type table: Orange.data.Table
        
        :param weight: ignored (required due to base class signature);
        """

        examples = Orange.core.Preprocessor_dropMissingClasses(data)
        class_var = examples.domain.class_var
        if len(examples) == 0:
            raise ValueError("Example table is without any defined classes")

        # Fix the svm_type parameter if we have a class_var/svm_type mismatch
        if self.svm_type in [0, 1] and \
            isinstance(class_var, Orange.feature.Continuous):
            self.svm_type += 3
            #raise AttributeError, "Cannot learn a discrete classifier from non descrete class data. Use EPSILON_SVR or NU_SVR for regression"
        if self.svm_type in [3, 4] and \
            isinstance(class_var, Orange.feature.Discrete):
            self.svm_type -= 3
            #raise AttributeError, "Cannot do regression on descrete class data. Use C_SVC or NU_SVC for classification"
        if self.kernel_type == kernels.Custom and not self.kernel_func:
            raise ValueError("Custom kernel function not supplied")

        import warnings

        nu = self.nu
        if self.svm_type == SVMLearner.Nu_SVC: #is nu feasible
            max_nu = self.max_nu(examples)
            if self.nu > max_nu:
                if getattr(self, "verbose", 0):
                    warnings.warn("Specified nu %.3f is infeasible. \
                    Setting nu to %.3f" % (self.nu, max_nu))
                nu = max(max_nu - 1e-7, 0.0)

        for name in ["svm_type", "kernel_type", "kernel_func", "C", "nu", "p",
                     "gamma", "degree", "coef0", "shrinking", "probability",
                     "verbose", "cache_size", "eps"]:
            setattr(self.learner, name, getattr(self, name))
        self.learner.nu = nu
        self.learner.set_weights(self.weight)

        if self.svm_type == SVMLearner.OneClass and self.probability:
            self.learner.probability = False
            warnings.warn("One-class SVM probability output not supported yet.")
        return self.learn_classifier(examples)

    def learn_classifier(self, data):
        if self.normalization:
            data = self._normalize(data)
            svm = self.learner(data)
            return SVMClassifierWrapper(svm)
        return self.learner(data)

    @Orange.misc.deprecated_keywords({"progressCallback": "progress_callback"})
    def tune_parameters(self, data, parameters=None, folds=5, verbose=0,
                       progress_callback=None):
        """Tune the ``parameters`` on the given ``data`` using 
        internal cross validation.
        
        :param data: data for parameter tuning
        :type data: Orange.data.Table 
        :param parameters: names of parameters to tune
            (default: ["nu", "C", "gamma"])
        :type parameters: list of strings
        :param folds: number of folds for internal cross validation
        :type folds: int
        :param verbose: set verbose output
        :type verbose: bool
        :param progress_callback: callback function for reporting progress
        :type progress_callback: callback function
            
        Here is example of tuning the `gamma` parameter using
        3-fold cross validation. ::

            svm = Orange.classification.svm.SVMLearner()
            svm.tune_parameters(table, parameters=["gamma"], folds=3)
                    
        """

        import orngWrap

        if parameters is None:
            parameters = ["nu", "C", "gamma"]

        searchParams = []
        normalization = self.normalization
        if normalization:
            data = self._normalize(data)
            self.normalization = False
        if self.svm_type in [SVMLearner.Nu_SVC, SVMLearner.Nu_SVR] \
                    and "nu" in parameters:
            numOfNuValues = 9
            if isinstance(data.domain.class_var, variable.Discrete):
                max_nu = max(self.max_nu(data) - 1e-7, 0.0)
            else:
                max_nu = 1.0
            searchParams.append(("nu", [i / 10.0 for i in range(1, 9) if \
                                        i / 10.0 < max_nu] + [max_nu]))
        elif "C" in parameters:
            searchParams.append(("C", [2 ** a for a in  range(-5, 15, 2)]))
        if self.kernel_type == 2 and "gamma" in parameters:
            searchParams.append(("gamma", [2 ** a for a in range(-5, 5, 2)] + [0]))
        tunedLearner = orngWrap.TuneMParameters(object=self,
                            parameters=searchParams,
                            folds=folds,
                            returnWhat=orngWrap.TuneMParameters.returnLearner,
                            progressCallback=progress_callback
                            if progress_callback else lambda i:None)
        tunedLearner(data, verbose=verbose)
        if normalization:
            self.normalization = normalization

    def _normalize(self, data):
        dc = Orange.core.DomainContinuizer()
        dc.class_treatment = Orange.core.DomainContinuizer.Ignore
        dc.continuous_treatment = Orange.core.DomainContinuizer.NormalizeBySpan
        dc.multinomial_treatment = Orange.core.DomainContinuizer.NValues
        newdomain = dc(data)
        return data.translate(newdomain)

SVMLearner = Orange.misc.deprecated_members({
    "learnClassifier": "learn_classifier",
    "tuneParameters": "tune_parameters",
    "kernelFunc" : "kernel_func",
    },
    wrap_methods=["__init__", "tune_parameters"])(SVMLearner)

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

    def class_distribution(self, example):
        example = Orange.data.Instance(self.wrapped.domain, example)
        return self.wrapped.class_distribution(example)

    def get_decision_values(self, example):
        example = Orange.data.Instance(self.wrapped.domain, example)
        return self.wrapped.get_decision_values(example)

    def get_model(self):
        return self.wrapped.get_model()

    def __reduce__(self):
        return SVMClassifierWrapper, (self.wrapped,), dict([(name, val) \
            for name, val in self.__dict__.items() \
            if name not in self.wrapped.__dict__])

SVMClassifierWrapper = Orange.misc.deprecated_members({
    "classDistribution": "class_distribution",
    "getDecisionValues": "get_decision_values",
    "getModel" : "get_model",
    })(SVMClassifierWrapper)

class SVMLearnerSparse(SVMLearner):

    """
    A :class:`SVMLearner` that learns from data stored in meta
    attributes. Meta attributes do not need to be registered with the
    data set domain, or present in all data instances.
    """

    @Orange.misc.deprecated_keywords({"useNonMeta": "use_non_meta"})
    def __init__(self, **kwds):
        SVMLearner.__init__(self, **kwds)
        self.use_non_meta = kwds.get("use_non_meta", False)
        self.learner = Orange.core.SVMLearnerSparse(**kwds)

    def _normalize(self, data):
        if self.use_non_meta:
            dc = Orange.core.DomainContinuizer()
            dc.class_treatment = Orange.core.DomainContinuizer.Ignore
            dc.continuous_treatment = Orange.core.DomainContinuizer.NormalizeBySpan
            dc.multinomial_treatment = Orange.core.DomainContinuizer.NValues
            newdomain = dc(data)
            data = data.translate(newdomain)
        return data

class SVMLearnerEasy(SVMLearner):

    """A class derived from :obj:`SVMLearner` that automatically
    scales the data and performs parameter optimization using
    :func:`SVMLearner.tune_parameters`. The procedure is similar to
    that implemented in easy.py script from the LibSVM package.
    
    """

    def __init__(self, **kwds):
        self.folds = 4
        self.verbose = 0
        SVMLearner.__init__(self, **kwds)
        self.learner = SVMLearner(**kwds)

    def learn_classifier(self, data):
        transformer = Orange.core.DomainContinuizer()
        transformer.multinomialTreatment = Orange.core.DomainContinuizer.NValues
        transformer.continuousTreatment = \
            Orange.core.DomainContinuizer.NormalizeBySpan
        transformer.classTreatment = Orange.core.DomainContinuizer.Ignore
        newdomain = transformer(data)
        newexamples = data.translate(newdomain)
        #print newexamples[0]
        params = {}
        parameters = []
        self.learner.normalization = False ## Normalization already done

        if self.svm_type in [1, 4]:
            numOfNuValues = 9
            if self.svm_type == SVMLearner.Nu_SVC:
                max_nu = max(self.max_nu(newexamples) - 1e-7, 0.0)
            else:
                max_nu = 1.0
            parameters.append(("nu", [i / 10.0 for i in range(1, 9) \
                                      if i / 10.0 < max_nu] + [max_nu]))
        else:
            parameters.append(("C", [2 ** a for a in  range(-5, 15, 2)]))
        if self.kernel_type == 2:
            parameters.append(("gamma", [2 ** a for a in range(-5, 5, 2)] + [0]))
        import orngWrap
        tunedLearner = orngWrap.TuneMParameters(object=self.learner,
                                                parameters=parameters,
                                                folds=self.folds)

        return SVMClassifierWrapper(tunedLearner(newexamples,
                                                 verbose=self.verbose))

class SVMLearnerSparseClassEasy(SVMLearnerEasy, SVMLearnerSparse):
    def __init__(self, **kwds):
        SVMLearnerSparse.__init__(self, **kwds)

def default_preprocessor():
    # Construct and return a default preprocessor for use by
    # Orange.core.LinearLearner learner.
    impute = Preprocessor_impute()
    cont = Preprocessor_continuize(multinomialTreatment=
                                   DomainContinuizer.AsOrdinal)
    preproc = Preprocessor_preprocessorList(preprocessors=
                                            [impute, cont])
    return preproc

class LinearSVMLearner(Orange.core.LinearLearner):
    """Train a linear SVM model."""

    L2R_L2LOSS_DUAL = Orange.core.LinearLearner.L2R_L2Loss_SVC_Dual
    L2R_L2LOSS = Orange.core.LinearLearner.L2R_L2Loss_SVC
    L2R_L1LOSS_DUAL = Orange.core.LinearLearner.L2R_L1Loss_SVC_Dual
    L2R_L1LOSS_DUAL = Orange.core.LinearLearner.L2R_L2Loss_SVC_Dual
    L1R_L2LOSS = Orange.core.LinearLearner.L1R_L2Loss_SVC

    __new__ = _orange__new__(base=Orange.core.LinearLearner)

    def __init__(self, solver_type=L2R_L2LOSS_DUAL, C=1.0, eps=0.01, **kwargs):
        """
        :param solver_type: One of the following class constants: ``LR2_L2LOSS_DUAL``, ``L2R_L2LOSS``, ``LR2_L1LOSS_DUAL``, ``L2R_L1LOSS`` or ``L1R_L2LOSS``
        
        :param C: Regularization parameter (default 1.0)
        :type C: float  
        
        :param eps: Stopping criteria (default 0.01)
        :type eps: float
         
        """
        self.solver_type = solver_type
        self.eps = eps
        self.C = C
        for name, val in kwargs.items():
            setattr(self, name, val)
        if self.solver_type not in [self.L2R_L2LOSS_DUAL, self.L2R_L2LOSS,
                self.L2R_L1LOSS_DUAL, self.L2R_L1LOSS_DUAL, self.L1R_L2LOSS]:
            pass
#            raise ValueError("Invalid solver_type parameter.")

        self.preproc = default_preprocessor()

    def __call__(self, instances, weight_id=None):
        instances = self.preproc(instances)
        classifier = super(LinearSVMLearner, self).__call__(instances, weight_id)
        return classifier

LinearLearner = LinearSVMLearner

class MultiClassSVMLearner(Orange.core.LinearLearner):
    """ Multi-class SVM (Crammer and Singer) from the `LIBLINEAR`_ library.
    """
    __new__ = _orange__new__(base=Orange.core.LinearLearner)

    def __init__(self, C=1.0, eps=0.01, **kwargs):
        """\
        :param C: Regularization parameter (default 1.0)
        :type C: float  
        
        :param eps: Stopping criteria (default 0.01)
        :type eps: float
        
        """
        self.C = C
        self.eps = eps
        for name, val in kwargs.items():
            setattr(self, name, val)

        self.solver_type = self.MCSVM_CS
        self.preproc = default_preprocessor()

    def __call__(self, instances, weight_id=None):
        instances = self.preproc(instances)
        classifier = super(MultiClassSVMLearner, self).__call__(instances, weight_id)
        return classifier

#TODO: Unified way to get attr weights for linear SVMs.

def get_linear_svm_weights(classifier, sum=True):
    """Extract attribute weights from the linear SVM classifier.
    
    For multi class classification, the result depends on the argument
    :obj:`sum`. If ``True`` (default) the function computes the
    squared sum of the weights over all binary one vs. one
    classifiers. If :obj:`sum` is ``False`` it returns a list of
    weights for each individual binary classifier (in the order of
    [class1 vs class2, class1 vs class3 ... class2 vs class3 ...]).
        
    """

    def update_weights(w, key, val, mul):
        if key in w:
            w[key] += mul * val
        else:
            w[key] = mul * val

    def to_float(val):
        return float(val) if not val.isSpecial() else 0.0

    SVs = classifier.support_vectors
    weights = []

    class_var = SVs.domain.class_var
    if classifier.svm_type in [SVMLearner.C_SVC, SVMLearner.Nu_SVC]:
        classes = class_var.values
    else:
        classes = [""]
    if len(classes) > 1:
        sv_ranges = [(0, classifier.nSV[0])]
        for n in classifier.nSV[1:]:
            sv_ranges.append((sv_ranges[-1][1], sv_ranges[-1][1] + n))
    else:
        sv_ranges = [(0, len(SVs))]

    for i in range(len(classes) - 1):
        for j in range(i + 1, len(classes)):
            w = {}
            coef_ind = j - 1
            for sv_ind in range(*sv_ranges[i]):
                attributes = SVs.domain.attributes + \
                SVs[sv_ind].getmetas(False, Orange.feature.Descriptor).keys()
                for attr in attributes:
                    if attr.varType == Orange.feature.Type.Continuous:
                        update_weights(w, attr, to_float(SVs[sv_ind][attr]), \
                                       classifier.coef[coef_ind][sv_ind])
            coef_ind = i
            for sv_ind in range(*sv_ranges[j]):
                attributes = SVs.domain.attributes + \
                SVs[sv_ind].getmetas(False, Orange.feature.Descriptor).keys()
                for attr in attributes:
                    if attr.varType == Orange.feature.Type.Continuous:
                        update_weights(w, attr, to_float(SVs[sv_ind][attr]), \
                                       classifier.coef[coef_ind][sv_ind])
            weights.append(w)

    if sum:
        scores = defaultdict(float)

        for w in weights:
            for attr, w_attr in w.items():
                scores[attr] += w_attr ** 2
        for key in scores:
            scores[key] = math.sqrt(scores[key])
        return scores
    else:
        return weights

getLinearSVMWeights = get_linear_svm_weights

def example_weighted_sum(example, weights):
    sum = 0
    for attr, w in weights.items():
        sum += float(example[attr]) * w
    return sum

exampleWeightedSum = example_weighted_sum

class ScoreSVMWeights(Orange.feature.scoring.Score):
    """
    Score a feature by the squared sum of weights using a linear SVM
    classifier.
        
    Example:
    
        >>> score = Orange.classification.svm.ScoreSVMWeights()
        >>> for feature in table.domain.features:
        ...     print "%15s: %.3f" % (feature.name, score(feature, table))
            compactness: 0.019
            circularity: 0.026
        distance circularity: 0.007
           radius ratio: 0.010
        pr.axis aspect ratio: 0.076
        max.length aspect ratio: 0.010
          scatter ratio: 0.046
          elongatedness: 0.094
        pr.axis rectangularity: 0.006
        max.length rectangularity: 0.031
        scaled variance along major axis: 0.001
        scaled variance along minor axis: 0.000
        scaled radius of gyration: 0.002
        skewness about major axis: 0.004
        skewness about minor axis: 0.003
        kurtosis about minor axis: 0.001
        kurtosis about major axis: 0.060
          hollows ratio: 0.028
              
    """

    def __new__(cls, attr=None, data=None, weight_id=None, **kwargs):
        self = Orange.feature.scoring.Score.__new__(cls, **kwargs)
        if data is not None and attr is not None:
            self.__init__(**kwargs)
            return self.__call__(attr, data, weight_id)
        else:
            return self

    def __reduce__(self):
        return ScoreSVMWeights, (), dict(self.__dict__)

    def __init__(self, learner=None, **kwargs):
        """
        :param learner: Learner used for weight estimation 
            (default LinearSVMLearner(solver_type=L2Loss_SVM_Dual))
        :type learner: Orange.core.LinearLearner 
        
        """
        if learner:
            self.learner = learner
        else:
            self.learner = LinearSVMLearner(solver_type=
                                    LinearSVMLearner.L2R_L2LOSS_DUAL)

        self._cached_examples = None

    def __call__(self, attr, data, weight_id=None):
        if data is self._cached_examples:
            weights = self._cached_weights
        else:
            classifier = self.learner(data, weight_id)
            self._cached_examples = data
            import numpy
            weights = numpy.array(classifier.weights)
            weights = numpy.sum(weights ** 2, axis=0)
            weights = dict(zip(data.domain.attributes, weights))
            self._cached_weights = weights
        return weights.get(attr, 0.0)

MeasureAttribute_SVMWeights = ScoreSVMWeights

class RFE(object):

    """Iterative feature elimination based on weights computed by
    linear SVM.
    
    Example::
    
        import Orange
        table = Orange.data.Table("vehicle.tab")
        l = Orange.classification.svm.SVMLearner(
            kernel_type=Orange.classification.svm.kernels.Linear, 
            normalization=False) # normalization=False will not change the domain
        rfe = Orange.classification.svm.RFE(l)
        data_subset_of_features = rfe(table, 5)
        
    """

    def __init__(self, learner=None):
        self.learner = learner or SVMLearner(kernel_type=
                            kernels.Linear, normalization=False)

    @Orange.misc.deprecated_keywords({"progressCallback": "progress_callback", "stopAt": "stop_at" })
    def get_attr_scores(self, data, stop_at=0, progress_callback=None):
        """Return a dictionary mapping attributes to scores.
        A score is a step number at which the attribute
        was removed from the recursive evaluation.
        
        """
        iter = 1
        attrs = data.domain.attributes
        attrScores = {}

        while len(attrs) > stop_at:
            weights = get_linear_svm_weights(self.learner(data), sum=False)
            if progress_callback:
                progress_callback(100. * iter / (len(attrs) - stop_at))
            score = dict.fromkeys(attrs, 0)
            for w in weights:
                for attr, wAttr in w.items():
                    score[attr] += wAttr ** 2
            score = score.items()
            score.sort(lambda a, b:cmp(a[1], b[1]))
            numToRemove = max(int(len(attrs) * 1.0 / (iter + 1)), 1)
            for attr, s in  score[:numToRemove]:
                attrScores[attr] = len(attrScores)
            attrs = [attr for attr, s in score[numToRemove:]]
            if attrs:
                data = data.select(attrs + [data.domain.classVar])
            iter += 1
        return attrScores

    @Orange.misc.deprecated_keywords({"numSelected": "num_selected", "progressCallback": "progress_callback"})
    def __call__(self, data, num_selected=20, progress_callback=None):
        """Return a new dataset with only `num_selected` best scoring attributes
        
        :param data: Data
        :type data: Orange.data.Table
        :param num_selected: number of features to preserve
        :type num_selected: int
        
        """
        scores = self.get_attr_scores(data, progress_callback=progress_callback)
        scores = sorted(scores.items(), key=lambda item: item[1])

        scores = dict(scores[-num_selected:])
        attrs = [attr for attr in data.domain.attributes if attr in scores]
        domain = Orange.data.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        data = Orange.data.Table(domain, data)
        return data

RFE = Orange.misc.deprecated_members({
    "getAttrScores": "get_attr_scores"},
    wrap_methods=["get_attr_scores", "__call__"])(RFE)

def example_table_to_svm_format(table, file):
    warnings.warn("Deprecated. Use table_to_svm_format", DeprecationWarning)
    table_to_svm_format(table, file)

exampleTableToSVMFormat = example_table_to_svm_format

def table_to_svm_format(data, file):
    """Save :obj:`Orange.data.Table` to a format used by LibSVM.
    
    :param data: Data
    :type data: Orange.data.Table
    :param file: file pointer
    :type file: file
    
    """

    attrs = data.domain.attributes + data.domain.getmetas().values()
    attrs = [attr for attr in attrs if attr.varType
             in [Orange.feature.Type.Continuous,
                 Orange.feature.Type.Discrete]]
    cv = data.domain.classVar

    for ex in data:
        if cv.varType == Orange.feature.Type.Discrete:
            file.write(str(int(ex[cv])))
        else:
            file.write(str(float(ex[cv])))

        for i, attr in enumerate(attrs):
            if not ex[attr].isSpecial():
                file.write(" " + str(i + 1) + ":" + str(float(ex[attr])))
        file.write("\n")

tableToSVMFormat = table_to_svm_format
