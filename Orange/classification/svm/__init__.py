import math

from collections import defaultdict
from operator import add

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
                        SVMClassifier as _SVMClassifier, \
                        SVMClassifierSparse as _SVMClassifierSparse

from Orange.data import preprocess

from Orange import feature as variable

from Orange.utils import _orange__new__


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


def is_discrete(feature):
    return isinstance(feature, Orange.feature.Discrete)


def is_continuous(feature):
    return isinstance(feature, Orange.feature.Continuous)


class SVMLearner(_SVMLearner):
    """
    :param svm_type: the SVM type
    :type svm_type: SVMLearner.SVMType
    :param kernel_type: the kernel type
    :type kernel_type: SVMLearner.Kernel
    :param degree: kernel parameter (only for ``Polynomial``)
    :type degree: int
    :param gamma: kernel parameter; if 0, it is set to 1.0/#features
        (for ``Polynomial``, ``RBF`` and ``Sigmoid``)
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
    :param normalization: normalize the input data prior to learning
        (default ``True``)
    :type normalization: bool
    :param weight: a list of class weights
    :type weight: list
    :param verbose: If `True` show training progress (default is `False`).
    :type verbose: bool

    Example:

        >>> import Orange
        >>> from Orange.classification import svm
        >>> from Orange.evaluation import testing, scoring
        >>> data = Orange.data.Table("vehicle.tab")
        >>> learner = svm.SVMLearner()
        >>> results = testing.cross_validation([learner], data, folds=5)
        >>> print "CA:  %.4f" % scoring.CA(results)[0]
        CA:  0.7908
        >>> print "AUC: %.4f" % scoring.AUC(results)[0]
        AUC: 0.9565

    """
    __new__ = _orange__new__(_SVMLearner)

    C_SVC = _SVMLearner.C_SVC
    Nu_SVC = _SVMLearner.Nu_SVC
    OneClass = _SVMLearner.OneClass
    Nu_SVR = _SVMLearner.Nu_SVR
    Epsilon_SVR = _SVMLearner.Epsilon_SVR

    @Orange.utils.deprecated_keywords({"kernelFunc": "kernel_func"})
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

        if self.svm_type in [3, 4] and \
            isinstance(class_var, Orange.feature.Discrete):
            self.svm_type -= 3

        if self.kernel_type == kernels.Custom and not self.kernel_func:
            raise ValueError("Custom kernel function not supplied")

        nu = self.nu
        if self.svm_type == SVMLearner.Nu_SVC:
            # Check if nu is feasible
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
            warnings.warn("One-class SVM probability output not supported.")
        return self.learn_classifier(examples)

    def learn_classifier(self, data):
        if self.normalization:
            data = self._normalize(data)
        svm = self.learner(data)
        return SVMClassifier(svm)

    @Orange.utils.deprecated_keywords({"progressCallback": "progress_callback"})
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
            if isinstance(data.domain.class_var, variable.Discrete):
                max_nu = max(self.max_nu(data) - 1e-7, 0.0)
            else:
                max_nu = 1.0
            searchParams.append(("nu", [i / 10.0 for i in range(1, 9) if \
                                        i / 10.0 < max_nu] + [max_nu]))
        elif "C" in parameters:
            searchParams.append(("C", [2 ** a for a in  range(-5, 15, 2)]))

        if self.kernel_type == 2 and "gamma" in parameters:
            searchParams.append(("gamma",
                                 [2 ** a for a in range(-5, 5, 2)] + [0])
                                )
        tunedLearner = orngWrap.TuneMParameters(object=self,
                            parameters=searchParams,
                            folds=folds,
                            returnWhat=orngWrap.TuneMParameters.returnLearner,
                            progressCallback=progress_callback
                            if progress_callback else lambda i: None)
        tunedLearner(data, verbose=verbose)
        if normalization:
            self.normalization = normalization

    def _normalize(self, data):
        dc = preprocess.DomainContinuizer()
        dc.class_treatment = preprocess.DomainContinuizer.Ignore
        dc.continuous_treatment = preprocess.DomainContinuizer.NormalizeBySpan
        dc.multinomial_treatment = preprocess.DomainContinuizer.NValues
        newdomain = dc(data)
        return data.translate(newdomain)


SVMLearner = Orange.utils.deprecated_members({
    "learnClassifier": "learn_classifier",
    "tuneParameters": "tune_parameters",
    "kernelFunc": "kernel_func",
    },
    wrap_methods=["__init__", "tune_parameters"])(SVMLearner)


class SVMClassifier(_SVMClassifier):
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], _SVMClassifier):
            # Will wrap a C++ object
            return _SVMClassifier.__new__(cls, name=args[0].name)
        elif args and isinstance(args[0], variable.Descriptor):
            # The constructor call for the C++ object.
            # This is a hack to support loading of old pickled classifiers
            return _SVMClassifier.__new__(_SVMClassifier, *args, **kwargs)
        else:
            raise ValueError

    def __init__(self, wrapped):
        self.class_var = wrapped.class_var
        self.domain = wrapped.domain
        self.computes_probabilities = wrapped.computes_probabilities
        self.examples = wrapped.examples
        self.svm_type = wrapped.svm_type
        self.kernel_func = wrapped.kernel_func
        self.kernel_type = wrapped.kernel_type
        self.__wrapped = wrapped

        assert(type(wrapped) in [_SVMClassifier, _SVMClassifierSparse])

        if self.svm_type in [SVMLearner.C_SVC, SVMLearner.Nu_SVC] \
                and len(wrapped.support_vectors) > 0:
            # Reorder the support vectors of the binary classifiers
            label_map = self._get_libsvm_labels_map()
            start = 0
            support_vectors = []
            for n in wrapped.n_SV:
                support_vectors.append(
                    wrapped.support_vectors[start: start + n]
                )
                start += n
            support_vectors = [support_vectors[i] for i in label_map \
                               if i is not None]
            support_vectors = reduce(add, support_vectors)
            self.support_vectors = Orange.data.Table(support_vectors)
        else:
            self.support_vectors = wrapped.support_vectors

    @property
    def coef(self):
        """Coefficients of the underlying svm model.

        If this is a classification model then this is a list of
        coefficients for each binary 1vs1 classifiers, i.e.
        #Classes * (#Classses - 1) list of lists where
        each sublist contains tuples of (coef, support_vector_index)

        For regression models it is still a list of lists (for consistency)
        but of length 1 e.g. [[(coef, support_vector_index), ... ]]

        """
        if is_discrete(self.class_var):
            # We need to reorder the coef values
            # see http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f804
            # for more information on how the coefs are stored by libsvm
            # internally.
            import numpy as np
            c_map = self._get_libsvm_bin_classifier_map()
            label_map = self._get_libsvm_labels_map()
            coef = []
            n_class = len(label_map)
            n_SV = self.__wrapped.n_SV
            coef_array = np.array(self.__wrapped.coef)
            p = 0
            libsvm_class_indices = np.cumsum([0] + list(n_SV), dtype=int)
            class_indices = np.cumsum([0] + list(self.n_SV), dtype=int)
            for i in range(n_class - 1):
                for j in range(i + 1, n_class):
                    ni = label_map[i]
                    nj = label_map[j]

                    if ni is None or nj is None:
                        # One of the classes is missing from the model.
                        continue

                    bc_index, mult = c_map[p]

                    if ni > nj:
                        # The order in libsvm model is switched.
                        ni, nj = nj, ni

                    # Original class indices
                    c1_range = range(libsvm_class_indices[ni],
                                     libsvm_class_indices[ni + 1])
                    c2_range = range(libsvm_class_indices[nj],
                                     libsvm_class_indices[nj + 1])

                    coef1 = mult * coef_array[nj - 1, c1_range]
                    coef2 = mult * coef_array[ni, c2_range]

                    # Mapped class indices
                    c1_range = range(class_indices[i],
                                     class_indices[i + 1])
                    c2_range = range(class_indices[j],
                                     class_indices[j + 1])
                    if mult == -1.0:
                        c1_range, c2_range = c2_range, c1_range

                    nonzero1 = np.abs(coef1) > 0.0
                    nonzero2 = np.abs(coef2) > 0.0

                    coef1 = coef1[nonzero1]
                    coef2 = coef2[nonzero2]

                    c1_range = [sv_i for sv_i, nz in zip(c1_range, nonzero1)
                                if nz]
                    c2_range = [sv_i for sv_i, nz in zip(c2_range, nonzero2)
                                if nz]

                    coef.append(list(zip(coef1, c1_range)) + \
                                list(zip(coef2, c2_range)))

                    p += 1
        else:
            coef = [zip(self.__wrapped.coef[0],
                        range(len(self.support_vectors)))]

        return coef

    @property
    def rho(self):
        """Constant (bias) terms of the svm model.

        For classification models this is a list of bias terms
        for each binary 1vs1 classifier.

        For regression models it is a list with a single value.

        """
        rho = self.__wrapped.rho
        if is_discrete(self.class_var):
            c_map = self._get_libsvm_bin_classifier_map()
            return [rho[i] * m for i, m in c_map]
        else:
            return list(rho)

    @property
    def n_SV(self):
        """Number of support vectors for each class.
        For regression models this is `None`.

        """
        n_SV = self.__wrapped.n_SV
        if n_SV is not None:
            labels_map = self._get_libsvm_labels_map()
            return [n_SV[i] if i is not None else 0 for i in labels_map]
        else:
            return None

    # Pairwise probability is expresed as:
    #   1.0 / (1.0 + exp(dec_val[i] * prob_a[i] + prob_b[i]))
    # Since dec_val already changes signs if we switch the
    # classifier direction only prob_b must change signs
    @property
    def prob_a(self):
        if self.__wrapped.prob_a is not None:
            if isinstance(self.class_var, variable.Discrete):
                c_map = self._get_libsvm_bin_classifier_map()
                prob_a = self.__wrapped.prob_a
                return [prob_a[i] for i, _ in c_map]
            else:
                # A single value for regression
                return list(self.__wrapped.prob_a)
        else:
            return None

    @property
    def prob_b(self):
        if self.__wrapped.prob_b is not None:
            c_map = self._get_libsvm_bin_classifier_map()
            prob_b = self.__wrapped.prob_b
            # Change sign when changing the classifier direction
            return [prob_b[i] * m for i, m in c_map]
        else:
            return None

    def __call__(self, instance, what=Orange.core.GetValue):
        """Classify a new ``instance``
        """
        instance = Orange.data.Instance(self.domain, instance)
        return self.__wrapped(instance, what)

    def class_distribution(self, instance):
        """Return a class distribution for the ``instance``
        """
        instance = Orange.data.Instance(self.domain, instance)
        return self.__wrapped.class_distribution(instance)

    def get_decision_values(self, instance):
        """Return the decision values of the binary 1vs1
        classifiers for the ``instance`` (:class:`~Orange.data.Instance`).

        """
        instance = Orange.data.Instance(self.domain, instance)
        dec_values = self.__wrapped.get_decision_values(instance)
        if isinstance(self.class_var, variable.Discrete):
            # decision values are ordered by libsvm internal class values
            # i.e. the order of labels in the data
            c_map = self._get_libsvm_bin_classifier_map()
            return [dec_values[i] * m for i, m in c_map]
        else:
            return list(dec_values)

    def get_model(self):
        """Return a string representing the model in the libsvm model format.
        """
        return self.__wrapped.get_model()

    def _get_libsvm_labels_map(self):
        """Get the mapping from indices in `class_var.values` to
        internal libsvm labels. If a class value is missing from the libsvm
        model the returned corresponding entry is `None`)

        """
        if is_discrete(self.class_var):
            n_classes = len(self.class_var.values)
        else:
            # OneClass/Regression models
            n_classes = 1
        model_string = self.__wrapped.get_model()
        # Get the labels definition line from the model string
        # (the labels, if present, are always integer strings
        # indexing self.class_var.values)
        labels = [line for line in model_string.splitlines() \
                  if line.startswith("label")]
        labels = labels[0].split(" ")[1:] if labels else ["0"]
        labels = [int(label) for label in labels]
        labels_map = dict((cls_index, i) for i, cls_index in enumerate(labels))
        return [labels_map.get(i) for i in range(n_classes)]

    def _get_libsvm_bin_classifier_map(self):
        """Return the libsvm binary classifier mapping (due to label ordering).
        """
        if not is_discrete(self.class_var):
            raise TypeError("SVM classification model expected")

        label_map = self._get_libsvm_labels_map()
        bin_c_map = []
        n_class_values = len(self.class_var.values)
        nr_class = len([i for i in label_map if i is not None])
        for i in range(n_class_values - 1):
            for j in range(i + 1, n_class_values):
                ni = label_map[i]
                nj = label_map[j]
                mult = 1

                if ni is None or nj is None:
                    # One or both classes are missing from the libsvm model.
                    continue

                if ni > nj:
                    # The order in libsvm is switched
                    ni, nj = nj, ni
                    mult = -1

                # classifier index
                cls_index = nr_class * (nr_class - 1) / 2 - \
                            (nr_class - ni - 1) * (nr_class - ni - 2) / 2 - \
                            (nr_class - nj)
                bin_c_map.append((cls_index, mult))
        return bin_c_map

    def __reduce__(self):
        return SVMClassifier, (self.__wrapped,), dict(self.__dict__)

    def get_binary_classifier(self, c1, c2):
        """Return a binary classifier for classes `c1` and `c2`.
        """
        import numpy as np
        if self.svm_type not in [SVMLearner.C_SVC, SVMLearner.Nu_SVC]:
            raise TypeError("SVM classification model expected.")

        c1 = int(self.class_var(c1))
        c2 = int(self.class_var(c2))

        n_class = len(self.class_var.values)

        if c1 == c2:
            raise ValueError("Different classes expected.")

        bin_class_var = Orange.feature.Discrete("%s vs %s" % \
                        (self.class_var.values[c1], self.class_var.values[c2]),
                        values=["0", "1"])

        mult = 1.0
        if c1 > c2:
            c1, c2 = c2, c1
            mult = -1.0

        classifier_i = n_class * (n_class - 1) / 2 - \
                       (n_class - c1 - 1) * (n_class - c1 - 2) / 2 - \
                       (n_class - c2)

        coef = self.coef[classifier_i]

        coef1 = [(mult * alpha, sv_i) for alpha, sv_i in coef \
                 if int(self.support_vectors[sv_i].get_class()) == c1]
        coef2 = [(mult * alpha, sv_i) for alpha, sv_i in coef \
                 if int(self.support_vectors[sv_i].get_class()) == c2]

        rho = mult * self.rho[classifier_i]

        model = self._binary_libsvm_model_string(bin_class_var,
                                                 [coef1, coef2],
                                                 [rho])

        all_sv = [self.support_vectors[sv_i] \
                  for c, sv_i in coef1 + coef2]

        all_sv = Orange.data.Table(all_sv)

        svm_classifier_type = type(self.__wrapped)

        # Build args for svm_classifier_type constructor
        args = (bin_class_var, self.examples, all_sv, model)

        if isinstance(svm_classifier_type, _SVMClassifierSparse):
            args = args + (int(self.__wrapped.use_non_meta),)

        if self.kernel_type == kernels.Custom:
            args = args + (self.kernel_func,)

        native_classifier = svm_classifier_type(*args)
        return SVMClassifier(native_classifier)

    def _binary_libsvm_model_string(self, class_var, coef, rho):
        """Return a libsvm formated model string for binary classifier
        """
        import itertools

        if not isinstance(self.class_var, variable.Discrete):
            raise TypeError("SVM classification model expected")

        model = []

        # Take the model up to nr_classes
        libsvm_model = self.__wrapped.get_model()
        for line in libsvm_model.splitlines():
            if line.startswith("nr_class"):
                break
            else:
                model.append(line.rstrip())
        nr_class = len(class_var.values)
        model.append("nr_class %i" % nr_class)
        model.append("total_sv %i" % reduce(add, [len(c) for c in coef]))
        model.append("rho " + " ".join(str(r) for r in rho))
        model.append("label " + " ".join(str(i) for i in range(nr_class)))
        # No probA and probB

        model.append("nr_sv " + " ".join(str(len(c)) for c in coef))
        model.append("SV")

        def instance_to_svm(inst):
            values = [(i, float(inst[v])) \
                      for i, v in enumerate(inst.domain.attributes) \
                      if not inst[v].is_special() and float(inst[v]) != 0.0]
            return " ".join("%i:%f" % (i + 1, v) for i, v in values)

        def sparse_instance_to_svm(inst):
            non_meta = []
            base = 1
            if self.__wrapped.use_non_meta:
                non_meta = [instance_to_svm(inst)]
                base += len(inst.domain)
            metas = []
            for m_id, value in sorted(inst.get_metas().items(), reverse=True):
                if not value.isSpecial() and float(value) != 0:
                    metas.append("%i:%f" % (base - m_id, float(value)))
            return " ".join(non_meta + metas)

        if isinstance(self.__wrapped, _SVMClassifierSparse):
            converter = sparse_instance_to_svm
        else:
            converter = instance_to_svm

        if self.kernel_type == kernels.Custom:
            SV = libsvm_model.split("SV\n", 1)[1]
            # Get the sv indices (the last entry in the SV lines)
            indices = [int(s.split(":")[-1]) for s in SV.splitlines() \
                       if s.strip()]

            # Reorder the indices
            label_map = self._get_libsvm_labels_map()
            start = 0
            reordered_indices = []
            for n in self.__wrapped.n_SV:
                reordered_indices.append(indices[start: start + n])
                start += n
            reordered_indices = [reordered_indices[i] for i in label_map]
            indices = reduce(add, reordered_indices)

            for (c, sv_i) in itertools.chain(*coef):
                model.append("%f 0:%i" % (c, indices[sv_i]))
        else:
            for (c, sv_i) in itertools.chain(*coef):
                model.append(
                    "%f %s" % (c, converter(self.support_vectors[sv_i]))
                )

        model.append("")
        return "\n".join(model)


SVMClassifier = Orange.utils.deprecated_members({
    "classDistribution": "class_distribution",
    "getDecisionValues": "get_decision_values",
    "getModel" : "get_model",
    }, wrap_methods=[])(SVMClassifier)


# Backwards compatibility (pickling)
SVMClassifierWrapper = SVMClassifier


class SVMLearnerSparse(SVMLearner):

    """
    A :class:`SVMLearner` that learns from data stored in meta
    attributes. Meta attributes do not need to be registered with the
    data set domain, or present in all data instances.
    """

    @Orange.utils.deprecated_keywords({"useNonMeta": "use_non_meta"})
    def __init__(self, **kwds):
        SVMLearner.__init__(self, **kwds)
        self.use_non_meta = kwds.get("use_non_meta", False)
        self.learner = Orange.core.SVMLearnerSparse(**kwds)

    def _normalize(self, data):
        if self.use_non_meta:
            dc = preprocess.DomainContinuizer()
            dc.class_treatment = dc.Ignore
            dc.continuous_treatment = dc.NormalizeBySpan
            dc.multinomial_treatment = dc.NValues
            newdomain = dc(data)
            data = data.translate(newdomain)
        return data


class SVMLearnerEasy(SVMLearner):
    """A class derived from :obj:`SVMLearner` that automatically
    scales the data and performs parameter optimization using
    :func:`SVMLearner.tune_parameters`. The procedure is similar to
    that implemented in easy.py script from the LibSVM package.

    """

    def __init__(self, folds=4, verbose=0, **kwargs):
        """
        :param folds: the number of folds to use in cross validation
        :type folds:  int

        :param verbose: verbosity of the tuning procedure.
        :type verbose: int

        ``kwargs`` is passed to :class:`SVMLearner`

        """
        SVMLearner.__init__(self, **kwargs)
        self.folds = folds
        self.verbose = verbose

        self.learner = SVMLearner(**kwargs)

    def learn_classifier(self, data):
        transformer = preprocess.DomainContinuizer()
        transformer.multinomialTreatment = preprocess.DomainContinuizer.NValues
        transformer.continuousTreatment = \
            preprocess.DomainContinuizer.NormalizeBySpan
        transformer.classTreatment = preprocess.DomainContinuizer.Ignore
        newdomain = transformer(data)
        newexamples = data.translate(newdomain)

        parameters = []
        self.learner.normalization = False  # Normalization already done

        if self.svm_type in [1, 4]:
            if self.svm_type == SVMLearner.Nu_SVC:
                max_nu = max(self.max_nu(newexamples) - 1e-7, 0.0)
            else:
                max_nu = 1.0
            parameters.append(("nu", [i / 10.0 for i in range(1, 9) \
                                      if i / 10.0 < max_nu] + [max_nu]))
        else:
            parameters.append(("C", [2 ** a for a in  range(-5, 15, 2)]))
        if self.kernel_type == 2:
            parameters.append(
                ("gamma", [2 ** a for a in range(-5, 5, 2)] + [0])
            )

        import orngWrap
        tunedLearner = orngWrap.TuneMParameters(learner=self.learner,
                                                parameters=parameters,
                                                folds=self.folds)

        return tunedLearner(newexamples, verbose=self.verbose)


class SVMLearnerSparseEasy(SVMLearnerEasy):
    def __init__(self, folds=4, verbose=0, **kwargs):
        SVMLearnerEasy.__init__(self, folds=folds, verbose=verbose,
                                **kwargs)
        self.learner = SVMLearnerSparse(**kwargs)


"""
LIBLINEAR learners interface
"""


class LinearSVMLearner(Orange.core.LinearLearner):
    """Train a linear SVM model."""

    L2R_L2LOSS_DUAL = Orange.core.LinearLearner.L2R_L2Loss_SVC_Dual
    L2R_L2LOSS = Orange.core.LinearLearner.L2R_L2Loss_SVC
    L2R_L1LOSS_DUAL = Orange.core.LinearLearner.L2R_L1Loss_SVC_Dual
    L1R_L2LOSS = Orange.core.LinearLearner.L1R_L2Loss_SVC

    __new__ = _orange__new__(base=Orange.core.LinearLearner)

    def __init__(self, solver_type=L2R_L2LOSS_DUAL, C=1.0, eps=0.01,
                 bias=1.0, normalization=True, **kwargs):
        """
        :param solver_type: One of the following class constants:
            ``L2R_L2LOSS_DUAL``, ``L2R_L2LOSS``,
            ``L2R_L1LOSS_DUAL``, ``L1R_L2LOSS``

            The first part (``L2R`` or ``L1R``) is the regularization term
            on the weight vector (squared or absolute norm respectively),
            the ``L1LOSS`` or ``L2LOSS`` indicate absolute or squared
            loss function ``DUAL`` means the optimization problem is
            solved in the dual space (for more information see the
            documentation on `LIBLINEAR`_).

        :param C: Regularization parameter (default 1.0)
        :type C: float

        :param eps: Stopping criteria (default 0.01)
        :type eps: float

        :param bias: If non negative then each instance is appended a constant
            bias term (default 1.0).

        :type bias: float

        :param normalization: Normalize the input data prior to learning
            (default True)
        :type normalization: bool

        Example

            >>> linear_svm = LinearSVMLearner(
            ...     solver_type=LinearSVMLearner.L1R_L2LOSS,
            ...     C=2.0)
            ...

        """
        self.solver_type = solver_type
        self.eps = eps
        self.C = C
        self.bias = bias
        self.normalization = normalization

        for name, val in kwargs.items():
            setattr(self, name, val)
        if self.solver_type not in [self.L2R_L2LOSS_DUAL, self.L2R_L2LOSS,
                self.L2R_L1LOSS_DUAL, self.L1R_L2LOSS]:
            warnings.warn("""\
Deprecated 'solver_type', use
'Orange.classification.logreg.LibLinearLogRegLearner'
to build a logistic regression model using LIBLINEAR.
""",
                DeprecationWarning)

    def __call__(self, data, weight_id=None):
        if not isinstance(data.domain.class_var, variable.Discrete):
            raise TypeError("Can only learn a discrete class.")

        if data.domain.has_discrete_attributes(False) or self.normalization:
            dc = Orange.data.continuization.DomainContinuizer()
            dc.multinomial_treatment = dc.NValues
            dc.class_treatment = dc.Ignore
            dc.continuous_treatment = \
                    dc.NormalizeBySpan if self.normalization else dc.Leave
            c_domain = dc(data)
            data = data.translate(c_domain)

        return super(LinearSVMLearner, self).__call__(data, weight_id)

LinearLearner = LinearSVMLearner


class MultiClassSVMLearner(Orange.core.LinearLearner):
    """ Multi-class SVM (Crammer and Singer) from the `LIBLINEAR`_ library.
    """
    __new__ = _orange__new__(base=Orange.core.LinearLearner)

    def __init__(self, C=1.0, eps=0.01, bias=1.0,
                 normalization=True, **kwargs):
        """\
        :param C: Regularization parameter (default 1.0)
        :type C: float

        :param eps: Stopping criteria (default 0.01)
        :type eps: float

        :param bias: If non negative then each instance is appended a constant
            bias term (default 1.0).

        :type bias: float

        :param normalization: Normalize the input data prior to learning
            (default True)
        :type normalization: bool

        """
        self.C = C
        self.eps = eps
        self.bias = bias
        self.normalization = normalization
        for name, val in kwargs.items():
            setattr(self, name, val)

        self.solver_type = self.MCSVM_CS

    def __call__(self, data, weight_id=None):
        if not isinstance(data.domain.class_var, variable.Discrete):
            raise TypeError("Can only learn a discrete class.")

        if data.domain.has_discrete_attributes(False) or self.normalization:
            dc = Orange.data.continuization.DomainContinuizer()
            dc.multinomial_treatment = dc.NValues
            dc.class_treatment = dc.Ignore
            dc.continuous_treatment = \
                    dc.NormalizeBySpan if self.normalization else dc.Leave
            c_domain = dc(data)
            data = data.translate(c_domain)

        return super(MultiClassSVMLearner, self).__call__(data, weight_id)

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
    class_var = SVs.domain.class_var

    if classifier.svm_type in [SVMLearner.C_SVC, SVMLearner.Nu_SVC]:
        weights = []
        classes = classifier.class_var.values
        for i in range(len(classes) - 1):
            for j in range(i + 1, len(classes)):
                # Get the coef and rho values from the binary sub-classifier
                # Easier then using the full coef matrix (due to libsvm
                # internal class  reordering)
                bin_classifier = classifier.get_binary_classifier(i, j)
                n_sv0 = bin_classifier.n_SV[0]
                SVs = bin_classifier.support_vectors
                w = {}

                for coef, sv_ind in bin_classifier.coef[0]:
                    SV = SVs[sv_ind]
                    attributes = SVs.domain.attributes + \
                    SV.getmetas(False, Orange.feature.Descriptor).keys()
                    for attr in attributes:
                        if attr.varType == Orange.feature.Type.Continuous:
                            update_weights(w, attr, to_float(SV[attr]), coef)

                weights.append(w)
        if sum:
            scores = defaultdict(float)
            for w in weights:
                for attr, w_attr in w.items():
                    scores[attr] += w_attr ** 2
            for key in scores:
                scores[key] = math.sqrt(scores[key])
            weights = dict(scores)
    else:
        weights = {}
        for coef, sv_ind in classifier.coef[0]:
            SV = SVs[sv_ind]
            attributes = SVs.domain.attributes + \
            SV.getmetas(False, Orange.feature.Descriptor).keys()
            for attr in attributes:
                if attr.varType == Orange.feature.Type.Continuous:
                    update_weights(weights, attr, to_float(SV[attr]), coef)

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
    Score a feature using squared weights of a linear SVM model.

    Example:

        >>> table = Orange.data.Table("vehicle.tab")
        >>> score = Orange.classification.svm.ScoreSVMWeights()
        >>> svm_scores = [(score(f, table), f) for f in table.domain.features]
        >>> for feature_score, feature in sorted(svm_scores, reverse=True):
        ...     print "%-35s: %.3f" % (feature.name, feature_score)
        pr.axis aspect ratio               : 44.263
        kurtosis about major axis          : 42.593
        max.length rectangularity          : 39.377
        radius ratio                       : 28.741
        skewness about major axis          : 26.682
        hollows ratio                      : 20.993
        compactness                        : 20.085
        elongatedness                      : 17.410
        distance circularity               : 14.542
        scaled radius of gyration          : 12.584
        max.length aspect ratio            : 10.686
        scatter ratio                      : 10.574
        scaled variance along minor axis   : 10.049
        circularity                        : 8.360
        pr.axis rectangularity             : 7.473
        scaled variance along major axis   : 5.731
        skewness about minor axis          : 1.368
        kurtosis about minor axis          : 0.690


    """

    handles_discrete = True
    handles_continuous = True
    computes_thresholds = False
    needs = Orange.feature.scoring.Score.Generator

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
            (by default
            ``LinearSVMLearner(solver_type=L2R_L2LOSS_DUAL, C=1.0)``
            will be used for classification problems and
            ``SVMLearner(svm_type=Epsilon_SVR, kernel_type=Linear, C=1.0, p=0.25)``
            for regression problems).

        :type learner: Orange.core.LinearLearner

        """
        self.learner = learner
        self._cached_data = None
        self._cached_data_crc = None
        self._cached_weights = None
        self._cached_classifier = None

    def __call__(self, attr, data, weight_id=None):
        if attr not in data.domain.attributes:
            raise ValueError("Feature %r is not from the domain." % attr)

        if self.learner is not None:
            learner = self.learner
        elif isinstance(data.domain.class_var, variable.Discrete):
            learner = LinearSVMLearner(
                            solver_type=LinearSVMLearner.L2R_L2LOSS_DUAL,
                            C=1.0)

        elif isinstance(data.domain.class_var, variable.Continuous):
            learner = SVMLearner(svm_type=SVMLearner.Epsilon_SVR,
                                 kernel_type=kernels.Linear,
                                 C=1.0, p=0.25)
        else:
            raise TypeError("Cannot handle the class variable type %r" % \
                                type(data.domain.class_var))

        crc = data.checksum()
        if data is self._cached_data and crc == self._cached_data_crc:
            weights = self._cached_weights
        else:
            classifier = learner(data, weight_id)
            self._cached_data = data
            self._cached_data_crc = data.checksum()
            self._cached_classifier = classifier
            weights = self._extract_weights(classifier, data.domain.attributes)
            self._cached_weights = weights
        return weights.get(attr, 0.0)

    def _extract_weights(self, classifier, original_features):
        """Extract weights from a svm classifer (``SVMClassifier`` or a
        ``LinearLearner`` instance).

        """
        import numpy as np
        if isinstance(classifier, SVMClassifier):
            weights = get_linear_svm_weights(classifier, sum=True)
            if isinstance(classifier.class_var, variable.Continuous):
                # The weights are in the the original non squared form
                weights = dict((f, w ** 2) for f, w in weights.items())
        elif isinstance(classifier, Orange.core.LinearClassifier):
            weights = np.array(classifier.weights)
            weights = np.sum(weights ** 2, axis=0)
            weights = dict(zip(classifier.domain.attributes, weights))
        else:
            raise TypeError("Don't know how to use classifier type %r" % \
                                type(classifier))

        # collect dummy variables that were created for discrete features
        sources = self._collect_source(weights.keys())
        source_weights = dict.fromkeys(original_features, 0.0)
        for f in original_features:
            if f in weights:
                source_weights[f] = weights[f]
            elif f not in weights and f in sources:
                dummys = sources[f]
                # Use averege weight
                source_weights[f] = np.average([weights[d] for d in dummys])
            else:
                raise ValueError(f)

        return source_weights

    def _collect_source(self, vars):
        """ Given a list of variables ``var``, return a mapping from source
        variables (``source_variable`` or ``get_value_from.variable`` members)
        back to the variables in ``vars``.

        """
        source = defaultdict(list)
        for var in vars:
            if var.source_variable:
                source[var.source_variable].append(var)
            elif isinstance(var.get_value_from, Orange.core.ClassifierFromVar):
                source[var.get_value_from.variable].append(var)
            elif isinstance(var.get_value_from, Orange.core.ImputeClassifier):
                imputer = var.get_value_from.classifier_from_var
                source[imputer.variable].append(var)
            else:
                source[var].append(var)
        return dict(source)

MeasureAttribute_SVMWeights = ScoreSVMWeights


class RFE(object):

    """Iterative feature elimination based on weights computed by
    linear SVM.

    Example::

        >>> table = Orange.data.Table("promoters.tab")
        >>> svm_l = Orange.classification.svm.SVMLearner(
        ...     kernel_type=Orange.classification.svm.kernels.Linear)
        ...
        >>> rfe = Orange.classification.svm.RFE(learner=svm_l)
        >>> data_with_subset_of_features = rfe(table, 10)
        >>> data_with_subset_of_features.domain
        [p-45, p-36, p-35, p-34, p-33, p-31, p-18, p-12, p-10, p-04, y]

    """

    def __init__(self, learner=None):
        """
        :param learner: A linear svm learner for use with
            :class:`ScoreSVMWeights`.

        """
        self.learner = learner

    @Orange.utils.deprecated_keywords({"progressCallback": "progress_callback",
                                       "stopAt": "stop_at"})
    def get_attr_scores(self, data, stop_at=0, progress_callback=None):
        """Return a dictionary mapping attributes to scores.
        A score is a step number at which the attribute
        was removed from the recursive evaluation.

        """
        iter = 1
        attrs = data.domain.attributes
        attr_scores = {}
        scorer = ScoreSVMWeights(learner=self.learner)

        while len(attrs) > stop_at:
            scores = [(scorer(attr, data), attr) for attr in attrs]
            if progress_callback:
                progress_callback(100. * iter / (len(attrs) - stop_at))
            scores = sorted(scores)
            num_to_remove = max(int(len(attrs) * 1.0 / (iter + 1)), 1)
            for s, attr in  scores[:num_to_remove]:
                attr_scores[attr] = len(attr_scores)
            attrs = [attr for s, attr in scores[num_to_remove:]]
            if attrs:
                data = data.select(attrs + [data.domain.class_var])
            iter += 1
        return attr_scores

    @Orange.utils.deprecated_keywords(
        {"numSelected": "num_selected",
         "progressCallback": "progress_callback"})
    def __call__(self, data, num_selected=20, progress_callback=None):
        """Return a new dataset with only `num_selected` best scoring
        attributes.

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


RFE = Orange.utils.deprecated_members({
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
