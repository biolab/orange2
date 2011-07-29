"""\
====================================================
Multivariate Adaptive Regression Splines (``earth``)
====================================================

`Multivariate adaptive regression splines (MARS)`_ is a non-parametric
regression method that extends a linear model with non-linear
interactions.

This module borrows the implementation of the technique from the `Earth R 
package`_ by Stephen Milborrow. 

.. _`Multivariate adaptive regression splines (MARS)`: http://en.wikipedia.org/wiki/Multivariate_adaptive_regression_splines
.. _`Earth R package`: http://cran.r-project.org/web/packages/earth/index.html

.. autoclass:: EarthLearner

.. autoclass:: EarthClassifier

.. autoclass:: EarthLearnerML

.. autoclass:: EarthClassifierML

"""

import Orange
from Orange.core import (EarthLearner as BaseEarthLearner,
                         EarthClassifier as BaseEarthClassifier)
            
import numpy

class EarthLearner(Orange.core.LearnerFD):
    """ Earth learner class.
    """
    def __new__(cls, examples=None, weight_id=None, **kwargs):
        self = Orange.core.LearnerFD.__new__(cls)
        if examples is not None:
            self.__init__(**kwargs)
            return self.__call__(examples, weight_id)
        else:
            return self
        
    def __init__(self, degree=1, terms=21, penalty= None, thresh=1e-3,
                 min_span=0, new_var_penalty=0, fast_k=20, fast_beta=1,
                 pruned_terms=None, scale_resp=False, store_examples=True,
                 multi_label=False, **kwds):
        """ 
        .. todo:: min_span, prunning_method
        """
        self.degree = degree
        self.terms = terms
        if penalty is None:
            penalty = 3 if degree > 1 else 2
        self.penalty = penalty 
        self.thresh = thresh
        self.min_span = min_span
        self.new_var_penalty = new_var_penalty
        self.fast_k = fast_k
        self.fast_beta = fast_beta
        self.pruned_terms = pruned_terms
        self.scale_resp = scale_resp
        self.store_examples = store_examples
        self.multi_label = multi_label
        self.__dict__.update(kwds)
        
    def __call__(self, examples, weight_id=None):
        if self.multi_label:
            label_mask = data_label_mask(examples.domain)
        else:
            label_mask = numpy.zeros(len(examples.domain.variables),
                                     dtype=bool)
            label_mask[-1] = bool(examples.domain.class_var)
            
        if not any(label_mask):
            raise ValueError("The domain has no response variable.")
         
        data = examples.to_numpy_MA("Ac")[0]
        y = data[:, label_mask]
        x = data[:, ~ label_mask]
        
        # TODO: y scaling
        n_terms, used, bx, dirs, cuts = forward_pass(x, y,
            degree=self.degree, terms=self.terms, penalty=self.penalty,
            thresh=self.thresh, fast_k=self.fast_k, fast_beta=self.fast_beta,
            new_var_penalty=self.new_var_penalty)
        
        # discard unused terms from bx, dirs, cuts
        bx = bx[:, used]
        dirs = dirs[used, :]
        cuts = cuts[used, :]
        
        # pruning
        used, subsets, rss_per_subset, gcv_per_subset = \
            pruning_pass(bx, y, self.penalty,
                         pruned_terms=self.pruned_terms)
        
        # Fit betas
        bx_used = bx[:, used]
        betas, res, rank, s = numpy.linalg.lstsq(bx_used, y)
        
        return EarthClassifier(examples.domain, used, dirs, cuts, betas.T,
                               subsets, rss_per_subset, gcv_per_subset,
                               examples=examples if self.store_examples else None,
                               label_mask=label_mask, multi_flag=self.multi_label)
    

class EarthClassifier(Orange.core.ClassifierFD):
    """ Earth classifier.
    """
    def __init__(self, domain, best_set, dirs, cuts, betas, subsets=None,
                 rss_per_subset=None, gcv_per_subset=None, examples=None,
                 label_mask=None, multi_flag=False, **kwargs):
        self.multi_flag = multi_flag
        self.domain = domain
        self.class_var = domain.class_var
        self.best_set = best_set
        self.dirs = dirs
        self.cuts = cuts
        self.betas = betas
        self.subsets = subsets
        self.rss_per_subset = rss_per_subset
        self.gcv_per_subset = gcv_per_subset
        self.examples = examples
        self.label_mask = label_mask
        self.__dict__.update(kwargs)
        
    def __call__(self, example, result_type=Orange.core.GetValue):
        resp_vars = [v for v, m in zip(self.domain.variables, self.label_mask)\
                     if m]
        vals = self.predict(example)
        vals = [var(val) for var, val in zip(resp_vars, vals)]
        
        probs = []
        for var, val in zip(resp_vars, vals):
            dist = Orange.statistics.distribution.Distribution(var)
            dist[val] = 1.0
            probs.append(dist)
            
        if not self.multi_flag:
            vals, probs = vals[0], probs[0]
            
        if result_type == Orange.core.GetValue:
            return vals
        elif result_type == Orange.core.GetBoth:
            return zip(vals, probs) if self.multi_flag else (vals, probs)
        else:
            return probs
    
    def format_model(self, percision=3, indent=3):
        """ Return a string representation of the model.
        """
        return format_model(self, percision, indent)
    
    def print_model(self, percision=3, indent=3):
        """ Print the model to stdout.
        """
        print self.format_model(percision, indent)
        
    def base_matrix(self, examples=None):
        """Return the base matrix (bx) of the Earth model for the table.
        If table is not supplied the base matrix of the training examples 
        is returned.
        
        :param examples: Input examples for the base matrix.
        :type examples: :class:`Orange.data.Table` 
        
        """
        if examples is None:
            examples = self.examples
        (data,) = examples.to_numpy_MA("Ac")
        data = data[:, ~ self.label_mask]
        bx = base_matrix(data, self.best_set, self.dirs, self.cuts)
        return bx
    
    def predict(self, example):
        """ Predict the response values for the example
        
        :param example: example instance
        :type example: :class:`Orange.data.Example`
        """
        data = Orange.data.Table(self.domain, [example])
        bx = self.base_matrix(data)
        bx_used = bx[:, self.best_set]
        vals = numpy.dot(bx_used, self.betas.T).ravel()
        return vals
    
    def used_attributes(self, term=None):
        """ Return the used terms for term (index). If no term is given
        return all attributes in the model.
        
        :param term: term index
        :type term: int
        
        """
        if term is None:
            return reduce(set.union, [self.used_attributes(i) \
                                      for i in range(self.best_set.size)],
                          set())
        attrs = [a for a, m in zip(self.domain.variables, self.label_mask)
                 if not m]
        
        used_mask = self.dirs[term, :] != 0.0
        return [a for a, u in zip(attrs, used_mask) if u]
    
    def evimp(self, used_only=True):
        """ Return the estimated variable importances.
        
        :param used_only: if True return only used attributes
         
        """  
        return evimp(self, used_only)
    
    def __reduce__(self):
        return (type(self), (self.domain, self.best_set, self.dirs,
                            self.cuts, self.betas),
                dict(self.__dict__))

"""
Utility functions
-----------------
"""

from Orange.misc.render import contextmanager

@contextmanager 
def member_set(obj, name, val):
    """ A context manager that sets member ``name`` on ``obj`` to ``val``
    and restores the previous value on exit. 
    """
    old_val = getattr(obj, name, val)
    setattr(obj, name, val)
    yield
    setattr(obj, name, old_val)
    
    
def base_matrix(data, best_set, dirs, cuts):
    """ Return the base matrix for the earth model.
    
    :param data: Input data
    :type data: :class:`numpy.ndarray`
    
    :param best_set: A array of booleans indicating used terms.
    :type best_set: :class:`numpy.ndarray`
    
    :param dirs: Earth model's dirs members
    :type dirs: :class:`numpy.ndarray`
    
    :param cuts: Earth model's cuts members
    :type cuts: :class:`numpy.ndarray`
    
    """
    data = numpy.asarray(data)
    best_set = numpy.asarray(best_set)
    dirs = numpy.asarray(dirs)
    cuts = numpy.asarray(cuts)
    
    bx = numpy.zeros((data.shape[0], best_set.shape[0]))
    bx[:, 0] = 1.0 # The intercept
    for termi in range(1, best_set.shape[0]):
        term_dirs = dirs[termi]
        term_cuts = cuts[termi]
        
        dir_p1 = numpy.where(term_dirs == 1)[0]
        dir_m1 = numpy.where(term_dirs == -1)[0]
        dir_2 = numpy.where(term_dirs == 2)[0]
        
        x1 = data[:, dir_p1] - term_cuts[dir_p1]
        x2 = term_cuts[dir_m1] - data[:, dir_m1]
        x3 = data[:, dir_2]
        
        x1 = numpy.where(x1 > 0.0, x1, 0.0)
        x2 = numpy.where(x2 > 0.0, x2, 0.0)
        
        X = numpy.hstack((x1, x2, x3))
        X = numpy.cumprod(X, axis=1)
        bx[:, termi] = X[:, -1] if X.size else 0.0
        
    return bx

    
def gcv(rss, n, n_effective_params):
    """ Return the generalized cross validation.
    
    .. math: gcv = rss / (n * (1 - n_effective_params / n) ^ 2)
    
    :param rss: Residual sum of squares.
    :param n: Number of training examples.
    :param n_effective_params: Number of effective paramaters.
     
    """
    return  rss / (n * (1 - n_effective_params / n) ** 2)
    

def subsets_selection_xtx_numpy(X, Y):
    """ A numpy implementation of EvalSubsetsUsingXtx in the Earth package. 
    """
    X = numpy.asarray(X)
    Y = numpy.asarray(Y)
    
    var_count= X.shape[1]
    rss_vec = numpy.zeros(var_count)
    working_set = range(var_count)
    subsets = numpy.zeros((var_count, var_count), dtype=int)
    
    for subset_size in reversed(range(var_count)):
        subsets[subset_size, :subset_size + 1] = working_set
        X_work = X[:, working_set]
        b, res, rank, s = numpy.linalg.lstsq(X_work, Y)
        if res.size > 0:
            rss_vec[subset_size] = numpy.sum(res)
        else:
            rss_vec[subset_size] = numpy.sum((Y - numpy.dot(X_work, b)) ** 2)
            
        XtX = numpy.dot(X_work.T, X_work)
        iXtX = numpy.linalg.pinv(XtX)
        diag = numpy.diag(iXtX)
        
        if subset_size == 0:
            break
        
        delta_rss = b ** 2 / diag
        delete_i = numpy.argmin(delta_rss[1:]) + 1 # Keep the intercept
        del working_set[delete_i]
    return subsets, rss_vec


"""
Multi-label utility functions
"""

def is_label_attr(attr):
    """ Is attribute a label.
    """
    return attr.attributes.has_key("label")
    
def data_label_indices(domain):
    """ Return the indices of label attributes in data.
    """
    return numpy.where(data_label_mask(domain))[0]

def data_label_mask(domain):
    """ Return an array of booleans indicating whether a variable in the
    domain is a label.
    """
    is_label = map(is_label_attr, domain.variables)
    if domain.class_var:
        is_label[-1] = True
    return numpy.array(is_label, dtype=bool)

"""
ctypes interface to ForwardPass and EvalSubsetsUsingXtx.
"""
        
import ctypes
from numpy import ctypeslib
import orange

_c_orange_lib = ctypeslib.load_library(orange.__file__, "")
_c_forward_pass_ = _c_orange_lib.EarthForwardPass

_c_forward_pass_.argtypes = \
    [ctypes.POINTER(ctypes.c_int),  #pnTerms:
     ctypeslib.ndpointer(dtype=numpy.bool, ndim=1),  #FullSet
     ctypeslib.ndpointer(dtype=numpy.double, ndim=2, flags="F_CONTIGUOUS"), #bx
     ctypeslib.ndpointer(dtype=numpy.int, ndim=2, flags="F_CONTIGUOUS"),    #Dirs
     ctypeslib.ndpointer(dtype=numpy.double, ndim=2, flags="F_CONTIGUOUS"), #Cuts
     ctypeslib.ndpointer(dtype=numpy.int, ndim=1),  #nFactorsInTerms
     ctypeslib.ndpointer(dtype=numpy.int, ndim=1),  #nUses
     ctypeslib.ndpointer(dtype=numpy.double, ndim=2, flags="F_CONTIGUOUS"), #x
     ctypeslib.ndpointer(dtype=numpy.double, ndim=2, flags="F_CONTIGUOUS"), #y
     ctypeslib.ndpointer(dtype=numpy.double, ndim=1), # Weights
     ctypes.c_int,  #nCases
     ctypes.c_int,  #nResp
     ctypes.c_int,  #nPred
     ctypes.c_int,  #nMaxDegree
     ctypes.c_int,  #nMaxTerms
     ctypes.c_double,   #Penalty
     ctypes.c_double,   #Thresh
     ctypes.c_int,  #nFastK
     ctypes.c_double,   #FastBeta
     ctypes.c_double,   #NewVarPenalty
     ctypeslib.ndpointer(dtype=numpy.int, ndim=1),  #LinPreds
     ctypes.c_bool, #UseBetaCache
     ctypes.c_char_p    #sPredNames
     ]
    
def forward_pass(x, y, degree=1, terms=21, penalty=None, thresh=0.001,
                  fast_k=21, fast_beta=1, new_var_penalty=2):
    """ Do earth forward pass.
    """
    import ctypes, orange
    x = numpy.asfortranarray(x, dtype="d")
    y = numpy.asfortranarray(y, dtype="d")
    if x.shape[0] != y.shape[0]:
        raise ValueError("First dimensions of x and y must be the same.")
    if y.ndim == 1:
        y = y.reshape((-1, 1), order="F")
    if penalty is None:
        penalty = 2
    n_cases = x.shape[0]
    n_preds = x.shape[1]
    
    n_resp = y.shape[1] if y.ndim == 2 else y.shape[0]
    
    # Output variables
    n_term = ctypes.c_int()
    full_set = numpy.zeros((terms,), dtype=numpy.bool, order="F")
    bx = numpy.zeros((n_cases, terms), dtype=numpy.double, order="F")
    dirs = numpy.zeros((terms, n_preds), dtype=numpy.int, order="F")
    cuts = numpy.zeros((terms, n_preds), dtype=numpy.double, order="F")
    n_factors_in_terms = numpy.zeros((terms,), dtype=numpy.int, order="F")
    n_uses = numpy.zeros((n_preds,), dtype=numpy.int, order="F")
    weights = numpy.ones((n_cases,), dtype=numpy.double, order="F")
    lin_preds = numpy.zeros((n_preds,), dtype=numpy.int, order="F")
    use_beta_cache = True
    
    _c_forward_pass_(ctypes.byref(n_term), full_set, bx, dirs, cuts,
                     n_factors_in_terms, n_uses, x, y, weights, n_cases,
                     n_resp, n_preds, degree, terms, penalty, thresh,
                     fast_k, fast_beta, new_var_penalty, lin_preds, 
                     use_beta_cache, None)
    return n_term.value, full_set, bx, dirs, cuts


_c_eval_subsets_xtx = _c_orange_lib.EarthEvalSubsetsUsingXtx

_c_eval_subsets_xtx.argtypes = \
    [ctypeslib.ndpointer(dtype=numpy.bool, ndim=2, flags="F_CONTIGUOUS"),   #PruneTerms
     ctypeslib.ndpointer(dtype=numpy.double, ndim=1),   #RssVec
     ctypes.c_int,  #nCases
     ctypes.c_int,  #nResp
     ctypes.c_int,  #nMaxTerms
     ctypeslib.ndpointer(dtype=numpy.double, ndim=2, flags="F_CONTIGUOUS"),   #bx
     ctypeslib.ndpointer(dtype=numpy.double, ndim=2, flags="F_CONTIGUOUS"),   #y
     ctypeslib.ndpointer(dtype=numpy.double, ndim=1)  #WeightsArg
     ]

def subset_selection_xtx(X, Y):
    """ Subsets selection using EvalSubsetsUsingXtx in the Earth package.
    """
    X = numpy.asfortranarray(X, dtype=numpy.double)
    Y = numpy.asfortranarray(Y, dtype=numpy.double)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1), order="F")
        
    if X.shape[0] != Y.shape[0]:
        raise ValueError("First dimensions of bx and y must be the same")
        
    var_count = X.shape[1]
    resp_count = Y.shape[1]
    cases = X.shape[0]
    subsets = numpy.zeros((var_count, var_count), dtype=numpy.bool,
                              order="F")
    rss_vec = numpy.zeros((var_count,), dtype=numpy.double, order="F")
    weights = numpy.ones((cases,), dtype=numpy.double, order="F")
    
    _c_eval_subsets_xtx(subsets, rss_vec, cases, resp_count, var_count,
                        X, Y, weights)
    
    subsets_ind = numpy.zeros((var_count, var_count), dtype=int)
    for i, used in enumerate(subsets.T):
        subsets_ind[i, :i + 1] = numpy.where(used)[0]
        
    return subsets_ind, rss_vec
    
    
def pruning_pass(bx, y, penalty, pruned_terms=-1):
    """ Do pruning pass
    
    .. todo:: leaps
    
    """
    subsets, rss_vec = subset_selection_xtx(bx, y)
    
    cases, terms = bx.shape
    n_effective_params = numpy.arange(terms) + 1.0
    n_effective_params += penalty * (n_effective_params - 1.0) / 2.0
    
    gcv_vec = gcv(rss_vec, cases, n_effective_params)
    
    min_i = numpy.argmin(gcv_vec)
    used = numpy.zeros((terms), dtype=bool)
    
    used[subsets[min_i, :min_i + 1]] = True
    
    return used, subsets, rss_vec, gcv_vec
    
    
def evimp(model, used_only=True):
    """ Return the estimated variable importance for the model.
    
    :param model: Earth model.
    :type model: `EarthClassifier`
    
    """
    if model.subsets is None:
        raise ValueError("No subsets. Use the learner with 'prune=True'.")
    
    subsets = model.subsets
    n_subsets = numpy.sum(model.best_set)
    
    rss = -numpy.diff(model.rss_per_subset)
    gcv = -numpy.diff(model.gcv_per_subset)
    attributes = list(model.domain.variables)
    
    attr2ind = dict(zip(attributes, range(len(attributes))))
    importances = numpy.zeros((len(attributes), 4))
    importances[:, 0] = range(len(attributes))
    
    for i in range(1, n_subsets):
        term_subset = subsets[i, :i + 1]
        used_attributes = reduce(set.union, [model.used_attributes(term) \
                                             for term in term_subset], set())
        for attr in used_attributes:
            importances[attr2ind[attr]][1] += 1.0
            importances[attr2ind[attr]][2] += gcv[i - 1]
            importances[attr2ind[attr]][3] += rss[i - 1]
    imp_min = numpy.min(importances[:, [2, 3]], axis=0)
    imp_max = numpy.max(importances[:, [2, 3]], axis=0)
    
    #Normalize importances.
    importances[:, [2, 3]] = 100.0 * (importances[:, [2, 3]] \
                            - [imp_min]) / ([imp_max - imp_min])
    
    importances = list(importances)
    # Sort by n_subsets and gcv.
    importances = sorted(importances, key=lambda row: (row[1], row[2]),
                         reverse=True)
    importances = numpy.array(importances)
    
    if used_only:
        importances = importances[importances[:,1] > 0.0]
    
    res = [(attributes[int(row[0])], tuple(row[1:])) for row in importances]
    return res


def plot_evimp(evimp):
    """ Plot the return value from :obj:`EarthClassifier.evimp` call.
    """
    import pylab
    fig = pylab.figure()
    axes1 = fig.add_subplot(111)
    attrs = [a for a, _ in evimp]
    imp = [s for _, s in evimp]
    imp = numpy.array(imp)
    X = range(len(attrs))
    l1 = axes1.plot(X, imp[:,0], "b-",)
    axes2 = axes1.twinx()
    
    l2 = axes2.plot(X, imp[:,1], "g-",)
    l3 = axes2.plot(X, imp[:,2], "r-",)
    
    x_axis = axes1.xaxis
    x_axis.set_ticks(X)
    x_axis.set_ticklabels([a.name for a in attrs], rotation=45)
    
    axes1.yaxis.set_label_text("nsubsets")
    axes2.yaxis.set_label_text("normalizes gcc or rss")

    axes1.legend([l1, l2, l3], ["nsubsets", "gcv", "rss"])
    axes1.set_title("Variable importance")
    fig.show()
    

"""
Printing functions.
"""

def print_model(model, percision=3, indent=3):
    """ Print model to stdout.
    """
    print format_model(model, percision, indent)
    
def format_model(model, percision=3, indent=3):
    """ Return a formated string representation of the model.
    """
    mask = model.label_mask
    r_vars = [v for v, m in zip(model.domain.variables,
                                model.label_mask)
              if m]
    r_names = [v.name for v in r_vars]
    betas = model.betas
        
    resp = []
    for name, betas in zip(r_names, betas):
        resp.append(_format_response(model, name, betas,
                                     percision, indent))
    return "\n\n".join(resp)

def _format_response(model, resp_name, betas, percision=3, indent=3):
    header = "%s =" % resp_name
    indent = " " * indent
    fmt = "%." + str(percision) + "f"
    terms = [([], fmt % betas[0])]
    beta_i = 0
    for i, used in enumerate(model.best_set[1:], 1):
        if used:
            beta_i += 1
            beta = fmt % abs(betas[beta_i])
            knots = [_format_knot(model, attr.name, d, c) for d, c, attr in \
                     zip(model.dirs[i], model.cuts[i], model.domain.attributes) \
                     if d != 0]
            term_attrs = [a for a, d in zip(model.domain.attributes, model.dirs[i]) \
                          if d != 0]
            term_attrs = sorted(term_attrs)
            sign = "-" if betas[beta_i] < 0 else "+"
            if knots:
                terms.append((term_attrs,
                              sign + " * ".join([beta] + knots)))
            else:
                terms.append((term_attrs, sign + beta))
    # Sort by len(term_attrs), then by term_attrs
    terms = sorted(terms, key=lambda t: (len(t[0]), t[0]))
    return "\n".join([header] + [indent + t for _, t in terms])
        
def _format_knot(model, name, dir, cut):
    if dir == 1:
        txt = "max(0, %s - %.3f)" % (name, cut)
    elif dir == -1:
        txt = "max(0, %.3f - %s)" % (cut, name)
    elif dir == 2:
        txt = name
    return txt

def _format_term(model, i, attr_name):
    knots = [_format_knot(model, attr, d, c) for d, c, attr in \
             zip(model.dirs[i], model.cuts[i], model.domain.attributes) \
             if d != 0]
    return " * ".join(knots)



#class _EarthLearner(BaseEarthLearner):
#    """ An earth learner. 
#    """
#    def __new__(cls, data=None, weightId=None, **kwargs):
#        self = BaseEarthLearner.__new__(cls, **kwargs)
#        if data is not None:
#            self.__init__(**kwargs)
#            return self.__call__(data, weightId)
#        return self
#    
#    def __init__(self, max_degree=1, max_terms=21, new_var_penalty=0.0,
#                 threshold=0.001, prune=True, penalty=None, fast_k=20,
#                 fast_beta=0.0, store_examples=True, **kwargs):
#        """ Initialize the learner instance.
#        
#        :param max_degree:
#        """
#        self.max_degree = max_degree
#        self.max_terms = max_terms
#        self.new_var_penalty = new_var_penalty
#        self.threshold = threshold
#        self.prune = prunes
#        if penaty is None:
#            penalty = 2.0 if degree > 1 else 3.0
#        self.penalty = penalty
#        self.fast_k = fast_k
#        self.fast_beta = fast_beta
#        self.store_examples = store_examples
#        
#        for key, val in kwargs.items():
#            setattr(self, key, val)
#    
#    def __call__(self, data, weightId=None):
#        if not data.domain.class_var:
#            raise ValueError("No class var in the domain.")
#        
#        with member_set(self, "prune", False):
#            # We overwrite the prune argument (will do the pruning in python).
#            base_clsf =  BaseEarthLearner.__call__(self, data, weightId)
#        
#        if self.prune:
#            (best_set, betas, rss, subsets, rss_per_subset,
#             gcv_per_subset) = self.pruning_pass(base_clsf, data)
#            
#            return _EarthClassifier(base_clsf, data if self.store_examples else None,
#                                   best_set=best_set, dirs=base_clsf.dirs,
#                                   cuts=base_clsf.cuts,
#                                   betas=betas,
#                                   subsets=subsets,
#                                   rss_per_subset=rss_per_subset,
#                                   gcv_per_subset=gcv_per_subset)
#        else:
#            return _EarthClassifier(base_clsf, data if self.store_examples else None)
#    
#    
#    def pruning_pass(self, base_clsf, examples):
#        """ Prune the terms constructed in the forward pass.
#        (Pure numpy reimplementation)
#        """
#        n_terms = numpy.sum(base_clsf.best_set)
#        n_eff_params = n_terms + self.penalty * (n_terms - 1) / 2.0
#        data, y, _ = examples.to_numpy_MA()
#        data = data.filled(0.0)
#        best_set = numpy.asarray(base_clsf.best_set, dtype=bool)
#        
#        bx = base_matrix(data, base_clsf.best_set,
#                         base_clsf.dirs, base_clsf.cuts,
#                         )
#        
#        bx_used = bx[:, best_set]
#        subsets, rss_per_subset = subsets_selection_xtx(bx_used, y) # TODO: Use leaps like library
#        gcv_per_subset = [gcv(rss, bx.shape[0], i + self.penalty * (i - 1) / 2.0) \
#                              for i, rss in enumerate(rss_per_subset, 1)]
#        gcv_per_subset = numpy.array(gcv_per_subset)
#        
#        best_i = numpy.argmin(gcv_per_subset[1:]) + 1 # Ignore the intercept
#        best_ind = subsets[best_i, :best_i + 1]
#        bs_i = 0
#        for i, b in enumerate(best_set):
#            if b:
#                best_set[i] = bs_i in best_ind
#                bs_i += 1
#                
#        bx_subset = bx[:, best_set]
#        betas, rss, rank, s = numpy.linalg.lstsq(bx_subset, y)
#        return best_set, betas, rss, subsets, rss_per_subset, gcv_per_subset
#    
#        
#class _EarthClassifier(Orange.core.ClassifierFD):
#    def __init__(self, base_classifier=None, examples=None, best_set=None,
#                 dirs=None, cuts=None, betas=None, subsets=None,
#                 rss_per_subset=None,
#                 gcv_per_subset=None):
#        self._base_classifier = base_classifier
#        self.examples = examples
#        self.domain = base_classifier.domain
#        self.class_var = base_classifier.class_var
#        
#        best_set = base_classifier.best_set if best_set is None else best_set
#        dirs = base_classifier.dirs if dirs is None else dirs
#        cuts = base_classifier.cuts if cuts is None else cuts
#        betas = base_classifier.betas if betas is None else betas
#        
#        self.best_set = numpy.asarray(best_set, dtype=bool)
#        self.dirs = numpy.array(dirs, dtype=int)
#        self.cuts = numpy.array(cuts)
#        self.betas = numpy.array(betas)
#        
#        self.subsets = subsets
#        self.rss_per_subset = rss_per_subset
#        self.gcv_per_subset = gcv_per_subset
#        
#    @property
#    def num_terms(self):
#        """ Number of terms in the model (including the intercept).
#        """
#        return numpy.sum(numpy.asarray(self.best_set, dtype=int))
#    
#    @property
#    def max_terms(self):
#        """ Maximum number of terms (as specified in the learning step).
#        """
#        return self.best_set.size
#    
#    @property
#    def num_preds(self):
#        """ Number of predictors (variables) included in the model.
#        """
#        return len(self.used_attributes(term))
#    
#    def __call__(self, example, what=Orange.core.GetValue):
#        value = self.predict(example)
#        if isinstance(self.class_var, Orange.data.variable.Continuous):
#            value = self.class_var(value)
#        else:
#            value = self.class_var(int(round(value)))
#            
#        dist = Orange.statistics.distribution.Distribution(self.class_var)
#        dist[value] = 1.0
#        if what == Orange.core.GetValue:
#            return value
#        elif what == Orange.core.GetProbabilities:
#            return dist
#        else:
#            return (value, dist)
#    
#    def base_matrix(self, examples=None):
#        """ Return the base matrix (bx) of the Earth model for the table.
#        If table is not supplied the base matrix of the training examples 
#        is returned.
#        
#        
#        :param examples: Input examples for the base matrix.
#        :type examples: Orange.data.Table 
#        
#        """
#        if examples is None:
#            examples = self.examples
#            
#        if examples is None:
#            raise ValueError("base matrix is only available if 'store_examples=True'")
#        
#        if isinstance(examples, Orange.data.Table):
#            data, cls, w = examples.to_numpy_MA()
#            data = data.filled(0.0)
#        else:
#            data = numpy.asarray(examples)
#            
#        return base_matrix(data, self.best_set, self.dirs, self.cuts)
#    
#    def _anova_order(self):
#        """ Return indices that sort the terms into the 'ANOVA' format.
#        """
#        terms = [([], 0)] # intercept
#        for i, used in enumerate(self.best_set[1:], 1):
#            attrs = sorted(self.used_attributes(i))
#            if used and attrs:
#                terms.append((attrs, i))
#        terms = sotred(terms, key=lambda t:(len(t[0]), t[0]))
#        return [i for _, i in terms]
#    
#    def format_model(self, percision=3, indent=3):
#        return format_model(self, percision, indent)
#    
#    def print_model(self, percision=3, indent=3):
#        print self.format_model(percision, indent)
#        
#    def predict(self, ex):
#        """ Return the predicted value (float) for example.
#        """
#        x = Orange.data.Table(self.domain, [ex])
#        x, c, w = x.to_numpy_MA()
#        x = x.filled(0.0)[0]
#        
#        bx = numpy.ones(self.best_set.shape)
#        betas = numpy.zeros_like(self.betas)
#        betas[0] = self.betas[0]
#        beta_i = 0
#        for termi in range(1, len(self.best_set)):
#            dirs = self.dirs[termi]
#            cuts = self.cuts[termi]
#            dir_p1 = numpy.where(dirs == 1)[0]
#            dir_m1 = numpy.where(dirs == -1)[0]
#            dir_2 = numpy.where(dirs == 2)[0]
#            
#            x1 = x[dir_p1] - cuts[dir_p1]
#            x2 = cuts[dir_m1] - x[dir_m1]
#            x3 = x[dir_2]
#            
#            x1 = numpy.maximum(x1, 0.0)
#            x2 = numpy.maximum(x2, 0.0)
#
#            X = numpy.hstack((x1, x2, x3))
#            X = numpy.cumprod(X)
#            
#            bx[termi] = X[-1] if X.size else 0.0
#            if self.best_set[termi] != 0:
#                beta_i += 1
#                betas[beta_i] = self.betas[beta_i]
#
#        return numpy.sum(bx[self.best_set] * betas)
#            
#    def used_attributes(self, term=None):
#        """ Return a list of used attributes. If term (index) is given
#        return only attributes used in that single term.
#        
#        """
#        if term is None:
#            terms = numpy.where(self.best_set)[0]
#        else:
#            terms = [term]
#        attrs = set()
#        for ti in terms:
#            attri = numpy.where(self.dirs[ti] != 0.0)[0]
#            attrs.update([self.domain.attributes[i] for i in attri])
#        return attrs
#        
#    def evimp(self, used_only=True):
#        """ Return the estimated variable importance.
#        """
#        return evimp(self, used_only)
#        
#    def __reduce__(self):
#        return (EarthClassifier, (self._base_classifier, self.examples,
#                                  self.best_set, self.dirs, self.cuts,
#                                  self.betas, self.subsets,
#                                  self.rss_per_subset, self.gcv_per_subset),
#                {})
                                 
