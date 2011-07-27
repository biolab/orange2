"""\
EARTH (Multivariate adaptive regression splines - MARS) (``earth``)

 
.. autoclass :: EarthLearner

.. autoclass :: EarthClassifier

"""

import Orange
from Orange.core import (EarthLearner as BaseEarthLearner,
                         EarthClassifier as BaseEarthClassifier)
            
import numpy

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
    
def base_matrix(data, best_set, dirs, cuts, betas):
    """ Return the base matrix for the earth model.
    
    """
    data = numpy.asarray(data)
    best_set = numpy.asarray(best_set)
    dirs = numpy.asarray(dirs)
    cuts = numpy.asarray(cuts)
    betas = numpy.asarray(betas)
    
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


class EarthLearner(BaseEarthLearner):
    """ 
    
    """
    def __new__(cls, data=None, weightId=None, **kwargs):
        self = BaseEarthLearner.__new__(cls, **kwargs)
        if data is not None:
            self.__init__(**kwargs)
            return self.__call__(data, weightId)
        return self
    
    def __init__(self, store_examples=True, **kwargs):
        self.store_examples = store_examples
        for key, val in kwargs.items():
            setattr(self, key, val)
    
    def __call__(self, data, weightId=None):
        with member_set(self, "prune", False):
            # We overwrite the prune argument (will do the pruning in python).
            base_clsf =  BaseEarthLearner.__call__(self, data, weightId)
        
        if self.prune:
            (best_set, betas, rss, subsets, rss_per_subset,
             gcv_per_subset) = self.pruning_pass(base_clsf, data)
            
            return EarthClassifier(base_clsf, data if self.store_examples else None,
                                   best_set=best_set, dirs=base_clsf.dirs,
                                   cuts=base_clsf.cuts,
                                   betas=betas,
                                   subsets=subsets,
                                   rss_per_subset=rss_per_subset,
                                   gcv_per_subset=gcv_per_subset)
        else:
            return EarthClassifier(base_clsf, data if self.store_examples else None)
    
    
    def pruning_pass(self, base_clsf, examples):
        """ Prune the terms constructed in the forward pass.
        """
        n_terms = numpy.sum(base_clsf.best_set)
        n_eff_params = n_terms + self.penalty * (n_terms - 1) / 2.0
        data, y, _ = examples.to_numpy_MA()
        data = data.filled(0.0)
        best_set = numpy.asarray(base_clsf.best_set, dtype=bool)
        
        bx = base_matrix(data, base_clsf.best_set,
                         base_clsf.dirs, base_clsf.cuts,
                         base_clsf.betas)
        
        bx_used = bx[:, best_set]
        subsets, rss_per_subset = subsets_selection_xtx(bx_used, y) # TODO: Use leaps like library
        gcv_per_subset = [gcv(rss, bx.shape[0], i + self.penalty * (i - 1) / 2.0) \
                              for i, rss in enumerate(rss_per_subset, 1)]
        gcv_per_subset = numpy.array(gcv_per_subset)
        
        best_i = numpy.argmin(gcv_per_subset[1:]) + 1 # Ignore the intercept
        best_ind = subsets[best_i, :best_i + 1]
        bs_i = 0
        for i, b in enumerate(best_set):
            if b:
                best_set[i] = bs_i in best_ind
                bs_i += 1
                
        bx_subset = bx[:, best_set]
        betas, rss, rank, s = numpy.linalg.lstsq(bx_subset, y)
        return best_set, betas, rss, subsets, rss_per_subset, gcv_per_subset
        
        
class EarthClassifier(Orange.core.ClassifierFD):
    def __init__(self, base_classifier=None, examples=None, best_set=None,
                 dirs=None, cuts=None, betas=None, subsets=None,
                 rss_per_subset=None,
                 gcv_per_subset=None):
        self._base_classifier = base_classifier
        self.examples = examples
        self.domain = base_classifier.domain
        self.class_var = base_classifier.class_var
        
        best_set = base_classifier.best_set if best_set is None else best_set
        dirs = base_classifier.dirs if dirs is None else dirs
        cuts = base_classifier.cuts if cuts is None else cuts
        betas = base_classifier.betas if betas is None else betas
        
        self.best_set = numpy.asarray(best_set, dtype=bool)
        self.dirs = numpy.array(dirs, dtype=int)
        self.cuts = numpy.array(cuts)
        self.betas = numpy.array(betas)
        
        self.subsets = subsets
        self.rss_per_subset = rss_per_subset
        self.gcv_per_subset = gcv_per_subset
        
    @property
    def num_terms(self):
        """ Number of terms in the model (including the intercept).
        """
        return numpy.sum(numpy.asarray(self.best_set, dtype=int))
    
    @property
    def max_terms(self):
        """ Maximum number of terms (as specified in the learning step).
        """
        return self.best_set.size
    
    @property
    def num_preds(self):
        """ Number of predictors (variables) included in the model.
        """
        return len(self.used_attributes(term))
    
    def __call__(self, example, what=Orange.core.GetValue):
        value = self.predict(example)
        if isinstance(self.class_var, Orange.data.variable.Continuous):
            value = self.class_var(value)
        else:
            value = self.class_var(int(round(value)))
            
        dist = Orange.statistics.distribution.Distribution(self.class_var)
        dist[value] = 1.0
        if what == Orange.core.GetValue:
            return value
        elif what == Orange.core.GetProbabilities:
            return dist
        else:
            return (value, dist)
    
     
    def terms(self):
        """ Return the terms in the Earth model.
        """
        raise NotImplementedError
    
    def filters(self):
        """ Orange.core.filter objects for each term (where the hinge
        function is not 0).
         
        """
        
    
    def base_matrix(self, examples=None):
        """ Return the base matrix (bx) of the Earth model for the table.
        If table is not supplied the base matrix of the training examples 
        is returned.
        
        
        :param examples: Input examples for the base matrix.
        :type examples: Orange.data.Table 
        
        """
        if examples is None:
            examples = self.examples
            
        if examples is None:
            raise ValueError("base matrix is only available if 'store_examples=True'")
        
        if isinstance(examples, Orange.data.Table):
            data, cls, w = examples.to_numpy_MA()
            data = data.filled(0.0)
        else:
            data = numpy.asarray(examples)
            
        return base_matrix(data, self.best_set, self.dirs, self.cuts, self.betas)
    
    def _anova_order(self):
        """ Return indices that sort the terms into the 'ANOVA' format.
        """
        terms = [([], 0)] # intercept
        for i, used in enumerate(self.best_set[1:], 1):
            attrs = sorted(self.used_attributes(i))
            if used and attrs:
                terms.append((attrs, i))
        terms = sotred(terms, key=lambda t:(len(t[0]), t[0]))
        return [i for _, i in terms]
    
    def format_model(self, percision=3, indent=3):
        header = "%s =" % self.class_var.name
        indent = " " * indent
        fmt = "%." + str(percision) + "f"
        terms = [([], fmt % self.betas[0])]
        beta_i = 0
        for i, used in enumerate(self.best_set[1:], 1):
            if used:
                beta_i += 1
                beta = fmt % abs(self.betas[beta_i])
                knots = [self._format_knot(attr.name, d, c) for d, c, attr in \
                         zip(self.dirs[i], self.cuts[i], self.domain.attributes) \
                         if d != 0]
                term_attrs = [a for a, d in zip(self.domain.attributes, self.dirs[i]) \
                              if d != 0]
                term_attrs = sorted(term_attrs)
                sign = "-" if self.betas[beta_i] < 0 else "+"
                if knots:
                    terms.append((term_attrs,
                                  sign + " * ".join([beta] + knots)))
                else:
                    terms.append((term_attr, sign + beta))
        # Sort by len(term_attrs), then by term_attrs
        terms = sorted(terms, key=lambda t: (len(t[0]), t[0]))
        return "\n".join([header] + [indent + t for _, t in terms])
            
    def _format_knot(self, name, dir, cut):
        if dir == 1:
            txt = "max(0, %s - %.3f)" % (name, cut)
        elif dir == -1:
            txt = "max(0, %.3f - %s)" % (cut, name)
        elif dir == 2:
            txt = name
        return txt
    
    def _format_term(self, i):
        knots = [self._format_knot(attr.name, d, c) for d, c, attr in \
                 zip(self.dirs[i], self.cuts[i], self.domain.attributes) \
                 if d != 0]
        return " * ".join(knots)
    
    def print_model(self, percision=3, indent=3):
        print self.format_model(percision, indent)
        
    def predict(self, ex):
        """ Return the predicted value (float) for example.
        """
        x = Orange.data.Table(self.domain, [ex])
        x, c, w = x.to_numpy_MA()
        x = x.filled(0.0)[0]
        
        bx = numpy.ones(self.best_set.shape)
        betas = numpy.zeros_like(self.betas)
        betas[0] = self.betas[0]
        beta_i = 0
        for termi in range(1, len(self.best_set)):
            dirs = self.dirs[termi]
            cuts = self.cuts[termi]
            dir_p1 = numpy.where(dirs == 1)[0]
            dir_m1 = numpy.where(dirs == -1)[0]
            dir_2 = numpy.where(dirs == 2)[0]
            
            x1 = x[dir_p1] - cuts[dir_p1]
            x2 = cuts[dir_m1] - x[dir_m1]
            x3 = x[dir_2]
            
            x1 = numpy.maximum(x1, 0.0)
            x2 = numpy.maximum(x2, 0.0)

            X = numpy.hstack((x1, x2, x3))
            X = numpy.cumprod(X)
            
            bx[termi] = X[-1] if X.size else 0.0
            if self.best_set[termi] != 0:
                beta_i += 1
                betas[beta_i] = self.betas[beta_i]

        return numpy.sum(bx[self.best_set] * betas)
            
    def used_attributes(self, term=None):
        """ Return a list of used attributes. If term (index) is given
        return only attributes used in that single term.
        
        """
        if term is None:
            terms = numpy.where(self.best_set)[0]
        else:
            terms = [term]
        attrs = set()
        for ti in terms:
            attri = numpy.where(self.dirs[ti] != 0.0)[0]
            attrs.update([self.domain.attributes[i] for i in attri])
        return attrs
        
    def evimp(self, used_only=True):
        """ Return the estimated variable importance.
        """
        if self.subsets is None:
            raise ValueError("No subsets. Use the learner with 'prune=True'.")
        
        subsets = self.subsets
        n_subsets = self.num_terms
        
        rss = -numpy.diff(self.rss_per_subset)
        gcv = -numpy.diff(self.gcv_per_subset)
        attributes = list(self.domain.attributes)
        
        attr2ind = dict(zip(attributes, range(len(attributes))))
        importances = numpy.zeros((len(attributes), 4))
        importances[:, 0] = range(len(attributes))
        
        for i in range(1, n_subsets):
            term_subset = self.subsets[i, :i + 1]
            used_attributes = reduce(set.union, [self.used_attributes(term) for term in term_subset], set())
            for attr in used_attributes:
                importances[attr2ind[attr]][1] += 1.0
                importances[attr2ind[attr]][2] += gcv[i - 1]
                importances[attr2ind[attr]][3] += rss[i - 1]
        imp_min = numpy.min(importances[:, [2, 3]], axis=0)
        imp_max = numpy.max(importances[:, [2, 3]], axis=0)
        importances[:, [2, 3]] = 100.0 * (importances[:, [2, 3]] - [imp_min]) / ([imp_max - imp_min])
        
        importances = list(importances)
        # Sort by n_subsets and gcv.
        importances = sorted(importances, key=lambda row: (row[1], row[2]),
                             reverse=True)
        importances = numpy.array(importances)
        
        if used_only:
            importances = importances[importances[:,0] > 0.0]
        
        res = [(attributes[int(row[0])], tuple(row[1:])) for row in importances]
        return res
            
    def plot(self):
        import pylab
        n_terms = self.num_terms
        grid_size = int(numpy.ceil(numpy.sqrt(n_terms)))
        fig = pylab.figure()
        
    def __reduce__(self):
        return (EarthClassifier, (self._base_classifier, self.examples,
                                  self.best_set, self.dirs, self.cuts,
                                  self.betas, self.subsets,
                                  self.rss_per_subset, self.gcv_per_subset),
                {})
                                 
    
def gcv(rss, n, n_effective_params):
    """ Return the generalized cross validation.
    
    .. math: gcv = rss / (n * (1 - n_effective_params / n) ^ 2)
    
    :param rss: Residual sum of squares.
    :param n: Number of training examples.
    :param n_effective_params: Number of effective paramaters.
     
    """
    return  rss / (n * (1 - n_effective_params / n) ** 2)
    

def subsets_selection_xtx(X, Y):
    """
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


def plot_evimp(evimp):
    """ Plot the return value from EarthClassifier.evimp.
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
    
    