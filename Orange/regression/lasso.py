from numpy import dot, std, array, zeros, maximum, sqrt, sign, log, abs, \
                  ascontiguousarray, random as rnd
from scipy.linalg import norm, eigh

import Orange
from Orange.utils import deprecated_members, deprecated_keywords


def get_bootstrap_sample(data):
    """Generate a bootstrap sample of a given data set.

    :param data: the original data sample
    :type data: :class:`Orange.data.Table`
    """
    n = len(data)
    bootstrap = Orange.data.Table(data.domain)
    for id in rnd.randint(0, n, n):
        bootstrap.append(data[id])
    return bootstrap

def permute_responses(data):
    """Permute values of the class (response) variable.
    The independence between independent variables and the response
    is obtained but the distribution of the response variable is kept.

    :param data: Original data.
    :type data: :class:`Orange.data.Table`
    """
    n = len(data)
    perm = rnd.permutation(n)
    perm_data = Orange.data.Table(data.domain, data)
    for i, ins in enumerate(data):
        perm_data[i].set_class(data[perm[i]].get_class())
    return perm_data

class LassoRegressionLearner(Orange.regression.base.BaseRegressionLearner):
    """Fits the lasso regression model using FISTA
    (Fast Iterative Shrinkage-Thresholding Algorithm).
    """

    def __init__(self, lasso_lambda=0.1, max_iter=20000, eps=1e-6,
                 n_boot=0, n_perm=0, imputer=None, continuizer=None,
                 name='Lasso'):
        """
        :param lasso_lambda: Regularization parameter.
        :type lasso_lambda: float

        :param max_iter: Maximum number of iterations for
                         the optimization method.
        :type max_iter: int

        :param eps: Stop optimization when improvements are lower than eps.
        :type eps: float

        :param n_boot: Number of bootstrap samples used for non-parametric
                       estimation of standard errors.
        :type n_boot: int

        :param n_perm: Number of permuations used for non-parametric
                       estimation of p-values.
        :type n_perm: int

        :param name: Learner name.
        :type name: str
        
        """
        self.lasso_lambda = lasso_lambda
        self.max_iter = max_iter
        self.eps = eps
        self.n_boot = n_boot
        self.n_perm = n_perm
        self.set_imputer(imputer=imputer)
        self.set_continuizer(continuizer=continuizer)
        self.name = name

    def get_lipschitz(self, X):
        """Return the Lipschitz constant of :math:`\\nabla f`,
        where :math:`f(w) = \\frac{1}{2}||Xw-y||^2`.
        """
        n, m = X.shape
        if n > m:
            X = ascontiguousarray(X.T)
        k = min(n, m) - 1
        eigvals = eigh(dot(X, X.T), eigvals_only=True, eigvals=(k, k))
        return eigvals[-1]

    def fista(self, X, y, l, lipschitz, w_init=None):
        """Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)."""
        z = w_old = zeros(X.shape[1]) if w_init is None else w_init
        t_old, obj_old = 1, 1e400
        XT = ascontiguousarray(X.T)
        for i in range(self.max_iter):
            z -= 1. / lipschitz * dot(XT, dot(X, z) - y)
            w = maximum(0, abs(z) - l / lipschitz) * sign(z)
            t = (1 + sqrt(1 + 4 * t_old**2)) / 2
            z = w + (t_old - 1) / t * (w - w_old)
            obj = ((y - dot(X, w))**2).sum() + l * norm(w, 1) 
            if abs(obj_old - obj) / obj < self.eps:
                stop += 1
                if obj < obj_old and stop > log(i + 1):
                    break
            else:
                stop = 0
            w_old, t_old = w, t
            obj_old = obj
        return w

    def __call__(self, data, weight=None):
        """
        :param data: Training data.
        :type data: :class:`Orange.data.Table`
        :param weight: Weights for instances. Not implemented yet.
        
        """
        # dicrete values are continuized        
        data = self.continuize_table(data)
        # missing values are imputed
        data = self.impute_table(data)
        domain = data.domain
        # prepare numpy matrices
        X, y, _ = data.to_numpy()
        n, m = X.shape
        coefficients = zeros(m)
        std_errors = array([float('nan')] * m)
        p_vals = array([float('nan')] * m)
        # standardize y
        coef0, sigma_y = y.mean(), y.std() + 1e-6
        y = (y - coef0) / sigma_y
        # standardize X and remove constant vars
        mu_x = X.mean(axis=0)
        X -= mu_x
        sigma_x = X.std(axis=0)
        nz = sigma_x != 0
        X = ascontiguousarray(X[:, nz])
        sigma_x = sigma_x[nz]
        X /= sigma_x
        m = sum(nz)

        # run optimization method
        lipschitz = self.get_lipschitz(X)
        l = 0.5 * self.lasso_lambda * n / m
        coefficients[nz] = self.fista(X, y, l, lipschitz)
        coefficients[nz] *= sigma_y / sigma_x

        d = dict(self.__dict__)
        d.update({'n_boot': 0, 'n_perm': 0})

        # bootstrap estimator of standard error of the coefficient estimators
        # assumption: fixed regularization parameter
        if self.n_boot > 0:
            coeff_b = [] # bootstrapped coefficients
            for i in range(self.n_boot):
                tmp_data = get_bootstrap_sample(data)
                l = LassoRegressionLearner(**d)
                c = l(tmp_data)
                coeff_b.append(c.coefficients)
            std_errors[nz] = std(coeff_b, axis=0)

        # permutation test to obtain the significance of
        # the regression coefficients
        if self.n_perm > 0:
            coeff_p = []
            for i in range(self.n_perm):
                tmp_data = permute_responses(data)
                l = LassoRegressionLearner(**d)
                c = l(tmp_data)
                coeff_p.append(c.coefficients)
            p_vals[nz] = (abs(coeff_p) > abs(coefficients)).sum(axis=0)
            p_vals[nz] /= float(self.n_perm)

        # dictionary of regression coefficients with standard errors
        # and p-values
        model = {}
        for i, var in enumerate(domain.attributes):
            model[var.name] = (coefficients[i], std_errors[i], p_vals[i])

        return LassoRegression(domain=domain, class_var=domain.class_var,
            coef0=coef0, coefficients=coefficients, std_errors=std_errors,
            p_vals=p_vals, model=model, mu_x=mu_x)

deprecated_members({"nBoot": "n_boot",
                    "nPerm": "n_perm"},
                   wrap_methods=["__init__"],
                   in_place=True)(LassoRegressionLearner)

class LassoRegression(Orange.classification.Classifier):
    """Lasso regression predicts the value of the response variable
    based on the values of independent variables.

    .. attribute:: coef0

        Intercept (sample mean of the response variable).    

    .. attribute:: coefficients

        Regression coefficients. 

    .. attribute:: std_errors

        Standard errors of coefficient estimates for a fixed
        regularization parameter. The standard errors are estimated
        using the bootstrapping method.

    .. attribute:: p_vals

        List of p-values for the null hypotheses that the regression
        coefficients equal 0 based on a non-parametric permutation test.

    .. attribute:: model

        Dictionary with the statistical properties of the model:
        Keys - names of the independent variables
        Values - tuples (coefficient, standard error, p-value) 

    .. attribute:: mu_x

        Sample mean of independent variables.    

    """
    def __init__(self, domain=None, class_var=None, coef0=None,
                 coefficients=None, std_errors=None, p_vals=None,
                 model=None, mu_x=None):
        self.domain = domain
        self.class_var = class_var
        self.coef0 = coef0
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.p_vals = p_vals
        self.model = model
        self.mu_x = mu_x

    def _miss_2_0(self, x):
        return x if x != '?' else 0

    @deprecated_keywords({"resultType": "result_type"})
    def __call__(self, instance, result_type=Orange.core.GetValue):
        """
        :param instance: Data instance for which the value of the response
                         variable will be predicted.
        :type instance: :obj:`Orange.data.Instance`
        """
        ins = Orange.data.Instance(self.domain, instance)
        if '?' in ins: # missing value -> corresponding coefficient omitted
            ins = map(self._miss_2_0, ins)
            ins = array(ins)[:-1] - self.mu_x
        else:
            ins = array(ins.native())[:-1] - self.mu_x

        y_hat = dot(self.coefficients, ins) + self.coef0
        y_hat = self.class_var(y_hat)
        dist = Orange.statistics.distribution.Continuous(self.class_var)
        dist[y_hat] = 1.
        if result_type == Orange.core.GetValue:
            return y_hat
        if result_type == Orange.core.GetProbabilities:
            return dist
        else:
            return (y_hat, dist)

    @deprecated_keywords({"skipZero": "skip_zero"})
    def to_string(self, skip_zero=True):
        """Pretty-prints a lasso regression model,
        i.e. estimated regression coefficients with standard errors
        and significances. Standard errors are obtained using the
        bootstrapping method and significances by a permuation test.

        :param skip_zero: If True, variables with estimated coefficient
                          equal to 0 are omitted.
        :type skip_zero: bool
        """
        labels = ('Variable', 'Coeff Est', 'Std Error', 'p')
        lines = [' '.join(['%10s' % l for l in labels])]

        fmt = '%10s ' + ' '.join(['%10.3f'] * 3) + ' %5s'
        fmt1 = '%10s %10.3f'

        def get_star(p):
            if p < 0.001: return  '*' * 3
            elif p < 0.01: return '*' * 2
            elif p < 0.05: return '*'
            elif p < 0.1: return  '.'
            else: return ' '

        stars = get_star(self.p_vals[0])
        lines.append(fmt1 % ('Intercept', self.coef0))
        skipped = []
        for i in range(len(self.domain.attributes)):
            if self.coefficients[i] == 0. and skip_zero:
                skipped.append(self.domain.attributes[i].name)
                continue
            stars = get_star(self.p_vals[i])
            lines.append(fmt % (self.domain.attributes[i].name,
                         self.coefficients[i], self.std_errors[i],
                         self.p_vals[i], stars))
        lines.append('Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1\n')
        if skip_zero:
            k = len(skipped)
            if k == 0:
                lines.append('All variables have non-zero regression coefficients.')
            else:
                lines.append('For %d variable%s the regression coefficient equals 0:'
                             % (k, 's' if k > 1 else ''))
                lines.append(', '.join(var for var in skipped))
        return '\n'.join(lines)

    def __str__(self):
        return self.to_string(skip_zero=True)

deprecated_members({"muX": "mu_x",
                    "stdErrorsFixedT": "std_errors",
                    "pVals": "p_vals",
                    "dictModel": "model"},
                   wrap_methods=["__init__"],
                   in_place=True)(LassoRegression)

if __name__ == "__main__":
    data = Orange.data.Table('housing')
    c = LassoRegressionLearner(data)
    print c
