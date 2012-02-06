"""\
============================
Lasso regression (``lasso``)
============================

.. index:: regression

.. _`Lasso regression. Regression shrinkage and selection via the lasso`:
    http://www-stat.stanford.edu/~tibs/lasso/lasso.pdf


`The Lasso <http://www-stat.stanford.edu/~tibs/lasso/lasso.pdf>`_ is a shrinkage
and selection method for linear regression. It minimizes the usual sum of squared
errors, with a bound on the sum of the absolute values of the coefficients. 

To fit the regression parameters on housing data set use the following code:

.. literalinclude:: code/lasso-example.py
   :lines: 7,9,10,11

.. autoclass:: LassoRegressionLearner
    :members:

.. autoclass:: LassoRegression
    :members:


.. autoclass:: LassoRegressionLearner
    :members:

.. autoclass:: LassoRegression
    :members:

Utility functions
-----------------

.. autofunction:: center

.. autofunction:: get_bootstrap_sample

.. autofunction:: permute_responses


========
Examples
========

To predict values of the response for the first five instances
use the code

.. literalinclude:: code/lasso-example.py
   :lines: 14,15

Output

::

    Actual: 24.00, predicted: 24.58 
    Actual: 21.60, predicted: 23.30 
    Actual: 34.70, predicted: 24.98 
    Actual: 33.40, predicted: 24.78 
    Actual: 36.20, predicted: 24.66 

To see the fitted regression coefficients, print the model

.. literalinclude:: code/lasso-example.py
   :lines: 17

The output

::

    Variable  Coeff Est  Std Error          p
     Intercept     22.533
          CRIM     -0.000      0.023      0.480      
         INDUS     -0.010      0.023      0.300      
            RM      1.303      0.994      0.000   ***
           AGE     -0.002      0.000      0.320      
       PTRATIO     -0.191      0.209      0.050     .
         LSTAT     -0.126      0.105      0.000   ***
    Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1


    For 7 variables the regression coefficient equals 0: 
    ZN
    CHAS
    NOX
    DIS
    RAD
    TAX
    B

shows that some of the regression coefficients are equal to 0.    





"""

import Orange
import numpy

from Orange.regression import base

from Orange.misc import deprecated_members, deprecated_keywords

def center(X):
    """Centers the data, i.e. subtracts the column means.
    Returns the centered data and the mean.

    :param X: the data arry
    :type table: :class:`numpy.array`
    """
    mu = X.mean(axis=0)
    return X - mu, mu

def standardize(X):
    """Standardizes the data, i.e. subtracts the column means and divide by 
    standard deviation.
    Returns the centered data, the mean, and deviations.

    :param X: the data arry
    :type table: :class:`numpy.array`
    """
    mu = numpy.mean(X, axis=0)
    std = numpy.std(X, axis=0)
    return (X - mu) / std, mu, std

def get_bootstrap_sample(table):
    """Generates boostrap sample from an Orange Example Table
    and stores it in a new :class:`Orange.data.Table` object

    :param table: the original data sample
    :type table: :class:`Orange.data.Table`
    """
    n = len(table)
    bootTable = Orange.data.Table(table.domain)
    for i in range(n):
        id = numpy.random.randint(0,n)
        bootTable.append(table[id])
    return bootTable

def permute_responses(table):
    """ Permutes values of the class (response) variable.
    The independence between independent variables and the response
    is obtained but the distribution of the response variable is kept.

    :param table: the original data sample
    :type table: :class:`Orange.data.Table`
    """
    n = len(table)
    perm = numpy.random.permutation(n)
    permTable = Orange.data.Table(table.domain, table)
    for i, ins in enumerate(table):
        permTable[i].set_class(table[perm[i]].get_class())
    return permTable

class LassoRegressionLearner(base.BaseRegressionLearner):
    """Fits the lasso regression model, i.e. learns the regression parameters
    The class is derived from
    :class:`Orange.regression.base.BaseRegressionLearner`
    which is used for preprocessing the data (continuization and imputation)
    before fitting the regression parameters

    """

    def __init__(self, name='lasso regression', t=1, s=None, tol=0.001, \
                 n_boot=100, n_perm=100, imputer=None, continuizer=None):
        """
        :param name: name of the linear model, default 'lasso regression'
        :type name: string
        
        :param t: tuning parameter, upper bound for the L1-norm of the
            regression coefficients
        :type t: float
        
        :param s: An alternative way to specify the tuning parameter ``t``.
            Here ``t`` is taken to be t = s * sum(abs(B)) where B are the
            coefficients of an ordinary least square linear fit. ``t`` parameter is ignored if ``s`` is specified (by default it
            is None).
        :type s: float
        
        :param tol: tolerance parameter, regression coefficients
            (absoulute value) under tol are set to 0,
            default=0.001
        :type tol: float
        
        :param n_boot: number of bootstrap samples used for non-parametric
            estimation of standard errors
        :type n_boot: int
        
        :param n_perm: number of permuations used for non-parametric
            estimation of p-values
        :type n_perm: int
        
        """

        self.name = name
        self.t = t
        self.s = s
        self.tol = tol
        self.n_boot = n_boot
        self.n_perm = n_perm
        self.set_imputer(imputer=imputer)
        self.set_continuizer(continuizer=continuizer)
        
        
    def __call__(self, table, weight=None):
        """
        :param table: data instances.
        :type table: :class:`Orange.data.Table`
        :param weight: the weights for instances. Default: None, i.e.
            all data instances are eqaully important in fitting
            the regression parameters
        :type weight: None or list of Orange.data.variable.Continuous
            which stores weights for instances
        
        """  
        # dicrete values are continuized        
        table = self.continuize_table(table)
        # missing values are imputed
        table = self.impute_table(table)

        domain = table.domain
        X, y, w = table.to_numpy()
        n, m = numpy.shape(X)
        
        X, mu_x, sigma_x = standardize(X)
        y, coef0 = center(y)
        
        t = self.t
        
        if self.s is not None:
            beta_full, rss, _, _ = numpy.linalg.lstsq(X, y)
            t = self.s * numpy.sum(numpy.abs(beta_full))
            print "t =", t
            
        import scipy.optimize
            
        # objective function to be minimized
        objective = lambda beta: numpy.linalg.norm(y - numpy.dot(X, beta))
        # initial guess for the regression parameters
        beta_init = numpy.random.random(m)
        # constraints for the regression coefficients
        cnstr = lambda beta: t - numpy.sum(numpy.abs(beta))
        # optimal solution
        coefficients = scipy.optimize.fmin_cobyla(objective, beta_init,\
                                                       cnstr, disp=0)

        # set small coefficients to 0
        def set_2_0(c): return c if abs(c) > self.tol else 0
        coefficients = numpy.array(map(set_2_0, coefficients))
        coefficients /= sigma_x
        
        # bootstrap estimator of standard error of the coefficient estimators
        # assumption: fixed t
        if self.n_boot > 0:
            coeff_b = [] # bootstrapped coefficients
            for i in range(self.n_boot):
                tmp_table = get_bootstrap_sample(table)
                l = LassoRegressionLearner(t=t, n_boot=0, n_perm=0)
                c = l(tmp_table)
                coeff_b.append(c.coefficients)
            std_errors_fixed_t = numpy.std(coeff_b, axis=0)
        else:
            std_errors_fixed_t = [float("nan")] * m

        # permutation test to obtain the significance of the regression
        #coefficients
        if self.n_perm > 0:
            coeff_p = []
            for i in range(self.n_perm):
                tmp_table = permute_responses(table)
                l = LassoRegressionLearner(t=t, n_boot=0, n_perm=0)
                c = l(tmp_table)
                coeff_p.append(c.coefficients)
            p_vals = \
                   numpy.sum(abs(numpy.array(coeff_p))>\
                             abs(numpy.array(coefficients)), \
                             axis=0)/float(self.n_perm)
        else:
            p_vals = [float("nan")] * m

        # dictionary of regression coefficients with standard errors
        # and p-values
        dict_model = {}
        for i, var in enumerate(domain.attributes):
            dict_model[var.name] = (coefficients[i], std_errors_fixed_t[i], p_vals[i])            
       
        return LassoRegression(domain=domain, class_var=domain.class_var,
                               coef0=coef0, coefficients=coefficients,
                               std_errors_fixed_t=std_errors_fixed_t,
                               p_vals=p_vals,
                               dict_model= dict_model,
                               mu_x=mu_x)

deprecated_members({"nBoot": "n_boot",
                    "nPerm": "n_perm"}, 
                   wrap_methods=["__init__"],
                   in_place=True)(LassoRegressionLearner)

class LassoRegression(Orange.classification.Classifier):
    """Lasso regression predicts value of the response variable
    based on the values of independent variables.

    .. attribute:: coef0

        Intercept (sample mean of the response variable).    

    .. attribute:: coefficients

        Regression coefficients, sotred in list. 

    .. attribute:: std_errors_fixed_t

        Standard errors of the coefficient estimator for the fixed
        tuning parameter t. The standard errors are estimated using
        bootstrapping method.

    .. attribute:: p_vals

        List of p-values for the null hypothesis that the regression
        coefficients equal 0 based on non-parametric permutation test.

    .. attribute:: dict_model

        Statistical properties of the model stored in dictionary:
        Keys - names of the independent variables
        Values - tuples (coefficient, standard error, p-value) 

    .. attribute:: mu_x

        Sample mean of the all independent variables.    

    """ 
    def __init__(self, domain=None, class_var=None, coef0=None,
                 coefficients=None, std_errors_fixed_t=None, p_vals=None,
                 dict_model=None, mu_x=None):
        self.domain = domain
        self.class_var = class_var
        self.coef0 = coef0
        self.coefficients = coefficients
        self.std_errors_fixed_t = std_errors_fixed_t
        self.p_vals = p_vals
        self.dict_model = dict_model
        self.mu_x = mu_x

    @deprecated_keywords({"resultType": "result_type"})
    def __call__(self, instance, result_type=Orange.core.GetValue):
        """
        :param instance: data instance for which the value of the response
            variable will be predicted
        :type instance: 
        """  
        ins = Orange.data.Instance(self.domain, instance)
        if "?" in ins: # missing value -> corresponding coefficient omitted
            def miss_2_0(x): return x if x != "?" else 0
            ins = map(miss_2_0, ins)
            ins = numpy.array(ins)[:-1] - self.mu_x
        else:
            ins = numpy.array(ins.native())[:-1] - self.mu_x

        y_hat = numpy.dot(self.coefficients, ins) + self.coef0 
        y_hat = self.class_var(y_hat)
        dist = Orange.statistics.distribution.Continuous(self.class_var)
        dist[y_hat] = 1.0
        if result_type == Orange.core.GetValue:
            return y_hat
        if result_type == Orange.core.GetProbabilities:
            return dist
        else:
            return (y_hat, dist)
        
    @deprecated_keywords({"skipZero": "skip_zero"})
    def to_string(self, skip_zero=True):
        """Pretty-prints Lasso regression model,
        i.e. estimated regression coefficients with standard errors
        and significances. Standard errors are obtained using bootstrapping
        method and significances by the permuation test

        :param skip_zero: if True variables with estimated coefficient equal to 0
            are omitted
        :type skip_zero: boolean
        """
        
        from string import join
        labels = ('Variable', 'Coeff Est', 'Std Error', 'p')
        lines = [join(['%10s' % l for l in labels], ' ')]

        fmt = "%10s " + join(["%10.3f"]*3, " ") + " %5s"
        fmt1 = "%10s %10.3f"

        def get_star(p):
            if p < 0.001: return  "*"*3
            elif p < 0.01: return "*"*2
            elif p < 0.05: return "*"
            elif p < 0.1: return  "."
            else: return " "

        stars =  get_star(self.p_vals[0])
        lines.append(fmt1 % ('Intercept', self.coef0))
        skipped = []
        for i in range(len(self.domain.attributes)):
            if self.coefficients[i] == 0. and skip_zero:
                skipped.append(self.domain.attributes[i].name)
                continue            
            stars = get_star(self.p_vals[i])
            lines.append(fmt % (self.domain.attributes[i].name, 
                         self.coefficients[i], self.std_errors_fixed_t[i], 
                         self.p_vals[i], stars))
        lines.append("Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1")
        lines.append("\n")
        if skip_zero:
            k = len(skipped)
            if k == 0:
                lines.append("All variables have non-zero regression coefficients. ")
            else:
                suff = "s" if k > 1 else ""
                lines.append("For %d variable%s the regression coefficient equals 0: " \
                      % (k, suff))
                for var in skipped:
                    lines.append(var)
        return "\n".join(lines)

    def __str__(self):
        return self.to_string(skip_zero=True)

deprecated_members({"muX": "mu_x",
                    "stdErrorsFixedT": "std_errors_fixed_t",
                    "pVals": "p_vals",
                    "dictModel": "dict_model"},
                   wrap_methods=["__init__"],
                   in_place=True)(LassoRegression)

if __name__ == "__main__":

    import Orange
    
    table = Orange.data.Table("housing.tab")        

    c = LassoRegressionLearner(table, t=len(table.domain))
    print c
