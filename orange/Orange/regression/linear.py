"""\
==============================
Linear regression (``linear``)
==============================

.. index:: regression, linear model

.. _`Linear regression`: http://en.wikipedia.org/wiki/Linear_regression

Example ::

    >>> from Orange.regression import linear
    >>> table = Orange.data.Table("housing")
    >>> c = linear.LinearRegressionLearner(table)
    >>> print c
    
      Variable  Coeff Est  Std Error    t-value          p
     Intercept     36.459      5.103      7.144      0.000   ***
          CRIM     -0.108      0.033     -3.287      0.001    **
            ZN      0.046      0.014      3.382      0.001   ***
         INDUS      0.021      0.061      0.334      0.738      
          CHAS      2.687      0.862      3.118      0.002    **
           NOX    -17.767      3.820     -4.651      0.000   ***
            RM      3.810      0.418      9.116      0.000   ***
           AGE      0.001      0.013      0.052      0.958      
           DIS     -1.476      0.199     -7.398      0.000   ***
           RAD      0.306      0.066      4.613      0.000   ***
           TAX     -0.012      0.004     -3.280      0.001    **
       PTRATIO     -0.953      0.131     -7.283      0.000   ***
             B      0.009      0.003      3.467      0.001   ***
         LSTAT     -0.525      0.051    -10.347      0.000   ***
    Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1
       
    >>> 


.. autoclass:: LinearRegressionLearner
    :members:

.. autoclass:: LinearRegression
    :members:

Utility functions
-----------------

.. autofunction:: print_linear_regression_model

.. autofunction:: stepwise


"""


import Orange
from Orange.regression import base
import numpy

try:
    from scipy import stats
except ImportError:
    import statc as stats

from numpy import dot, sqrt
from numpy.linalg import inv, pinv


from Orange.misc import deprecated_members, deprecated_keywords

class LinearRegressionLearner(base.BaseRegressionLearner):

    """Fits the linear regression model, i.e. learns the regression parameters
    The class is derived from
    :class:`Orange.regression.base.BaseRegressionLearner`
    which is used for preprocessing the data (continuization and imputation)
    before fitting the regression parameters.

    """    

    def __init__(self, name='linear regression', intercept=True, \
                 compute_stats=True, ridge_lambda=None,\
                 imputer=None, continuizer=None, \
                 use_vars=None, stepwise=False, add_sig=0.05,
                 remove_sig=0.2, **kwds):
        """
        :param name: name of the linear model, default 'linear regression'
        :type name: string
        :param intercept: if True, the intercept beta0 is included
            in the model
        :type intercept: boolean
        :param compute_stats: if True, statistical properties of
            the estimators (standard error, t-scores, significances)
            and statistical properties of the model
            (sum of squares, R2, adjusted R2) are computed
        :type compute_stats: boolean
        :param ridge_lambda: if not None, the lambda parameter
            in ridge regression
        :type ridge_lambda: integer or None
        :param use_vars: the list of independent varaiables included in
            regression model. If None (default) all variables are used
        :type use_vars: list of Orange.data.variable or None
        :param stepwise: if True, _`stepwise regression`:
            http://en.wikipedia.org/wiki/Stepwise_regression
            based on F-test is performed. The significance parameters are
            add_sig and remove_sig
        :type stepwise: boolean
        :param add_sig: lower bound of significance for which the variable
            is included in regression model
            default value = 0.05
        :type add_sig: float
        :param remove_sig: upper bound of significance for which
            the variable is excluded from the regression model
            default value = 0.2
        :type remove_sig: float
        """
        self.name = name
        self.intercept = intercept
        self.compute_stats = compute_stats
        self.ridge_lambda = ridge_lambda
        self.set_imputer(imputer=imputer)
        self.set_continuizer(continuizer=continuizer)
        self.stepwise = stepwise
        self.add_sig = add_sig
        self.remove_sig = remove_sig
        self.use_vars = use_vars
        self.__dict__.update(kwds)
        
    def __call__(self, table, weight=None, verbose=0):
        """
        :param table: data instances.
        :type table: :class:`Orange.data.Table`
        :param weight: the weights for instances. Default: None, i.e.
            all data instances are eqaully important in fitting
            the regression parameters
        :type weight: None or list of Orange.data.variable.Continuous
            which stores weights for instances
        """       
        if not self.use_vars is None:
            new_domain = Orange.data.Domain(self.use_vars,
                                            table.domain.class_var)
            new_domain.addmetas(table.domain.getmetas())
            table = Orange.data.Table(new_domain, table)

        # dicrete values are continuized        
        table = self.continuize_table(table)
          
        # missing values are imputed
        table = self.impute_table(table)

        if self.stepwise:
            use_vars = stepwise(table, weight, add_sig=self.add_sig,
                                      remove_sig=self.remove_sig)
            new_domain = Orange.data.Domain(use_vars, table.domain.class_var)
            new_domain.addmetas(table.domain.getmetas())
            table = Orange.data.Table(new_domain, table)

        # convertion to numpy
        A, y, w = table.to_numpy()
        if A is None:
            n, m = len(table), 0
        else:
            n, m = numpy.shape(A)
     
        if self.intercept:
             if A is None:
                 X = numpy.ones([n,1])
             else:
                 X = numpy.insert(A, 0, 1, axis=1) # adds a column of ones
        else:
             X = A
             
        domain = table.domain
        
        if numpy.std(y) < 10e-6: # almost constant variable
            return Orange.regression.mean.MeanLearner(table)
     
        # set weights to the instances
        W = numpy.identity(n)
        if weight:
            for i, ins in enumerate(table):
                W[i, i] = float(ins[weight])

        compute_stats = self.compute_stats
        # adds some robustness by computing the pseudo inverse;
        # normal inverse could fail due to singularity of the X.T * W * X
        if self.ridge_lambda is None:
            cov = pinv(dot(dot(X.T, W), X))
        else:
            cov = pinv(dot(dot(X.T, W), X) - self.ridge_lambda*numpy.eye(m+1))
            compute_stats = False # TO DO: find inferential properties of the estimators
        D = dot(dot(cov, X.T), W)
        coefficients = dot(D, y)

        mu_y, sigma_y = numpy.mean(y), numpy.std(y)
        if A is not None:
            cov_x = numpy.cov(X, rowvar=0)

            # standardized coefficients
            std_coefficients = (sqrt(cov_x.diagonal()) / sigma_y) \
                                * coefficients
        else:
            std_coefficients = None

        if compute_stats is False:
            return LinearRegression(domain.class_var, domain, coefficients=coefficients,
                                    std_coefficients=std_coefficients, intercept=self.intercept)
            

        fitted = dot(X, coefficients)
        residuals = [ins.get_class() - fitted[i] \
                     for i, ins in enumerate(table)]

        # model summary        
        # total sum of squares (total variance)
        sst = numpy.sum((y - mu_y) ** 2)
        # sum of squares due to regression (explained variance)
        ssr = numpy.sum((fitted - mu_y)**2)
        # error sum of squares (unexplaied variance)
        sse = sst - ssr
        # coefficient of determination
        r2 = ssr / sst
        r2adj = 1-(1-r2)*(n-1)/(n-m-1)
        F = (ssr/m)/(sst-ssr/(n-m-1))
        df = n-2 
        sigma_square = sse/(n-m-1)
        # standard error of the regression estimator, t-scores and p-values
        std_error = sqrt(sigma_square*pinv(dot(X.T, X)).diagonal())
        t_scores = coefficients/std_error
        p_vals = [stats.betai(df*0.5,0.5,df/(df + t*t)) \
                  for t in t_scores]

        # dictionary of regression coefficients with standard errors
        # and p-values
        dict_model = {}
        if self.intercept:
            dict_model["Intercept"] = (coefficients[0],\
                                      std_error[0], \
                                      t_scores[0], \
                                      p_vals[0])
        for i, var in enumerate(domain.attributes):
            j = i + 1 if self.intercept else i
            dict_model[var.name] = (coefficients[j], \
                                   std_error[j],\
                                   t_scores[j],\
                                   p_vals[j])
        
        return LinearRegression(domain.class_var, domain, coefficients, F,
                 std_error=std_error, t_scores=t_scores, p_vals=p_vals, dict_model=dict_model,
                 fitted=fitted, residuals=residuals, m=m, n=n, mu_y=mu_y,
                 r2=r2, r2adj=r2adj, sst=sst, sse=sse, ssr=ssr,
                 std_coefficients=std_coefficients, intercept=self.intercept)

deprecated_members({"ridgeLambda": "ridge_lambda",
                    "computeStats": "compute_stats",
                    "useVars": "use_vars",
                    "addSig": "add_sig",
                    "removeSig": "remove_sig",
                    }
                   , ["__init__"],
                   in_place=True)(LinearRegressionLearner)

class LinearRegression(Orange.classification.Classifier):

    """Linear regression predicts value of the response variable
    based on the values of independent variables.

    .. attribute:: F
    
        F-statistics of the model.

    .. attribute:: coefficients

        list of regression coefficients. If the intercept is included
        the first item corresponds to the estimated intercept

    .. attribute:: std_error

        list of standard errors of the coefficient estimator.    

    .. attribute:: t_scores

        list of t-scores for the estimated regression coefficients    

    .. attribute:: p_vals

        list of p-values for the null hypothesis that the regression
        coefficients equal 0 based on t-scores and two sided
        alternative hypothesis    

    .. attribute:: dict_model

        dictionary of statistical properties of the model.
        Keys - names of the independent variables (or "Intercept")
        Values - tuples (coefficient, standard error,
        t-value, p-value)

    .. attribute:: fitted

        estimated values of the dependent variable for all instances
        from the training table

    .. attribute:: residuals

        differences between estimated and actual values of the
        dependent variable for all instances from the training table

    .. attribute:: m

        number of independent variables    

    .. attribute:: n

        number of instances    

    .. attribute:: mu_y

        the sample mean of the dependent variable    

    .. attribute:: r2

        _`coefficient of determination`:
        http://en.wikipedia.org/wiki/Coefficient_of_determination

    .. attribute:: r2adj

        adjusted coefficient of determination

    .. attribute:: sst, sse, ssr

        total sum of squares, explained sum of squares and
        residual sum of squares respectively

    .. attribute:: std_coefficients

        standardized regression coefficients

    """   


    
    def __init__(self, class_var=None, domain=None, coefficients=None, F=None,
                 std_error=None, t_scores=None, p_vals=None, dict_model=None,
                 fitted=None, residuals=None, m = None, n=None, mu_y=None,
                 r2=None, r2adj=None, sst=None, sse=None, ssr=None,
                 std_coefficients=None, intercept=None):
        """
        :param model: fitted linear regression model
        :type model: :class:`LinearRegressionLearner`
        """
        self.class_var = class_var
        self.domain = domain
        self.coefficients = coefficients
        self.F = F
        self.std_error = std_error
        self.t_scores = t_scores
        self.p_vals = p_vals
        self.dict_model = dict_model
        self.fitted = fitted
        self.residuals = residuals
        self.m = m
        self.n = n
        self.mu_y = mu_y
        self.r2 = r2
        self.r2adj = r2adj
        self.sst = sst
        self.sse = sse
        self.ssr = ssr
        self.std_coefficients = std_coefficients
        self.intercept = intercept

    def __call__(self, instance, \
                 resultType=Orange.classification.Classifier.GetValue):
        """
        :param instance: data instance for which the value of the response
            variable will be predicted
        :type instance: 
        """        
        ins = Orange.data.Instance(self.domain, instance)
        ins = numpy.array(ins.native())
        if "?" in ins: # missing value -> corresponding coefficient omitted
            def miss_2_0(x): return x if x != "?" else 0
            ins = map(miss_2_0, ins)

        if self.intercept:
            if len(self.coefficients) > 1:
                y_hat = self.coefficients[0] + \
                       dot(self.coefficients[1:], ins[:-1])
            else:
                if len(ins) == 1:
                    print ins
                    y_hat = self.mu_y
                else:
                    y_hat = dot(self.coefficients, ins[:-1])
        else:
            y_hat = dot(self.coefficients, ins[:-1])
#        y_hat = Orange.data.Value(y_hat)
        y_hat = self.class_var(y_hat)
        dist = Orange.statistics.distribution.Continuous(self.class_var)
        dist[y_hat] = 1.0
        if resultType == Orange.classification.Classifier.GetValue:
            return y_hat
        if resultType == Orange.classification.Classifier.GetProbabilities:
            return dist
        return (y_hat, dist)


    def to_string(self):
        """Pretty-prints linear regression model,
        i.e. estimated regression coefficients with standard errors, t-scores
        and significances.

        """
        from string import join 
        labels = ('Variable', 'Coeff Est', 'Std Error', 't-value', 'p')
        lines = [join(['%10s' % l for l in labels], ' ')]

        fmt = "%10s " + join(["%10.3f"]*4, " ") + " %5s"
        if not self.p_vals:
            raise ValueError("Model does not contain model statistics.")
        def get_star(p):
            if p < 0.001: return  "*"*3
            elif p < 0.01: return "*"*2
            elif p < 0.05: return "*"
            elif p < 0.1: return  "."
            else: return " "
        
        if self.intercept == True:
            stars =  get_star(self.p_vals[0])
            lines.append(fmt % ('Intercept', self.coefficients[0], 
                         self.std_error[0], self.t_scores[0], self.p_vals[0], stars))
            for i in range(len(self.domain.attributes)):
                stars = get_star(self.p_vals[i+1])
                lines.append(fmt % (self.domain.attributes[i].name,
                             self.coefficients[i+1], self.std_error[i+1],
                             self.t_scores[i+1], self.p_vals[i+1], stars))
        else:
            for i in range(len(self.domain.attributes)):
                stars = get_star(self.p_vals[i])
                lines.append(fmt % (self.domain.attributes[i].name,
                             self.coefficients[i], self.std_error[i],
                             self.t_scores[i], self.p_vals[i], stars))
        lines.append("Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1")
        return "\n".join(lines)

    def __str__(self):
        return self.to_string()
        



def compare_models(c1, c2):
    """ Compares if classifiaction model c1 is significantly better
    than model c2. The comparison is based on F-test, the p-value
    is returned.

    :param c1, c2: linear regression model objects.
    :type lr: :class:`LinearRegression`     

    """
    if c1 == None or c2 == None:
        return 1.0
    p1, p2, n = c1.m, c2.m, c1.n
    RSS1, RSS2 = c1.sse, c2.sse
    if RSS1 <= RSS2 or p2 <= p1 or n <= p2 or RSS2 <= 0:
        return 1.0
    F = ((RSS1-RSS2)/(p2-p1))/(RSS2/(n-p2))
    return stats.fprob(int(p2-p1), int(n-p2), F)


@deprecated_keywords({"addSig": "add_sig", "removeSig": "remove_sig"})
def stepwise(table, weight, add_sig=0.05, remove_sig=0.2):
    """ Performs _`stepwise linear regression`:
    http://en.wikipedia.org/wiki/Stepwise_regression
    on table and returns the list of remaing independent variables
    which fit a significant linear regression model.coefficients

    :param table: data instances.
    :type table: :class:`Orange.data.Table`
    :param weight: the weights for instances. Default: None, i.e. all data
        instances are eqaully important in fitting the regression parameters
    :type weight: None or list of Orange.data.variable.Continuous
        which stores the weights
    :param add_sig: lower bound of significance for which the variable
        is included in regression model
        default value = 0.05
    :type add_sig: float
    :param remove_sig: upper bound of significance for which the variable
        is excluded from the regression model
        default value = 0.2
    :type remove_sig: float
    """

    
    inc_vars = []
    not_inc_vars = table.domain.attributes

    changed_model = True
    while changed_model:
        changed_model = False
        # remove all unsignificant conditions (learn several models,
        # where each time one variable is removed and check significance)
        c0 = LinearRegressionLearner(table, use_vars=inc_vars)
        reduced_model = [] # reduced model
        for ati in range(len(inc_vars)):
            try:
                reduced_model.append(LinearRegressionLearner(table, weight,
                        use_vars=inc_vars[:ati] + inc_vars[(ati + 1):]))
            except Exception:
                reduced_model.append(None)
        
        sigs = [compare_models(r, c0) for r in reduced_model]
        if sigs and max(sigs) > remove_sig:
            # remove that variable, start again
            crit_var = inc_vars[sigs.index(max(sigs))]
            not_inc_vars.append(crit_var)
            inc_vars.remove(crit_var)
            changed_model = True
            continue

        # add all significant conditions (go through all attributes in
        # not_inc_vars, is there one that significantly improves the model?
        extended_model = []
        for ati in range(len(not_inc_vars)):
            try:
                extended_model.append(LinearRegressionLearner(table,
                        weight, use_vars=inc_vars + [not_inc_vars[ati]]))
            except Exception:
                extended_model.append(None)
             
        sigs = [compare_models(c0, r) for r in extended_model]
        if sigs and min(sigs) < add_sig:
            best_var = not_inc_vars[sigs.index(min(sigs))]
            inc_vars.append(best_var)
            not_inc_vars.remove(best_var)
            changed_model = True
    return inc_vars


if __name__ == "__main__":

    import Orange
    from Orange.regression import linear

    table = Orange.data.Table("housing.tab")
    c = LinearRegressionLearner(table)
    print c
