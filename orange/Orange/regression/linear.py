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
    >>> linear.print_linear_regression_model(c)
    
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
import scipy.stats

from numpy import dot, sqrt
from numpy.linalg import inv, pinv


class LinearRegressionLearner(base.BaseRegressionLearner):

    """Fits the linear regression model, i.e. learns the regression parameters
    The class is derived from
    :class:`Orange.regression.base.BaseRegressionLearner`
    which is used for preprocessing the data (continuization and imputation)
    before fitting the regression parameters.

    """    

    def __init__(self, name='linear regression', intercept=True, \
                 computeStats=True, ridgeLambda=None,\
                 imputer=None, continuizer=None, \
                 useVars=None, stepwise=False, addSig=0.05,
                 removeSig=0.2, **kwds):
        """
        :param name: name of the linear model, default 'linear regression'
        :type name: string
        :param intercept: if True, the intercept beta0 is included
            in the model
        :type intercept: boolean
        :param computeStats: if True, statistical properties of
            the estimators (standard error, t-scores, significances)
            and statistical properties of the model
            (sum of squares, R2, adjusted R2) are computed
        :type computeStats: boolean
        :param ridgeLambda: if not None, the lambda parameter
            in ridge regression
        :type ridgeLambda: integer or None
        :param useVars: the list of independent varaiables included in
            regression model. If None (default) all variables are used
        :type useVars: list of Orange.data.variable or None
        :param stepwise: if True, _`stepwise regression`:
            http://en.wikipedia.org/wiki/Stepwise_regression
            based on F-test is performed. The significance parameters are
            addSig and removeSig
        :type stepwise: boolean
        :param addSig: lower bound of significance for which the variable
            is included in regression model
            default value = 0.05
        :type addSig: float
        :param removeSig: upper bound of significance for which
            the variable is excluded from the regression model
            default value = 0.2
        :type removeSig: float
        """
        self.name = name
        self.intercept = intercept
        self.computeStats = computeStats
        self.ridgeLambda = ridgeLambda
        self.set_imputer(imputer=imputer)
        self.set_continuizer(continuizer=continuizer)
        self.stepwise = stepwise
        self.addSig = addSig
        self.removeSig = removeSig
        self.useVars = useVars
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
        if not self.useVars is None:
            newDomain = Orange.data.Domain(self.useVars,
                                            table.domain.class_var)
            newDomain.addmetas(table.domain.getmetas())
            table = Orange.data.Table(newDomain, table)

        # dicrete values are continuized        
        table = self.continuize_table(table)
          
        # missing values are imputed
        table = self.impute_table(table)

        if self.stepwise:
            useVars = stepwise(table, weight, addSig=self.addSig,
                                      removeSig=self.removeSig)
            newDomain = Orange.data.Domain(useVars, table.domain.class_var)
            newDomain.addmetas(table.domain.getmetas())
            table = Orange.data.Table(newDomain, table)

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

        computeStats = self.computeStats
        # adds some robustness by computing the pseudo inverse;
        # normal inverse could fail due to singularity of the X.T * W * X
        if self.ridgeLambda is None:
            cov = pinv(dot(dot(X.T, W), X))        
        else:
            cov = pinv(dot(dot(X.T, W), X) - self.ridgeLambda*numpy.eye(m+1))
            computeStats = False # TO DO: find inferential properties of the estimators
        D = dot(dot(cov, X.T), W)
        coefficients = dot(D, y)

        muY, sigmaY = numpy.mean(y), numpy.std(y)
        if A is not None:
            covX = numpy.cov(X, rowvar=0)

            # standardized coefficients
            stdCoefficients = (sqrt(covX.diagonal()) / sigmaY) \
                               * coefficients

        if computeStats is False:
            return LinearRegression(domain.class_var, domain, coefficients=coefficients,
                                    std_coefficients=stdCoefficients, intercept=self.intercept)
            

        fitted = dot(X, coefficients)
        residuals = [ins.get_class()-fitted[i] \
                     for i, ins in enumerate(table)]

        # model summary        
        # total sum of squares (total variance)
        sst = numpy.sum((y - muY) ** 2)
        # sum of squares due to regression (explained variance)
        ssr = numpy.sum((fitted - muY)**2)
        # error sum of squares (unexplaied variance)
        sse = sst - ssr
        # coefficient of determination
        r2 = ssr / sst
        r2adj = 1-(1-r2)*(n-1)/(n-m-1)
        F = (ssr/m)/(sst-ssr/(n-m-1))
        df = n-2 
        sigmaSquare = sse/(n-m-1)
        # standard error of the regression estimator, t-scores and p-values
        stdError = sqrt(sigmaSquare*pinv(dot(X.T, X)).diagonal())
        tScores = coefficients/stdError
        pVals = [scipy.stats.betai(df*0.5,0.5,df/(df + t*t)) \
                 for t in tScores]

        # dictionary of regression coefficients with standard errors
        # and p-values
        dictModel = {}
        if self.intercept:
            dictModel["Intercept"] = (coefficients[0],\
                                      stdError[0], \
                                      tScores[0], \
                                      pVals[0])
        for i, var in enumerate(domain.attributes):
            j = i + 1 if self.intercept else i
            dictModel[var.name] = (coefficients[j], \
                                   stdError[j],\
                                   tScores[j],\
                                   pVals[j])
        
        return LinearRegression(domain.class_var, domain, coefficients, F,
                 std_error=stdError, t_scores=tScores, p_vals=pVals, dict_model=dictModel,
                 fitted=fitted, residuals=residuals, m=m, n=n, mu_y=muY,
                 r2=r2, r2adj=r2adj, sst=sst, sse=sse, ssr=ssr,
                 std_coefficients=stdCoefficients, intercept=self.intercept)


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
                yHat = self.coefficients[0] + \
                       dot(self.coefficients[1:], ins[:-1])
            else:
                if len(ins) == 1:
                    print ins
                    yHat = self.mu_y
                else:
                    yHat = dot(self.coefficients, ins[:-1])
        else:
            yHat = dot(self.coefficients, ins[:-1])
#        yHat = Orange.data.Value(yHat)
        yHat = self.class_var(yHat)
        dist = Orange.statistics.distribution.Continuous(self.class_var)
        dist[yHat] = 1.0
        if resultType == Orange.classification.Classifier.GetValue:
            return yHat
        if resultType == Orange.classification.Classifier.GetProbabilities:
            return dist
        return (yHat, dist)


def print_linear_regression_model(lr):
    """Pretty-prints linear regression model,
    i.e. estimated regression coefficients with standard errors, t-scores
    and significances.

    :param lr: a linear regression model object.
    :type lr: :class:`LinearRegression`    

    """
    from string import join
    m = lr   
    labels = ('Variable', 'Coeff Est', 'Std Error', 't-value', 'p')
    print join(['%10s' % l for l in labels], ' ')

    fmt = "%10s " + join(["%10.3f"]*4, " ") + " %5s"
    if not lr.p_vals:
        raise ValueError("Model does not contain model statistics.")
    def get_star(p):
        if p < 0.001: return  "*"*3
        elif p < 0.01: return "*"*2
        elif p < 0.05: return "*"
        elif p < 0.1: return  "."
        else: return " "
    
    if m.intercept == True:
        stars =  get_star(m.p_vals[0])
        print fmt % ('Intercept', m.coefficients[0], \
                     m.std_error[0], m.t_scores[0], m.p_vals[0], stars)
        for i in range(len(m.domain.attributes)):
            stars = get_star(m.p_vals[i+1])
            print fmt % (m.domain.attributes[i].name,\
                         m.coefficients[i+1], m.std_error[i+1],\
                         m.t_scores[i+1], m.p_vals[i+1], stars)
    else:
        for i in range(len(m.domain.attributes)):
            stars = get_star(m.p_vals[i])
            print fmt % (m.domain.attributes[i].name,\
                         m.coefficients[i], m.std_error[i],\
                         m.t_scores[i], m.p_vals[i], stars)
    print "Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1"



def compare_models(c1, c2):
    """ Compares if classifiaction model c1 is significantly better
    than model c2. The comparison is based on F-test, the p-value
    is returned.

    :param c1, c2: linear regression model objects.
    :type lr: :class:`LinearRegression`     

    """
    if c1 == None or c2 == None:
        return 1.0
    p1, p2, n = c1.model.m, c2.model.m, c1.model.n
    RSS1, RSS2 = c1.model.sse, c2.model.sse
    if RSS1 <= RSS2 or p2 <= p1 or n <= p2 or RSS2 <= 0:
        return 1.0
    F = ((RSS1-RSS2)/(p2-p1))/(RSS2/(n-p2))
    return scipy.stats.fprob(int(p2-p1), int(n-p2), F)


def stepwise(table, weight, addSig=0.05, removeSig=0.2):
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
    :param addSig: lower bound of significance for which the variable
        is included in regression model
        default value = 0.05
    :type addSig: float
    :param removeSig: upper bound of significance for which the variable
        is excluded from the regression model
        default value = 0.2
    :type removeSig: float
    """

    
    incVars = []
    notIncVars = table.domain.attributes

    changedModel = True
    while changedModel:
        changedModel = False
        # remove all unsignificant conditions (learn several models,
        # where each time one variable is removed and check significance)
        c0 = LinearRegressionLearner(table, useVars=incVars)
        reducedModel = [] # reduced model
        for ati in range(len(incVars)):
            try:
                reducedModel.append(LinearRegressionLearner(table, weight,
                        useVars=incVars[:ati] + incVars[(ati + 1):]))
            except:
                reducedModel.append(None)
        
        sigs = [compare_models(r, c0) for r in reducedModel]
        if sigs and max(sigs) > removeSig:
            # remove that variable, start again
            critVar = incVars[sigs.index(max(sigs))]
            notIncVars.append(critVar)
            incVars.remove(critVar)
            changedModel = True
            continue

        # add all significant conditions (go through all attributes in
        # notIncVars, is there one that significantly improves the model?
        extendedModel = []
        for ati in range(len(notIncVars)):
            try:
                extendedModel.append(LinearRegressionLearner(table,
                        weight, useVars=incVars + [notIncVars[ati]]))
            except:
                extendedModel.append(None)
             
        sigs = [compare_models(c0, r) for r in extendedModel]
        if sigs and min(sigs) < addSig:
            bestVar = notIncVars[sigs.index(min(sigs))]
            incVars.append(bestVar)
            notIncVars.remove(bestVar)
            changedModel = True
    return incVars


if __name__ == "__main__":

    import Orange
    from Orange.regression import linear

    table = Orange.data.Table("housing.tab")
    c = LinearRegressionLearner(table)
    print_linear_regression_model(c)
