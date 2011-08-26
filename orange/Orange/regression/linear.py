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
    before fitting the regression parameters

    .. attribute:: F
    
        F-statistics of the model.

    .. attribute:: coefficients

        list of regression coefficients. If the intercept is included
        the first item corresponds to the estimated intercept

    .. attribute:: stdError

        list of standard errors of the coefficient estimator.    

    .. attribute:: tScores

        list of t-scores for the estimated regression coefficients    

    .. attribute:: pVals

        list of p-values for the null hypothesis that the regression
        coefficients equal 0 based on t-scores and two sided
        alternative hypothesis    

    .. attribute:: dictModel

        dictionary of statistical properties of the model.
        Keys - names of the independent variables (or "Intercept")
        Values - tuples (coefficient, standard error,
        t-value, p-value)

    .. attribute:: fitted

        estimated values of the dependent variable for all instances
        from the table

    .. attribute:: residuals

        differences between estimated and actual values of the
        dependent variable for all instances from the table

    .. attribute:: m

        number of independent variables    

    .. attribute:: n

        number of instances    

    .. attribute:: muY

        the sample mean of the dependent variable    

    .. attribute:: r2

        _`coefficient of determination`:
        http://en.wikipedia.org/wiki/Coefficient_of_determination

    .. attribute:: r2adj

        adjusted coefficient of determination

    .. attribute:: sst, sse, ssr

        total sum of squares, explained sum of squares and
        residual sum of squares respectively

    .. attribute:: stdCoefficients

        standardized regression coefficients


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

        self.domain, self.m, self.n = table.domain, m, n

        if numpy.std(y) < 10e-6: # almost constant variable
            return Orange.regression.mean.MeanLearner(table)
     
        # set weights to the instances
        W = numpy.identity(n)
        if weight:
            for i, ins in enumerate(table):
                W[i, i] = float(ins[weight])

        # adds some robustness by computing the pseudo inverse;
        # normal inverse could fail due to singularity of the X.T * W * X
        if self.ridgeLambda is None:
            cov = pinv(dot(dot(X.T, W), X))        
        else:
            cov = pinv(dot(dot(X.T, W), X) - self.ridgeLambda*numpy.eye(m+1))
            self.computeStats = False # TO DO: find inferential properties of the estimators
        D = dot(dot(cov, X.T), W)
        self.coefficients = dot(D, y)

        self.muY, sigmaY = numpy.mean(y), numpy.std(y)
        if A is not None:
            covX = numpy.cov(X, rowvar=0)

            # standardized coefficients
            self.stdCoefficients = (sqrt(covX.diagonal()) / sigmaY) \
                                   * self.coefficients

        if self.computeStats is False:
            return LinearRegression(self)

        self.fitted = dot(X, self.coefficients)
        self.residuals = [ins.get_class()-self.fitted[i] \
                          for i, ins in enumerate(table)]

        # model summary        
        # total sum of squares (total variance)
        self.sst = numpy.sum((y - self.muY) ** 2)
        # sum of squares due to regression (explained variance)
        self.ssr = numpy.sum((self.fitted - self.muY)**2)
        # eror sum of squares (unexplaied variance)
        self.sse = self.sst - self.ssr
        # coefficient of determination
        self.r2 = self.ssr / self.sst
        self.r2adj = 1-(1-self.r2)*(n-1)/(n-m-1)
        self.F = (self.ssr/m)/(self.sst-self.ssr/(n-m-1))
        df = n-2 
        sigmaSquare = self.sse/(n-m-1)
        # standard error of the regression estimator, t-scores and p-values
        self.stdError = sqrt(sigmaSquare*pinv(dot(X.T, X)).diagonal())
        self.tScores = self.coefficients/self.stdError
        self.pVals=[scipy.stats.betai(df*0.5,0.5,df/(df + t*t)) \
                    for t in self.tScores]

        # dictionary of regression coefficients with standard errors
        # and p-values
        self.dictModel = {}
        if self.intercept:
            self.dictModel["Intercept"] = (self.coefficients[0],\
                                           self.stdError[0], \
                                           self.tScores[0], \
                                           self.pVals[0])
        for i, var in enumerate(self.domain.attributes):
            j = i+1 if self.intercept else i
            self.dictModel[var.name] = (self.coefficients[j], \
                                        self.stdError[j],\
                                        self.tScores[j],\
                                        self.pVals[j])
        
        return LinearRegression(self)


class LinearRegression(Orange.classification.Classifier):

    """Linear regression predicts value of the response variable
    based on the values of independent variables.

    .. attribute:: model
    
        fitted linear regression model   

    """   


    
    def __init__(self, model):
        """
        :param model: fitted linear regression model
        :type model: :class:`LinearRegressionLearner`
        """
        self.model = model

    def __call__(self, instance, \
                 resultType=Orange.classification.Classifier.GetValue):
        """
        :param instance: data instance for which the value of the response
            variable will be predicted
        :type instance: 
        """        
        ins = Orange.data.Instance(self.model.domain, instance)
        ins = numpy.array(ins.native())
        if "?" in ins: # missing value -> corresponding coefficient omitted
            def miss_2_0(x): return x if x != "?" else 0
            ins = map(miss_2_0, ins)

        if self.model.intercept:
            if len(self.model.coefficients) > 1:
                yHat = self.model.coefficients[0] + \
                       dot(self.model.coefficients[1:], ins[:-1])
            else:
                if len(ins) == 1:
                    print ins
                    yHat = self.model.muY
                else:
                    yHat = dot(self.model.coefficients, ins[:-1])
        else:
            yHat = dot(self.model.coefficients, ins[:-1])
        yHat = Orange.data.Value(yHat)
         
        if resultType == Orange.classification.Classifier.GetValue:
            return yHat
        if resultType == Orange.classification.Classifier.GetProbabilities:
            return Orange.statistics.distribution.Continuous({1.0: yHat})
        return (yHat, Orange.statistics.distribution.Continuous({1.0: yHat}))


def print_linear_regression_model(lr):
    """Pretty-prints linear regression model,
    i.e. estimated regression coefficients with standard errors, t-scores
    and significances.

    :param lr: a linear regression model object.
    :type lr: :class:`LinearRegression`    

    """
    from string import join
    m = lr.model    
    labels = ('Variable', 'Coeff Est', 'Std Error', 't-value', 'p')
    print join(['%10s' % l for l in labels], ' ')

    fmt = "%10s " + join(["%10.3f"]*4, " ") + " %5s"

    def get_star(p):
        if p < 0.001: return  "*"*3
        elif p < 0.01: return "*"*2
        elif p < 0.05: return "*"
        elif p < 0.1: return  "."
        else: return " "
    
    if m.intercept == True:
        stars =  get_star(m.pVals[0])
        print fmt % ('Intercept', m.coefficients[0], \
                     m.stdError[0], m.tScores[0], m.pVals[0], stars)
        for i in range(len(m.domain.attributes)):
            stars = get_star(m.pVals[i+1])
            print fmt % (m.domain.attributes[i].name,\
                         m.coefficients[i+1], m.stdError[i+1],\
                         m.tScores[i+1], m.pVals[i+1], stars)
    else:
        for i in range(len(m.domain.attributes)):
            stars = get_star(m.pVals[i])
            print fmt % (m.domain.attributes[i].name,\
                         m.coefficients[i], m.stdError[i],\
                         m.tScores[i], m.pVals[i], stars)
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
