"""\
============================
Lasso regression (``lasso``)
============================

.. index:: regression

.. _`Lasso regression. Regression shrinkage and selection via the lasso`:
    http://www-stat.stanford.edu/~tibs/lasso/lasso.pdf


Example ::

    >>> from Orange.regression import lasso
    >>> table = Orange.data.Table("housing")
    >>> c = lasso.LassoRegressionLearner(table)
    >>> linear.print_lasso_regression_model(c)
    
      Variable  Coeff Est  Std Error          p
     Intercept     22.533
          CRIM     -0.049      0.282      0.770      
            ZN      0.106      0.055      0.030     *
         INDUS     -0.111      0.442      0.920      
          CHAS      1.757      0.669      0.180      
           NOX      0.318      0.483      0.680      
            RM      1.643      0.461      0.480      
           AGE      0.062      0.051      0.230      
           DIS      0.627      0.538      0.930      
           RAD      1.260      0.472      0.070     .
           TAX     -0.074      0.027      0.120      
       PTRATIO      1.331      0.464      0.050     .
             B      0.017      0.007      0.080     .
         LSTAT     -0.209      0.323      0.650      
    Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1


    All variables have non-zero regression coefficients. 
       
    >>> 


.. autoclass:: LassoRegressionLearner
    :members:

.. autoclass:: LassoRegression
    :members:

Utility functions
-----------------

.. autofunction:: center

.. autofunction:: get_bootstrap_sample

.. autofunction:: permute_responses

.. autofunction:: print_lasso_regression_model


"""

import Orange
import numpy

from Orange.regression import base

def center(X):
    """Centers the data, i.e. subtracts the column means.
    Returns the centered data and the mean.

    :param X: the data arry
    :type table: :class:`numpy.array`
    """
    mu = X.mean(axis=0)
    return X - mu, mu

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

    .. attribute:: coeff0

        intercept (sample mean of the response variable)    

    .. attribute:: coefficients

        list of regression coefficients. 

    .. attribute:: stdErrorsFixedT

        list of standard errors of the coefficient estimator for the fixed
        tuning parameter t. The standard errors are estimated using
        bootstrapping method.

    .. attribute:: pVals

        list of p-values for the null hypothesis that the regression
        coefficients equal 0 based on non-parametric permutation test

    .. attribute:: dictModel

        dictionary of statistical properties of the model.
        Keys - names of the independent variables
        Values - tuples (coefficient, standard error, p-value) 

    .. attribute:: muX

        the sample mean of the all independent variables    


    """
    

    def __init__(self, name='lasso regression', t=1, tol=0.001, \
                 imputer=None, continuizer=None):
        """
        :param name: name of the linear model, default 'lasso regression'
        :type name: string
        :param t: tuning parameter, upper bound for the L1-norm of the
            regression coefficients
        :type t: float
        :param tol: tolerance parameter, regression coefficients
            (absoulute value) under tol are set to 0,
            default=0.001
        :type tol: float
        """

        self.name = name
        self.t = t
        self.tol = tol
        self.set_imputer(imputer=imputer)
        self.set_continuizer(continuizer=continuizer)
        
        
    def __call__(self, table, weight=None, nBoot=100, nPerm=100):
        """
        :param table: data instances.
        :type table: :class:`Orange.data.Table`
        :param weight: the weights for instances. Default: None, i.e.
            all data instances are eqaully important in fitting
            the regression parameters
        :type weight: None or list of Orange.data.variable.Continuous
            which stores weights for instances
        :param nBoot: number of bootstrap samples used for non-parametric
            estimation of standard errors
        :type nBoot: int
        :param nPerm: number of permuations used for non-parametric
            estimation of p-values
        :type nPerm: int
        
        """  
        # dicrete values are continuized        
        table = self.continuize_table(table)
        # missing values are imputed
        table = self.impute_table(table)

        self.domain = table.domain
        X, y, w = table.to_numpy()
        n, m = numpy.shape(X)
        X, self.muX = center(X)
        y, self.coef0 = center(y)

        import scipy.optimize

        # objective function to be minimized
        objective = lambda beta: numpy.linalg.norm(y-numpy.dot(X,beta))
        # initial guess for the regression parameters
        betaInit = numpy.random.random(m)
        # constraints for the regression coefficients
        cnstr = lambda beta: self.t - sum(numpy.abs(beta))
        # optimal solution
        self.coefficients = scipy.optimize.fmin_cobyla(objective, betaInit,\
                                                       cnstr)

        # set small coefficients to 0
        def set_2_0(c): return c if abs(c) > self.tol else 0
        self.coefficients = map(set_2_0, self.coefficients)

        # bootstrap estimator of standard error of the coefficient estimators
        # assumption: fixed t
        if nBoot > 0:
            coeffB = [] # bootstrapped coefficients
            for i in range(nBoot):
                tmpTable = get_bootstrap_sample(table)
                l = LassoRegressionLearner(t=self.t)
                c = l(tmpTable, nBoot=0, nPerm=0)
                coeffB.append(l.coefficients)
            self.stdErrorsFixedT = numpy.std(coeffB, axis=0)
        else:
            self.stdErrorsFixedT = [float("nan")] * m

        # permutation test to obtain the significance of the regression
        #coefficients
        if nPerm > 0:
            coeffP = []
            for i in range(nPerm):
                tmpTable = permute_responses(table)
                l = LassoRegressionLearner(t=self.t)
                c = l(tmpTable, nBoot=0, nPerm=0)
                coeffP.append(l.coefficients)
            self.pVals = \
                       numpy.sum(abs(numpy.array(coeffP))>\
                                 abs(numpy.array(self.coefficients)), \
                                 axis=0)/float(nPerm)
        else:
            self.pVals = [float("nan")] * m

        # dictionary of regression coefficients with standard errors
        # and p-values
        self.dictModel = {}
        for i, var in enumerate(self.domain.attributes):
            self.dictModel[var.name] = (self.coefficients[i], self.stdErrorsFixedT[i], self.pVals[i])            
       
        return LassoRegression(self)


class LassoRegression(Orange.classification.Classifier):
    """Lasso regression predicts value of the response variable
    based on the values of independent variables.

    .. attribute:: model
    
        fitted lasso regression model   

    """ 
    def __init__(self, model):
        """
        :param model: fitted lasso regression model
        :type model: :class:`LassoRegressionLearner`
        """
        self.model = model

    def __call__(self, instance,\
                 resultType=Orange.classification.Classifier.GetValue):
        """
        :param instance: data instance for which the value of the response
            variable will be predicted
        :type instance: 
        """  
        ins = Orange.data.Instance(self.model.domain, instance)
        if "?" in ins: # missing value -> corresponding coefficient omitted
            def miss_2_0(x): return x if x != "?" else 0
            ins = map(miss_2_0, ins)
            ins = numpy.array(ins)[:-1]-self.model.muX
        else:
            ins = numpy.array(ins.native())[:-1]-self.model.muX

        yHat = numpy.dot(self.model.coefficients, ins) + self.model.coef0 
        yHat = Orange.data.Value(yHat)
         
        if resultType == Orange.classification.Classifier.GetValue:
            return yHat
        if resultType == Orange.classification.Classifier.GetProbabilities:
            return Orange.statistics.distribution.Continuous({1.0: yHat})
        return (yHat, Orange.statistics.distribution.Continuous({1.0: yHat}))    



def print_lasso_regression_model(lr, skipZero=True):
    """Pretty-prints Lasso regression model,
    i.e. estimated regression coefficients with standard errors
    and significances. Standard errors are obtained using bootstrapping
    method and significances by the permuation test

    :param lr: a Lasso regression model object.
    :type lr: :class:`LassoRegression`
    :param skipZero: if True variables with estimated coefficient equal to 0
        are omitted
    :type skipZero: boolean
    """
    
    from string import join
    m = lr.model    
    labels = ('Variable', 'Coeff Est', 'Std Error', 'p')
    print join(['%10s' % l for l in labels], ' ')

    fmt = "%10s " + join(["%10.3f"]*3, " ") + " %5s"
    fmt1 = "%10s %10.3f"

    def get_star(p):
        if p < 0.001: return  "*"*3
        elif p < 0.01: return "*"*2
        elif p < 0.05: return "*"
        elif p < 0.1: return  "."
        else: return " "

    stars =  get_star(m.pVals[0])
    print fmt1 % ('Intercept', m.coef0)
    skipped = []
    for i in range(len(m.domain.attributes)):
        if m.coefficients[i] == 0. and skipZero:
            skipped.append(m.domain.attributes[i].name)
            continue            
        stars = get_star(m.pVals[i])
        print fmt % (m.domain.attributes[i].name, \
                     m.coefficients[i], m.stdErrorsFixedT[i], \
                     m.pVals[i], stars)
    print "Signif. codes:  0 *** 0.001 ** 0.01 * 0.05 . 0.1 empty 1"
    print "\n"
    if skipZero:
        k = len(skipped)
        if k == 0:
            print "All variables have non-zero regression coefficients. "
        else:
            suff = "s" if k > 1 else ""
            print "For %d variable%s the regression coefficient equals 0: " \
                  % (k, suff)
            for var in skipped:
                print var


if __name__ == "__main__":

    import Orange
    
    table = Orange.data.Table("housing.tab")        

    c = LassoRegressionLearner(table, t=len(table.domain))
    print_lasso_regression_model(c)