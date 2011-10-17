"""\
==========================================
Partial least sqaures regression (``PLS``)
==========================================

.. index:: regression

.. _`Parital Least Squares Regression`: http://en.wikipedia.org/wiki/Partial_least_squares_regression

Implementation is based on `Scikit learn python implementation`_

Example ::

    >>> import Orange
    >>>     from Orange.regression import pls
    >>> table = Orange.data.Table("test-pls.tab")
    >>> # set independent and response variables
    >>> x = [var for var in table.domain if var.name[0]=="X"]
    >>> y = [var for var in table.domain if var.name[0]=="Y"]
    >>> print x
        [FloatVariable 'X1', FloatVariable 'X2', FloatVariable 'X3']
    >>> print y
        [FloatVariable 'Y1', FloatVariable 'Y2', FloatVariable 'Y3', FloatVariable 'Y4']
    >>> # The information which variables are independent and which are responses
    >>> # can be provided in the data definition.
    >>> # If a variable var has key "label" in dictionary Orange.data.Domain[var].attributes
    >>> # it is considered as a response variable.
    >>> # In such situation x and y do not need to be specified.
    >>> l = pls.PLSRegressionLearner()
    >>> c = l(table, xVars=x, yVars=y)
    >>> c.print_pls_regression_coefficients()
           Y1     Y2     Y3     Y4     
    X1     0.513  0.915  0.341  -0.069  
    X2     0.641  -1.044  0.249  -0.015  
    X3     1.393  0.050  0.729  -0.108  
    >>> 

.. autoclass:: PLSRegressionLearner
    :members:

.. autoclass:: PLSRegression
    :members:

Utility functions
-----------------

.. autofunction:: normalize_matrix

.. autofunction:: nipals_xy

.. autofunction:: svd_xy

.. _`Scikit learn python implementation`: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pls.py

"""

import Orange
import numpy

from Orange.regression import base
from Orange.regression.earth import data_label_mask
from numpy import dot, zeros
from numpy.linalg import svd, inv

def normalize_matrix(X):
    """ Normalizes matrix, i.e. subtracts column means
    and divides them by column standard deviations.
    Returns the standardized matrix, sample mean and
    standard deviation

    :param X: data matrix
    :type X: :class:`numpy.array`
   
    """
    muX, sigmaX = numpy.mean(X, axis=0), numpy.std(X, axis=0)
    return (X-muX)/sigmaX, muX, sigmaX

def nipals_xy(X, Y, mode="PLS", maxIter=500, tol=1e-06):
    """
    NIPALS algorithm. Returns the first left and rigth singular
    vectors of X'Y.

    :param X, Y: data matrix
    :type X, Y: :class:`numpy.array`

    :param mode: possible values "PLS" (default) or "CCA" 
    :type mode: string

    :param maxIter: maximal number of iterations (default 500)
    :type maxIter: int

    :param tol: tolerance parameter, if norm of difference
        between two successive left singular vectors is less than tol,
        iteration is stopped
    :type tol: a not negative float
            
    """
    yScore, uOld, ite = Y[:, [0]], 0, 1
    Xpinv = Ypinv = None
    # Inner loop of the Wold algo.
    while True and ite < maxIter:
        # Update u: the X weights
        if mode == "CCA":
            if Xpinv is None:
                Xpinv = linalg.pinv(X) # compute once pinv(X)
            u = dot(Xpinv, yScore)
        else: # mode PLS
        # Mode PLS regress each X column on yScore
            u = dot(X.T, yScore) / dot(yScore.T, yScore)
        # Normalize u
        u /= numpy.sqrt(dot(u.T, u))
        # Update xScore: the X latent scores
        xScore = dot(X, u)

        # Update v: the Y weights
        if mode == "CCA":
            if Ypinv is None:
                Ypinv = linalg.pinv(Y) # compute once pinv(Y)
            v = dot(Ypinv, xScore)
        else:
            # Mode PLS regress each X column on yScore
            v = dot(Y.T, xScore) / dot(xScore.T, xScore)
        # Normalize v
        v /= numpy.sqrt(dot(v.T, v))
        # Update yScore: the Y latent scores
        yScore = dot(Y, v)

        uDiff = u - uOld
        if dot(uDiff.T, uDiff) < tol or Y.shape[1] == 1:
            break
        uOld = u
        ite += 1
    return u, v

def svd_xy(X, Y):
    """ Returns the first left and rigth singular
    vectors of X'Y.

    :param X, Y: data matrix
    :type X, Y: :class:`numpy.array`    
    """
    U, s, V = svd(dot(X.T, Y), full_matrices=False)
    u = U[:, [0]]
    v = V.T[:, [0]]
    return u, v


def select_attrs(table, attributes, class_var=None, metas=None):
    """ Select only ``attributes`` from the ``table``.
    """
    domain = Orange.data.Domain(attributes, class_var)
    if metas:
        domain.add_metas(metas)
    return Orange.data.Table(domain, table)


class PLSRegressionLearner(base.BaseRegressionLearner):
    """ Fits the partial least squares regression model,
    i.e. learns the regression parameters. The implementation is based on
    `Scikit learn python implementation`_
    
    The class is derived from
    :class:`Orange.regression.base.BaseRegressionLearner`
    which is used for preprocessing the data (continuization and imputation)
    before fitting the regression parameters

    Basic notations:
    n - number of data instances
    p - number of independent variables
    q - number of reponse variables

    .. attribute:: T
    
        A n x nComp numpy array of x-scores

    .. attribute:: U
    
        A n x nComp numpy array of y-scores

    .. attribute:: W
    
        A p x nComp numpy array of x-weights

    .. attribute:: C
    
        A q x nComp numpy array of y-weights

    .. attribute:: P
    
        A p x nComp numpy array of x-loadings

    .. attribute:: Q
    
        A q x nComp numpy array of y-loading

    .. attribute:: coefs
    
        A p x q numpy array coefficients
        of the linear model: Y = X coefs + E

    .. attribute:: xVars
    
        list of independent variables

    .. attribute:: yVars
    
        list of response variables 
        

    """

    def __init__(self, nComp=2, deflationMode="regression", mode="PLS",
                 algorithm="nipals", maxIter=500, 
                 imputer=None, continuizer=None,
                 **kwds):
        """
        .. attribute:: nComp
    
            number of components to keep. Default: 2

        .. attribute:: deflationMode
    
            "canonical" or "regression" (default)

        .. attribute:: mode
    
            "CCA" or "PLS" (default)


        .. attribute:: algorithm
    
            The algorithm used to estimate the weights:
            "nipals" or "svd" (default)


        """        
        self.nComp = nComp
        self.deflationMode = deflationMode
        self.mode = mode
        self.algorithm = algorithm
        self.maxIter = maxIter
        self.set_imputer(imputer=imputer)
        self.set_continuizer(continuizer=continuizer)
        self.__dict__.update(kwds)

    def __call__(self, table, weight_id=None, xVars=None, yVars=None):
        """
        :param table: data instances.
        :type table: :class:`Orange.data.Table`

        :param xVars, yVars: List of input and response variables
            (`Orange.data.variable.Continuous` or `Orange.data.variable.Discrete`).
            If None (default) it is assumed that data definition provides information
            which variables are reponses and which not. If a variable var
            has key "label" in dictionary Orange.data.Domain[var].attributes
            it is treated as a response variable
        :type xVars, yVars: list            

        """     
        domain = table.domain
        if xVars is None and yVars is None:
            # Response variables are defined in the table.
            label_mask = data_label_mask(domain)
            multilabel_flag = (sum(label_mask) - (1 if domain.class_var else 0)) > 0
            xVars = [v for v, label in zip(domain, label_mask) if not label]
            yVars = [v for v, label in zip(domain, label_mask) if label]
            x_table = select_attrs(table, xVars)
            y_table = select_attrs(table, yVars)
            
        elif xVars and yVars:
            # independent and response variables are passed by the caller
            if domain.class_var and domain.class_var not in yVars:
                # if the original table contains class variable
                # add it to the yVars
                yVars.append(domain.class_var)
            label_mask = [v in yVars for v in domain.variables]
            multilabel_flag = True
            x_table = select_attrs(table, xVars)
            y_table = select_attrs(table, yVars)
        else:
            raise ValueError("Both xVars and yVars must be defined.")
        
        # if independent and response variables are not listed in domain
#        if xVars is not None:
#            for var in xVars:
#                if table.domain[var].attributes.has_key("label"):
#                    del table.domain[var].attributes["label"]
#        if yVars is not None:
#            for var in yVars:
#                table.domain[var].attributes["label"] = True               

        # if the original table contains class variable        
#        if table.domain.class_var is not None:
#            oldClass = table.domain.class_var
#            newDomain = Orange.data.Domain(table.domain.variables, 0)
#            newDomain[oldClass].attributes["label"] = True
#            table = Orange.data.Table(newDomain, table)

        # dicrete values are continuized        
        x_table = self.continuize_table(x_table)
        y_table = self.continuize_table(y_table)
        # missing values are imputed
        x_table = self.impute_table(x_table)
        y_table = self.impute_table(y_table)
        
        # Collect the new transformed xVars/yVars 
        xVars = list(x_table.domain.variables)
        yVars = list(y_table.domain.variables)
        
        self.domain = Orange.data.Domain(xVars + yVars, False)
        label_mask = [False for _ in xVars] + [True for _ in yVars]
        
#        label_mask = data_label_mask(table.domain)
#        xy = table.toNumpy()[0]
#        y, x = xy[:, label_mask], xy[:, ~ label_mask]
#        self.yVars = [v for v, m in zip(self.domain.variables, label_mask) if m]
#        self.xVars = [v for v in self.domain.variables if v not in self.yVars]
        x = x_table.toNumpy()[0]
        y = y_table.toNumpy()[0]
        
        self.fit(x, y)
        return PLSRegression(label_mask=label_mask, domain=self.domain, \
                             coefs=self.coefs, muX=self.muX, muY=self.muY, \
                             sigmaX=self.sigmaX, sigmaY=self.sigmaY, \
                             xVars=xVars, yVars=yVars, multilabel_flag=multilabel_flag)

    def fit(self, X, Y):
        """ Fits all unknown parameters, i.e.
        weights, scores, loadings (for x and y) and regression coefficients.

        """
        # copy since this will contains the residuals (deflated) matrices

        X, Y = X.copy(), Y.copy()
        if Y.ndim == 1:
            Y = Y.reshape((Y.size, 1))
        n, p = X.shape
        q = Y.shape[1]

        # normalization of data matrices
        X, self.muX, self.sigmaX = normalize_matrix(X)
        Y, self.muY, self.sigmaY = normalize_matrix(Y)
        # Residuals (deflated) matrices
        Xk, Yk = X, Y
        # Results matrices
        self.T, self.U = zeros((n, self.nComp)), zeros((n, self.nComp))
        self.W, self.C = zeros((p, self.nComp)), zeros((q, self.nComp))
        self.P, self.Q = zeros((p, self.nComp)), zeros((q, self.nComp))      

        # NIPALS over components
        for k in xrange(self.nComp):
            # Weights estimation (inner loop)
            if self.algorithm == "nipals":
                u, v = nipals_xy(X=Xk, Y=Yk, mode=self.mode)
            elif self.algorithm == "svd":
                u, v = svd_xy(X=Xk, Y=Yk)
            # compute scores
            xScore, yScore = dot(Xk, u), dot(Yk, v)
            # Deflation (in place)
            # - regress Xk's on xScore
            xLoadings = dot(Xk.T, xScore) / dot(xScore.T, xScore)
            # - substract rank-one approximations to obtain remainder matrix
            Xk -= dot(xScore, xLoadings.T)
            if self.deflationMode == "canonical":
                # - regress Yk's on yScore, then substract rank-one approx.
                yLoadings = dot(Yk.T, yScore) / dot(yScore.T, yScore)
                Yk -= dot(yScore, yLoadings.T)
            if self.deflationMode == "regression":
                # - regress Yk's on xScore, then substract rank-one approx.
                yLoadings = dot(Yk.T, xScore) / dot(xScore.T, xScore)
                Yk -= dot(xScore, yLoadings.T)
            # Store weights, scores and loadings 
            self.T[:, k] = xScore.ravel() # x-scores
            self.U[:, k] = yScore.ravel() # y-scores
            self.W[:, k] = u.ravel() # x-weights
            self.C[:, k] = v.ravel() # y-weights
            self.P[:, k] = xLoadings.ravel() # x-loadings
            self.Q[:, k] = yLoadings.ravel() # y-loadings
        # X = TP' + E and Y = UQ' + E

        # Rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        self.xRotations = dot(self.W,
            inv(dot(self.P.T, self.W)))
        if Y.shape[1] > 1:
            self.yRotations = dot(self.C,
                inv(dot(self.Q.T, self.C)))
        else:
            self.yRotations = numpy.ones(1)

        if self.deflationMode == "regression":
            # Estimate regression coefficient
            # Y = TQ' + E = X W(P'W)^-1Q' + E = XB + E
            # => B = W*Q' (p x q)
            self.coefs = dot(self.xRotations, self.Q.T)
            self.coefs = 1. / self.sigmaX.reshape((p, 1)) * \
                    self.coefs * self.sigmaY
        return self

    """
    def transform(self, X, Y=None):

        # Normalize
        Xc = (X - self.muX) / self.sigmaX
        if Y is not None:
            Yc = (Y - self.muY) / self.sigmaY
        # Apply rotation
        xScores = dot(Xc, self.xRotations)
        if Y is not None:
            yScores = dot(Yc, self.yRotations)
            return xScores, yScores

        return xScores
    """


class PLSRegression(Orange.classification.Classifier):
    """ PLSRegression predicts value of the response variables
    based on the values of independent variables.
    
    """
    def __init__(self, label_mask=None, domain=None, \
                 coefs=None, muX=None, muY=None, sigmaX=None, sigmaY=None, \
                 xVars=None, yVars=None, multilabel_flag=0):
        self.label_mask = label_mask
        self.domain = domain
        self.coefs = coefs
        self.muX, self.muY = muX, muY
        self.sigmaX, self.sigmaY = sigmaX, sigmaY
        self.xVars, self.yVars = xVars, yVars
        self.multilabel_flag = multilabel_flag

    def __call__(self, instance, result_type=Orange.core.GetValue):
        """
        :param instance: data instance for which the value of the response
            variable will be predicted
        :type instance: :class:`Orange.data.Instance` 
        
        """ 
        instance = Orange.data.Instance(self.domain, instance)
        ins = [instance[v].native() for v in self.xVars]
        
        if "?" in ins: # missing value -> corresponding coefficient omitted
            def miss_2_0(x): return x if x != "?" else 0
            ins = map(miss_2_0, ins)
        ins = numpy.array(ins)
        xc = (ins - self.muX) / self.sigmaX
        predicted = dot(xc, self.coefs) * self.sigmaY + self.muY
        y_hat = [var(val) for var, val in zip(self.yVars, predicted)]
        if result_type == Orange.core.GetValue:
            return y_hat if self.multilabel_flag else y_hat[0]
        else:
            from Orange.statistics.distribution import Distribution
            probs = []
            for var, val in zip(self.yVars, y_hat):
                dist = Distribution(var)
                dist[val] = 1.0
                probs.append(dist)
            if result_type == Orange.core.GetBoth:
                return (y_hat, probs) if self.multilabel_flag else (y_hat[0], probs[0])
            else:
                return probs if self.multilabel_flag else probs[0]
            
    def print_pls_regression_coefficients(self):
        """ Pretty-prints the coefficient of the PLS regression model.
        """       
        xVars, yVars = [x.name for x in self.xVars], [y.name for y in self.yVars]
        print " " * 7 + "%-6s " * len(yVars) % tuple(yVars)
        fmt = "%-6s " + "%-5.3f  " * len(yVars)
        for i, coef in enumerate(self.coefs):
            print fmt % tuple([xVars[i]] + list(coef))  
    

if __name__ == "__main__":

    import Orange
    from Orange.regression import pls

    table = Orange.data.Table("test-pls.tab")
    l = pls.PLSRegressionLearner()

    x = [var for var in table.domain if var.name[0]=="X"]
    y = [var for var in table.domain if var.name[0]=="Y"]
    print x, y
#    c = l(table, xVars=x, yVars=y)
    c = l(table)
    c.print_pls_regression_coefficients()